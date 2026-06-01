abstract type AbstractBoundaryCondition end

struct Reflecting    <: AbstractBoundaryCondition end
struct Antireflecting <: AbstractBoundaryCondition end
struct Repeating     <: AbstractBoundaryCondition end
struct Periodic      <: AbstractBoundaryCondition end

struct Side{S}; end
@inline Side(s::Int) = Side{s}()

struct Dim{D}; end
@inline Dim(d::Int) = Dim{d}()

# HaloArray type params:
#   T           element type
#   N           spatial dimensions
#   A           storage array type
#   Halo        halo width (compile-time Int encoded as type param)
#   B           buffer type (recv_bufs / send_bufs)
#   BCondition  boundary condition type
#   Topo        topology type (CartesianTopology{N,C} when MPI is loaded)
#   CS          comm-state type (HaloCommState{N} when MPI is loaded)
mutable struct HaloArray{T,N,A,Halo,B,BCondition,Topo,CS} <: AbstractDistributedHaloArray{T,N}
    data::A
    topology::Topo
    comm_state::CS
    receive_bufs::B
    send_bufs::B
    boundary_condition::BCondition
end

@inline halo_backend(::Type{<:HaloArray}) = MPIHaloBackend()

# ---- basic accessors --------------------------------------------------

@inline halo_width(::HaloArray{T,N,A,Halo}) where {T,N,A,Halo} = Halo
@inline halo_width(::Type{<:HaloArray{T,N,A,Halo}}) where {T,N,A,Halo} = Halo

@inline Base.ndims(::HaloArray{T,N}) where {T,N} = N
@inline Base.ndims(::Type{<:HaloArray{T,N}}) where {T,N} = N

@inline Base.eltype(::HaloArray{T}) where {T} = T
@inline Base.eltype(::Type{<:HaloArray{T}}) where {T} = T

@inline Base.parent(halo::HaloArray) = halo.data

@inline storage_size(halo::HaloArray)         = size(halo.data)
@inline storage_size(halo::HaloArray, i::Int) = size(halo.data, i)

@inline function interior_size(halo::HaloArray{T,N,A,Halo}) where {T,N,A,Halo}
    ntuple(i -> size(halo.data, i) - 2*Halo, Val(N))
end

@inline function interior_range(halo::HaloArray{T,N,A,Halo}) where {T,N,A,Halo}
    ntuple(i -> (Halo+1):(storage_size(halo,i)-Halo), Val(N))
end

@inline function interior_view(halo::HaloArray)
    @views halo.data[interior_range(halo)...]
end

# size, axes, length, owned_axes, eachindex, iterate, versors, similar dispatchers,
# map!/map inherited from AbstractSingleHaloArray

# ---- global / topology accessors (pure field access, no MPI calls) ----

function global_size(halo::HaloArray{T,N}) where {T,N}
    local_interior = owned_size(halo)
    dims = halo.topology.dims
    ntuple(i -> local_interior[i] * dims[i], Val(N))
end

@inline get_comm(halo::HaloArray) = halo.topology.cart_comm
@inline isactive(a::HaloArray)    = isactive(a.topology)
@inline is_root(a::HaloArray; root::Integer=0) = is_root(a.topology; root=root)

function owned_to_global_index(halo::HaloArray{T,N}, owned_idx::NTuple{N,<:Integer}) where {T,N}
    coords     = halo.topology.cart_coords
    owned_dims = interior_size(halo)
    all(i -> 1 <= owned_idx[i] <= owned_dims[i], 1:N) ||
        throw(BoundsError(halo, owned_idx))
    ntuple(i -> coords[i]*owned_dims[i] + owned_idx[i], Val(N))
end

function global_to_storage_index(halo::HaloArray{T,N}, global_idx::NTuple{N,<:Integer}) where {T,N}
    owned_dims   = interior_size(halo)
    coords       = halo.topology.cart_coords
    h            = halo_width(halo)
    owner_coords = ntuple(i -> (global_idx[i]-1) ÷ owned_dims[i], Val(N))
    owner_coords != coords && return nothing
    interior_idx = ntuple(i -> global_idx[i] - coords[i]*owned_dims[i], Val(N))
    ntuple(i -> interior_idx[i] + h, Val(N))
end

@inline function _owned_global_to_storage_index(halo::HaloArray, I)
    idx = _check_global_scalar_indices(halo, I)
    storage_idx = global_to_storage_index(halo, idx)
    storage_idx === nothing &&
        throw(ArgumentError("Global index $idx is not owned by this MPI rank; HaloArray scalar indexing is local-only."))
    return storage_idx
end

function Base.getindex(halo::HaloArray, I::Vararg{Integer})
    @warn "Global scalar getindex on HaloArray is local-only (diagnostics only, not for hot loops)." maxlog=1
    storage_idx = _owned_global_to_storage_index(halo, I)
    @inbounds return parent(halo)[storage_idx...]
end

function Base.setindex!(halo::HaloArray, value, I::Vararg{Integer})
    @warn "Global scalar setindex! on HaloArray is local-only (diagnostics only, not for hot loops)." maxlog=1
    storage_idx = _owned_global_to_storage_index(halo, I)
    @inbounds parent(halo)[storage_idx...] = value
    return halo
end

# ---- versors ----------------------------------------------------------

@inline function versors(::Val{N}) where {N}
    ntuple(i -> ntuple(j -> ifelse(i==j, 1, 0), Val(N)), Val(N))
end

@inline versors(::HaloArray{T,N}) where {T,N} = versors(Val(N))

# ---- send/recv buffer views -------------------------------------------
#
# A halo exchange copies a slab of width `halo` along the exchange dimension
# D, spanning the owned (non-halo) extent in every other dimension. The only
# thing that differs across the send/recv × side combinations is the index
# window along D, captured by `_send_window` / `_recv_window`.

@inline _send_window(::Side{1}, sd::Int, halo::Int) = (halo+1):(2*halo)
@inline _send_window(::Side{2}, sd::Int, halo::Int) = (sd-2*halo+1):(sd-halo)
@inline _recv_window(::Side{1}, sd::Int, halo::Int) = 1:halo
@inline _recv_window(::Side{2}, sd::Int, halo::Int) = (sd-halo+1):sd

# Select `window` along dimension D and the owned span (halo+1 : size-halo)
# in every other dimension.
@inline function _halo_window_view(window, arr::AbstractArray{<:Any,N}, D::Integer, halo::Int) where {N}
    view(arr, ntuple(I -> I == D ? window : (halo+1):(size(arr,I)-halo), Val(N))...)
end

# HaloArray dispatch — both compile-time `Dim{D}` and runtime `D::Int` are used
# (the MPI exchange path passes a runtime dimension).
@inline get_send_view(s::Side, ::Dim{D}, a::HaloArray{T,N,A,Halo}) where {D,T,N,A,Halo} =
    _halo_window_view(_send_window(s, storage_size(a, D), Halo), parent(a), D, Halo)
@inline get_send_view(s::Side, D::Int, a::HaloArray{T,N,A,Halo}) where {T,N,A,Halo} =
    _halo_window_view(_send_window(s, storage_size(a, D), Halo), parent(a), D, Halo)
@inline get_recv_view(s::Side, ::Dim{D}, a::HaloArray{T,N,A,Halo}) where {D,T,N,A,Halo} =
    _halo_window_view(_recv_window(s, storage_size(a, D), Halo), parent(a), D, Halo)
@inline get_recv_view(s::Side, D::Int, a::HaloArray{T,N,A,Halo}) where {T,N,A,Halo} =
    _halo_window_view(_recv_window(s, storage_size(a, D), Halo), parent(a), D, Halo)

# Plain-array dispatch (used during buffer construction, before the HaloArray exists).
@inline get_send_view(s::Side, ::Dim{D}, arr::AbstractArray, halo::Int) where {D} =
    _halo_window_view(_send_window(s, size(arr, D), halo), arr, D, halo)
@inline get_recv_view(s::Side, ::Dim{D}, arr::AbstractArray, halo::Int) where {D} =
    _halo_window_view(_recv_window(s, size(arr, D), halo), arr, D, halo)

# ---- buffer allocation ------------------------------------------------

function make_recv_buffers(data::AbstractArray{T,N}, halo::Int) where {T,N}
    ntuple(D -> ntuple(S -> similar(get_recv_view(Side(S), Dim(D), data, halo)), Val(2)), Val(N))
end

function make_send_buffers(data::AbstractArray{T,N}, halo::Int) where {T,N}
    ntuple(D -> ntuple(S -> similar(get_send_view(Side(S), Dim(D), data, halo)), Val(2)), Val(N))
end

# validate_boundary_condition is inherited from AbstractCartesianTopology (abstract_haloarray.jl)

# ---- owned-dims helper (used by Base.similar in mpi_support.jl) ------

function _global_to_owned_dims(halo::HaloArray{T,N}, dims::NTuple{M,<:Integer}) where {T,N,M}
    M == N || throw(DimensionMismatch("HaloArray similar dims must have $N dimensions"))
    topo_dims = isactive(halo) ? halo.topology.dims : ntuple(_ -> 1, Val(N))
    all(d -> Int(dims[d]) % topo_dims[d] == 0, 1:N) ||
        throw(DimensionMismatch("HaloArray global similar dims $dims not divisible by topology dims $topo_dims"))
    ntuple(d -> Int(dims[d]) ÷ topo_dims[d], Val(N))
end

# ---- mutation ---------------------------------------------------------

# Base.copy, Base.zero, Base.fill!, Base.copyto! inherited from AbstractSingleHaloArray

# fill_interior, fill_from_local_indices!, Base.foreach, arithmetic,
# LinearAlgebra.norm inherited from AbstractSingleHaloArray

# Base.map!/map inherited from AbstractSingleHaloArray


# ---- fill helpers -----------------------------------------------------

function fill_from_global_indices!(f, halo::HaloArray{T,N,A,Halo}) where {T,N,A,Halo}
    local_shape = interior_range(halo)
    for local_I in CartesianIndices(local_shape)
        full_I = Tuple(local_I)
        local_interior_I = ntuple(i -> full_I[i]-Halo, Val(N))
        global_I = owned_to_global_index(halo, local_interior_I)
        halo.data[local_I] = f(global_I)
    end
    return halo
end


# ---- show -------------------------------------------------------------

function Base.show(io::IO, obj::HaloArray)
    print(io, "HaloArray of global size ", size(obj),
          " (owned: ", owned_size(obj), ", storage: ", storage_size(obj),
          "), halo=", halo_width(obj), "\n")
    print(io, "  eltype: ", eltype(obj), "\n")
    print(io, "  topology: ", obj.topology, "\n")
    print(io, "  boundary_condition: ", obj.boundary_condition, "\n")
end

function Base.show(io::IO, mime::MIME"text/plain", obj::HaloArray)
    println(io, "HaloArray (storage: ", storage_size(obj), ", halo=", halo_width(obj), ")")
    println(io, "  eltype: ", eltype(obj))
    println(io, "  topology: ", obj.topology)
    println(io, "  boundary_condition: ", obj.boundary_condition)
    println(io, "  interior data preview:")
    show(io, mime, interior_view(obj))
end
