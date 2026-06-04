"""
    AbstractBoundaryCondition

Supertype for the rule that fills a physical-edge ghost layer when the halo is
refreshed (by [`boundary_condition!`](@ref) / [`synchronize_halo!`](@ref)).

Concrete conditions: [`Periodic`](@ref), [`Repeating`](@ref),
[`Reflecting`](@ref), [`Antireflecting`](@ref), [`NoBoundaryCondition`](@ref).
A condition is given per `(dimension, side)`; constructors also accept the
symbols `:periodic`, `:repeating`, `:reflecting`, `:antireflecting`.
"""
abstract type AbstractBoundaryCondition end

"Even/mirror reflection about the wall: `ghost = +interior` (zero-gradient)."
struct Reflecting       <: AbstractBoundaryCondition end
"""Odd reflection about the wall: `ghost = -interior` (the field vanishes at the
wall) — e.g. a velocity component normal to a solid wall. Cf. [`Reflecting`](@ref)."""
struct Antireflecting  <: AbstractBoundaryCondition end
"Zero-gradient: each ghost cell copies the nearest owned value."
struct Repeating       <: AbstractBoundaryCondition end
"Ghost cells wrap around from the opposite side of the domain (periodic)."
struct Periodic        <: AbstractBoundaryCondition end
"Leave ghost cells untouched; the caller fills them manually (e.g. a custom or characteristic BC)."
struct NoBoundaryCondition <: AbstractBoundaryCondition end

"""
    Side{S}()  (S = 1 low, 2 high);  Side(s::Int)

Type-level tag for the low (`1`) or high (`2`) end of a dimension, used to
dispatch boundary/halo operations. See [`Dim`](@ref).
"""
struct Side{S}; end
@inline Side(s::Int) = Side{s}()

"""
    Dim{D}();  Dim(d::Int)

Type-level tag for spatial dimension `D`, used to dispatch boundary/halo
operations on a specific axis. See [`Side`](@ref).
"""
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
"""
    HaloArray(T, owned_dims, halo, topology; boundary_condition=:repeating)
    HaloArray(T, owned_dims, halo; boundary_condition=:repeating)   # builds a topology

The MPI-distributed halo array. Each rank owns an `owned_dims` patch of the
global grid surrounded by `halo` ghost cells; [`synchronize_halo!`](@ref)
exchanges those ghosts with neighbouring ranks over an MPI Cartesian topology
and applies the boundary condition at the physical domain edges.

`owned_dims` is the size of *this rank's* patch — the global grid is
`owned_dims .* topology.dims`. Reductions (`sum`, `dot`, `norm`, …) are global
(MPI `Allreduce`); scalar indexing is local-only and warns.

# Arguments
- `T`: element type (defaults to `Float64`).
- `owned_dims::NTuple{N,Int}`: this rank's owned (interior) extent.
- `halo::Int`: ghost-cell width on each side.
- `topology`: a [`CartesianTopology`](@ref). If omitted, one is built over
  `MPI.COMM_WORLD` with periodicity inferred from the boundary condition.
- `boundary_condition`: applied at physical edges (see [`LocalHaloArray`](@ref)
  for the accepted forms).

See also [`LocalHaloArray`](@ref), [`ThreadedHaloArray`](@ref),
[`halo_exchange!`](@ref), [`gather`](@ref).
"""
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

"""
    halo_width(u) -> Int

The ghost-cell width on each side (the same in every dimension). Encoded in the
type, so it is a compile-time constant. Also callable on the type.
"""
@inline halo_width(::HaloArray{T,N,A,Halo}) where {T,N,A,Halo} = Halo
@inline halo_width(::Type{<:HaloArray{T,N,A,Halo}}) where {T,N,A,Halo} = Halo

# eltype/ndims come from AbstractArray{T,N}; parent from AbstractSingleHaloArray.

"""
    storage_size(u[, i]) -> dims

Size of the backing storage **including** ghost padding (i.e. `owned + 2*halo`
per dimension); for [`ThreadedHaloArray`](@ref) this is the per-tile storage.
Contrast with [`owned_size`](@ref) (ghost-free) and [`global_size`](@ref).
"""
@inline storage_size(halo::HaloArray)         = size(halo.data)
@inline storage_size(halo::HaloArray, i::Int) = size(halo.data, i)

@inline function interior_size(halo::HaloArray{T,N,A,Halo}) where {T,N,A,Halo}
    ntuple(i -> size(halo.data, i) - 2*Halo, Val(N))
end

"""
    interior_range(u) -> NTuple of UnitRanges

The index ranges of the owned cells **within the padded storage** (`(halo+1) :
(size-halo)` per dimension). Use it to index `parent(u)` in a stencil with
ghost-safe offsets, e.g. `for I in CartesianIndices(interior_range(u))`. For
[`ThreadedHaloArray`](@ref) the range is the same for every tile.
See also [`interior_view`](@ref).
"""
@inline function interior_range(halo::HaloArray{T,N,A,Halo}) where {T,N,A,Halo}
    ntuple(i -> (Halo+1):(storage_size(halo,i)-Halo), Val(N))
end

"""
    interior_view(u) -> view

A mutable view of `u`'s owned (ghost-free) cells. The usual way to read/write
initial data: `interior_view(u) .= ...`. For [`ThreadedHaloArray`](@ref) pass a
tile id: `interior_view(u, tile_id)`. See also [`interior_range`](@ref).
"""
@inline function interior_view(halo::HaloArray)
    @views halo.data[interior_range(halo)...]
end

# size, axes, length, owned_axes, eachindex, iterate, versors, similar dispatchers,
# map!/map inherited from AbstractSingleHaloArray

# ---- global / topology accessors (pure field access, no MPI calls) ----

"""
    global_size(u) -> dims

Size of the **whole** grid across all ranks / tiles (`owned_size .* topology.dims`).
For [`LocalHaloArray`](@ref) it equals [`owned_size`](@ref); for [`HaloArray`](@ref)
(MPI) and [`ThreadedHaloArray`](@ref) it is larger than this rank's/tile's share.
"""
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
# Along one dimension D (storage size `sd = owned + 2*halo`), the storage is
#
#     index:   1 .. halo | halo+1 ..  sd-halo  | sd-halo+1 .. sd
#              ghost(lo)  |   owned (interior)  |    ghost(hi)
#
# Two width-`halo` slabs matter at each boundary side:
#   • the GHOST slab  — the cells to WRITE   → `_recv_window` ("receive")
#   • the EDGE slab   — the `halo` owned cells touching that ghost slab,
#                       the cells to READ    → `_send_window` ("send")
#
#   Side 1 (low):   ghost = 1:halo            edge = halo+1 : 2*halo
#   Side 2 (high):  ghost = sd-halo+1 : sd    edge = sd-2*halo+1 : sd-halo
#
# `get_send_view`/`get_recv_view` return *views* (writing into a recv view
# mutates the array) of width `halo` along D, spanning the OWNED extent in
# every other dimension. They behave identically for HaloArray (MPI) and
# LocalHaloArray.
#
# Halo exchange uses them as the names suggest: copy this rank's `send` slab to
# a neighbour, receive into the `recv` slab. For a *physical* boundary condition
# the same views are the natural primitives: write `get_recv_view(s, d, field)`
# (the ghosts) from `get_send_view(s, d, field)` (the adjacent interior edge) —
# both same-shaped, so per-cell it is just `recv[k] = g(edge[k])`. The two slabs
# straddle the boundary, so their indices run TOWARD each other: on side 1 the
# innermost ghost recv[halo] is adjacent to the edge cell send[1] (a mirror BC
# pairs recv[halo+1-k] with send[k]; a zeroth-order copy fills every ghost from
# the edge cell send[1]). A coupled BC across several fields does the same, but
# gathers send-edge values from all fields, transforms them together, and writes
# the result into each field's recv slab.

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

# fill_interior!, fill_from_local_indices!, Base.foreach, arithmetic,
# LinearAlgebra.norm inherited from AbstractSingleHaloArray

# Base.map!/map inherited from AbstractSingleHaloArray


# ---- fill helpers -----------------------------------------------------

"""
    fill_from_global_indices!(f, u)

Set each owned cell from `f(I, J, …)` evaluated at its **global** grid index
(1-based over the whole domain). Each rank/tile fills only the cells it owns, so
the same `f` produces a consistent global field across MPI ranks and threads —
the idiomatic way to set an initial condition. Cf. [`fill_from_local_indices!`](@ref).

# Example
```julia
fill_from_global_indices!(u) do i, j
    exp(-((i - nx/2)^2 + (j - ny/2)^2) / 50)
end
```
"""
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
