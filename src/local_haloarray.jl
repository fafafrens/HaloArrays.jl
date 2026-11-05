"""
    LocalHaloArray(T, owned_dims, halo; boundary_condition=:repeating, storage=zeros)
    LocalHaloArray(data::AbstractArray, halo, boundary_condition)

A single-process halo array: a plain array padded with `halo` ghost cells on
every side of the `owned_dims` interior, carrying the boundary condition to
apply when the halo is refreshed.

This is the simplest backend — no MPI, no threads. Logical indexing, `axes`,
and `eachindex` address the interior cells only; the ghost padding is hidden.

# Arguments
- `T`: element type (defaults to `Float64` if omitted).
- `owned_dims::NTuple{N,Int}`: size of the interior region.
- `halo::Int`: ghost-cell width on each side, in each dimension.
- `boundary_condition`: how ghosts are filled by [`synchronize_halo!`](@ref) /
  [`boundary_condition!`](@ref). A symbol (`:periodic`, `:repeating`,
  `:reflecting`, `:antireflecting`), a boundary-condition instance, or a
  per-dimension tuple of `(left, right)` pairs. See [`Periodic`](@ref) etc.
- `storage`: the allocator for the padded block, called as
  `storage(T, sizes...)`. `zeros` by default; `CUDA.zeros` / `Metal.zeros`
  places the array on a device.

# Examples
```julia
u = LocalHaloArray(Float64, (64, 64), 1; boundary_condition=:periodic)
interior_view(u) .= 1.0      # write the interior cells
synchronize_halo!(u)         # fill ghost cells from the boundary condition
g = LocalHaloArray(Float32, (64, 64), 1; boundary_condition=:periodic, storage=Metal.zeros)
```

See also [`HaloArray`](@ref) (MPI), [`ThreadedHaloArray`](@ref) (threads),
[`interior_view`](@ref), [`interior_range`](@ref).
"""
mutable struct LocalHaloArray{T,N,A,Halo,BCondition} <: AbstractSerialHaloArray{T,N}
    data::A
    boundary_condition::BCondition
end

@inline halo_backend(::Type{<:LocalHaloArray}) = LocalHaloBackend()

function LocalHaloArray(data::AbstractArray{T,N}, halo::Int, boundary_condition) where {T,N}
    bc = normalize_boundary_condition(boundary_condition, N)
    _check_periodic_extent(size(data), halo, bc)
    return LocalHaloArray{T,N,typeof(data),halo,typeof(bc)}(data, bc)
end

# A periodic ghost slab is a copy of the opposite interior edge, `halo` wide:
# with fewer interior cells than that the "edge" reaches into the ghosts and
# the fill reads what it writes. Reject the geometry rather than fill garbage.
function _check_periodic_extent(fullsize::NTuple{N,Int}, halo::Int, bc) where {N}
    for d in 1:N
        (bc[d][1] isa Periodic || bc[d][2] isa Periodic) || continue
        interior = fullsize[d] - 2halo
        interior >= halo || throw(ArgumentError(
            "periodic dimension $d has $interior interior cells, fewer than the halo width $halo"))
    end
    return nothing
end

function LocalHaloArray(::Type{T}, owned_dims::NTuple{N,<:Integer}, halo::Integer;
        boundary_condition=:repeating, storage=zeros) where {T,N}
    return LocalHaloArray(_padded_storage(storage, T, owned_dims, halo), Int(halo), boundary_condition)
end

LocalHaloArray(owned_dims::NTuple{N,<:Integer}, halo::Integer; kwargs...) where {N} =
    LocalHaloArray(Float64, owned_dims, halo; kwargs...)

# size, axes, length, eltype, ndims, parent, interior_axes, eachindex, iterate
# inherited from AbstractSingleHaloArray / AbstractArray.

is_root(::LocalHaloArray; root::Integer=0) = (root == 0)
# is_active, communicator inherited from AbstractSerialHaloArray

@inline halo_width(::Type{<:LocalHaloArray{T,N,A,Halo}}) where {T,N,A,Halo} = Halo

# storage_size / interior_size / interior_range come from AbstractSingleHaloArray.

# ---- the single-block kinds: one storage block per process ------------
# LocalHaloArray and HaloArray differ in their topology (none / MPI) and in
# nothing else about the block they hold: the view accessors, scalar indexing
# and `show` are written once on the union (the broadcast styles likewise).
const _SingleBlock{T,N} = Union{HaloArray{T,N},LocalHaloArray{T,N}}

# the block origin of a LocalHaloArray is the global origin (haloarray.jl)
@inline _block_origin(::LocalHaloArray{T,N}, _) where {T,N} = ntuple(_ -> 1, Val(N))

"""
    interior_view(u) -> view

A mutable view of `u`'s interior (ghost-free) cells. The usual way to read/write
initial data: `interior_view(u) .= ...`. For [`ThreadedHaloArray`](@ref) pass a
tile id: `interior_view(u, tile_id)`. See also [`interior_range`](@ref).
"""
@inline interior_view(halo::_SingleBlock) = @views halo.data[interior_range(halo)...]

@inline edge_view(a::_SingleBlock, s::Side, d::Dim)  = edge_view(parent(a), s, d, halo_width(a))
@inline ghost_view(a::_SingleBlock, s::Side, d::Dim) = ghost_view(parent(a), s, d, halo_width(a))
# versors, Base.similar dispatchers, Base.map!/map inherited from AbstractSingleHaloArray

function Base.similar(halo::LocalHaloArray{T,N,A,Halo,BCondition}, ::Type{AA},
        dims::Dims{M}) where {T,N,A,Halo,BCondition,AA,M}
    M == N || throw(DimensionMismatch("LocalHaloArray similar dims must have $N dimensions"))
    fullsize = ntuple(i -> Int(dims[i]) + 2 * halo_width(halo), Val(N))
    data = similar(parent(halo), AA, fullsize)
    return LocalHaloArray(data, halo_width(halo), halo.boundary_condition)
end

# Non-Int dims are normalized to Dims by Base's generic similar fallbacks.

# Base.copy, Base.zero, Base.fill!, Base.copyto!,
# Base.foreach, arithmetic, norm,
# interior_axes, eachindex, iterate, versors, Base.similar dispatchers,
# Base.map!/map — all inherited from AbstractSingleHaloArray

interior_to_global_index(::LocalHaloArray, owned_idx::NTuple{N,<:Integer}) where {N} = owned_idx
global_to_storage_index(halo::LocalHaloArray, global_idx::NTuple{N,<:Integer}) where {N} =
    all(i -> 1 <= global_idx[i] <= interior_size(halo, i), 1:N) ? ntuple(i -> global_idx[i] + halo_width(halo), Val(N)) : nothing
_global_size(halo::LocalHaloArray) = interior_size(halo)

# Scalar indexing by GLOBAL index on the single-block kinds. The explicit
# `nothing` check narrows the Union return before the splat (without it every
# access allocated 80 bytes through a boxed splat). An index this process does
# not hold is a BoundsError on a LocalHaloArray; on an MPI HaloArray, whose
# scalar indexing is for diagnostics only (a global index is held by one rank),
# an ArgumentError naming the rank-local semantics.
@inline function _storage_index(halo::_SingleBlock, I)
    idx = _check_global_scalar_indices(halo, I)
    storage_idx = global_to_storage_index(halo, idx)
    storage_idx === nothing && _not_owned(halo, idx)
    return storage_idx
end
_not_owned(halo::LocalHaloArray, idx) = throw(BoundsError(halo, idx))
_not_owned(::HaloArray, idx) = throw(ArgumentError(
    "Global index $idx is not in this MPI rank's interior region; HaloArray scalar indexing is local-only."))
# One `@warn` site per operation: `maxlog` counts per source location, so a
# shared site would warn for the first of the two and never for the other.
_index_note(::LocalHaloArray, ::Val) = nothing
_index_note(::HaloArray, ::Val{:getindex}) =
    @warn "Global scalar getindex on HaloArray is local-only (diagnostics only, not for hot loops)." maxlog=1
_index_note(::HaloArray, ::Val{:setindex!}) =
    @warn "Global scalar setindex! on HaloArray is local-only (diagnostics only, not for hot loops)." maxlog=1

function Base.getindex(halo::_SingleBlock, I::Vararg{Integer})
    _index_note(halo, Val(:getindex))
    @inbounds return parent(halo)[_storage_index(halo, I)...]
end

function Base.setindex!(halo::_SingleBlock, value, I::Vararg{Integer})
    _index_note(halo, Val(:setindex!))
    @inbounds parent(halo)[_storage_index(halo, I)...] = value
    return halo
end

_show_topology(io::IO, ::LocalHaloArray) = nothing
_show_topology(io::IO, u::HaloArray) = println(io, "  topology: ", u.topology)

function Base.show(io::IO, u::_SingleBlock)
    println(io, nameof(typeof(u)), " of global size ", size(u), " (interior size: ", interior_size(u),
            ", storage size: ", storage_size(u), "), halo width: ", halo_width(u))
    println(io, "  eltype: ", eltype(u))
    _show_topology(io, u)
    println(io, "  boundary_condition: ", u.boundary_condition)
end

function Base.show(io::IO, mime::MIME"text/plain", u::_SingleBlock)
    println(io, nameof(typeof(u)), " (storage size: ", storage_size(u), ", halo width: ", halo_width(u), ")")
    println(io, "  eltype: ", eltype(u))
    _show_topology(io, u)
    println(io, "  boundary_condition: ", u.boundary_condition)
    println(io, "  interior data preview:")
    show(io, mime, interior_view(u))
end
