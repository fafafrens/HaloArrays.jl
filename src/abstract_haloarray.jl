"""
    AbstractCartesianTopology{N}

Common supertype for N-dimensional Cartesian decomposition topologies.

Concrete subtypes:
- `CartesianTopology{N,C}` — MPI-backed distributed topology
- `ThreadedCartesianTopology{N}` — shared-memory tiled topology

Shared interface (all subtypes must implement):
- `Base.ndims(::AbstractCartesianTopology{N})` → N
- `isactive(topology)` → Bool
- `is_root(topology; root=0)` → Bool
- `topology.dims` → NTuple{N,Int}
- `topology.periodic_boundary_condition` → NTuple{N,Bool}
"""
abstract type AbstractCartesianTopology{N} end

abstract type AbstractHaloArray{T,N} <: AbstractArray{T,N} end

abstract type AbstractSingleHaloArray{T,N} <: AbstractHaloArray{T,N} end
abstract type AbstractDistributedHaloArray{T,N} <: AbstractSingleHaloArray{T,N} end
abstract type AbstractSerialHaloArray{T,N} <: AbstractSingleHaloArray{T,N} end

abstract type AbstractHaloCollection{T,N,S} <: AbstractHaloArray{T,N} end

abstract type AbstractHaloBackend end
struct MPIHaloBackend <: AbstractHaloBackend end
struct LocalHaloBackend <: AbstractHaloBackend end
struct ThreadedHaloBackend <: AbstractHaloBackend end

"""
    halo_backend(x) -> AbstractHaloBackend

Return a singleton trait describing the storage/execution backend of a halo
array or halo collection. Use this for dispatch when code needs separate MPI,
local, or threaded implementations while still accepting collection wrappers.
"""
function halo_backend end

@inline halo_backend(halo::AbstractSingleHaloArray) = halo_backend(typeof(halo))

# Single halo arrays all store their backing array in a `data` field.
# (eltype/ndims come from AbstractArray{T,N}.)
@inline Base.parent(halo::AbstractSingleHaloArray) = halo.data

# Serial backends (local + threaded) are always active and have no communicator.
@inline isactive(::AbstractSerialHaloArray) = true
@inline get_comm(::AbstractSerialHaloArray) = nothing

# Storage geometry for contiguous single arrays (one padded `data` array).
# ThreadedHaloArray is tiled and overrides all three with per-tile versions.
"""
    storage_size(u[, i]) -> dims

Size of the backing storage **including** ghost padding (`owned + 2*halo` per
dimension); for [`ThreadedHaloArray`](@ref) this is the per-tile storage.
Contrast with [`owned_size`](@ref) (ghost-free) and [`global_size`](@ref).
"""
@inline storage_size(halo::AbstractSingleHaloArray)         = size(parent(halo))
@inline storage_size(halo::AbstractSingleHaloArray, i::Int) = size(parent(halo), i)

@inline function interior_size(halo::AbstractSingleHaloArray)
    hw = halo_width(halo)
    return ntuple(i -> size(parent(halo), i) - 2hw, Val(ndims(halo)))
end

"""
    interior_range(u) -> NTuple of UnitRanges

The index ranges of the owned cells **within the padded storage** (`(halo+1) :
(size-halo)` per dimension). Use it to index `parent(u)` in a stencil with
ghost-safe offsets, e.g. `for I in CartesianIndices(interior_range(u))`. For
[`ThreadedHaloArray`](@ref) the range is the same for every tile.
See also [`interior_view`](@ref).
"""
@inline function interior_range(halo::AbstractSingleHaloArray)
    hw = halo_width(halo)
    return ntuple(i -> (hw + 1):(size(parent(halo), i) - hw), Val(ndims(halo)))
end

"""
    owned_size(halo)

Return the owned interior size of a halo container on the current process.

For `HaloArray` this is the owned MPI subdomain size, not the global
distributed size. For serial containers it is equal to their full logical
interior size.
"""
@inline owned_size(halo::AbstractHaloArray) = interior_size(halo)
@inline owned_size(halo::AbstractHaloArray, i::Int) = owned_size(halo)[i]

"""
    owned_axes(halo)

Return the owned-cell axes of a halo container on the current process.

Use `axes(halo)` for the global logical axes and `owned_axes(halo)` when
looping over data that this process can update directly.
"""
@inline owned_axes(halo::AbstractHaloArray) = map(Base.OneTo, owned_size(halo))
@inline owned_axes(halo::AbstractHaloArray, i::Int) = owned_axes(halo)[i]

function storage_size end
function owned_to_global_index end
function global_to_storage_index end
function is_root end

@inline halo_width(arr::AbstractArray{<:AbstractSingleHaloArray}) = halo_width(first(arr))
@inline tile_count(arr::AbstractArray{<:AbstractSingleHaloArray}) = tile_count(first(arr))
@inline tile_size(arr::AbstractArray{<:AbstractSingleHaloArray}) = tile_size(first(arr))
@inline tile_coordinates(arr::AbstractArray{<:AbstractSingleHaloArray}, tile_id::Integer) =
    tile_coordinates(first(arr), tile_id)
@inline neighbor_tile_id(arr::AbstractArray{<:AbstractSingleHaloArray}, tile_id::Integer,
        dim::Integer, side::Integer) =
    neighbor_tile_id(first(arr), tile_id, dim, side)

isactive(::AbstractCartesianTopology) = true   # default: subtypes may override

function validate_boundary_condition(topology::AbstractCartesianTopology, boundary_condition)
    isactive(topology) || return true
    N = ndims(topology)
    for d in 1:N
        left, right = boundary_condition[d]
        (left isa AbstractBoundaryCondition && right isa AbstractBoundaryCondition) ||
            error("boundary_condition[$d] must be a tuple of AbstractBoundaryCondition (got $(left), $(right))")
        # NoBoundaryCondition opts out of automatic BC; skip periodicity consistency check.
        (left isa NoBoundaryCondition || right isa NoBoundaryCondition) && continue
        topo_is_periodic = topology.periodic_boundary_condition[d]
        both_periodic = (left isa Periodic) && (right isa Periodic)
        any_periodic  = (left isa Periodic) || (right isa Periodic)
        if topo_is_periodic && !both_periodic
            error("Topology is periodic in dimension $d but boundary_condition[$d] is not (both sides must be Periodic).")
        elseif !topo_is_periodic && any_periodic
            error("Boundary condition in dimension $d uses Periodic but topology is not periodic.")
        end
    end
    return true
end

# ---- AbstractSingleHaloArray defaults ---------------------------------
# ThreadedHaloArray overrides fill!, copyto!, fill_interior!,
# fill_from_local_indices!, and Base.foreach with tforeach variants.

@inline Base.size(halo::AbstractSingleHaloArray)         = global_size(halo)
@inline Base.size(halo::AbstractSingleHaloArray, i::Int) = size(halo)[i]
@inline Base.axes(halo::AbstractSingleHaloArray)         = map(Base.OneTo, size(halo))
@inline Base.axes(halo::AbstractSingleHaloArray, i::Int) = Base.OneTo(size(halo, i))
@inline Base.length(halo::AbstractSingleHaloArray)       = prod(size(halo))

Base.:/(halo::AbstractSingleHaloArray, x::Number) = halo ./ x
Base.:*(halo::AbstractSingleHaloArray, x::Number) = halo .* x
Base.:*(x::Number, halo::AbstractSingleHaloArray) = x .* halo

function LinearAlgebra.norm(halo::AbstractSingleHaloArray, p::Real=2)
    if p == 2
        return sqrt(mapreduce(abs2, +, halo))
    elseif p == Inf
        return mapreduce(abs, max, halo)
    else
        return mapreduce(x -> abs(x)^p, +, halo)^(1/p)
    end
end

function Base.foreach(f, halo::AbstractSingleHaloArray)
    foreach(f, interior_view(halo))
    return nothing
end

"""
    fill_interior!(u, value)

Set every owned (ghost-free) cell of `u` to `value`. Ghosts are left untouched —
call [`synchronize_halo!`](@ref) afterwards if a stencil will read them.
"""
function fill_interior!(halo::AbstractSingleHaloArray, value)
    fill!(interior_view(halo), value)
    return halo
end

"""
    fill_from_local_indices!(f, u)

Set each owned cell from `f(i, j, …)`, where the indices are **local** to this
rank's/tile's owned region (1-based, ghost-free). For coordinates on the global
grid (the usual choice for initial conditions), use
[`fill_from_global_indices!`](@ref).
"""
function fill_from_local_indices!(f, halo::AbstractSingleHaloArray)
    interior = interior_view(halo)
    for I in CartesianIndices(interior)
        interior[I] = f(Tuple(I)...)
    end
    return nothing
end

function Base.copyto!(dest::AbstractSingleHaloArray, src::AbstractSingleHaloArray)
    copyto!(parent(dest), parent(src))
    return dest
end

function Base.copy(src::AbstractSingleHaloArray)
    dest = similar(src)
    copyto!(dest, src)
    return dest
end

function Base.zero(halo::AbstractSingleHaloArray)
    z = similar(halo)
    fill!(z, zero(eltype(halo)))
    return z
end

function Base.fill!(halo::AbstractSingleHaloArray, value)
    fill!(parent(halo), value)
    return halo
end

@inline owned_axes(halo::AbstractSingleHaloArray)         = axes(interior_view(halo))
@inline owned_axes(halo::AbstractSingleHaloArray, i::Int) = axes(interior_view(halo), i)

@inline versors(::AbstractSingleHaloArray{<:Any,N}) where {N} = versors(Val(N))

@inline Base.eachindex(halo::AbstractSingleHaloArray)             = eachindex(interior_view(halo))
@inline Base.iterate(halo::AbstractSingleHaloArray)               = iterate(interior_view(halo))
@inline Base.iterate(halo::AbstractSingleHaloArray, state)        = iterate(interior_view(halo), state)

Base.similar(halo::AbstractSingleHaloArray)                    = similar(halo, eltype(halo), size(halo))
Base.similar(halo::AbstractSingleHaloArray, ::Type{AA}) where {AA} = similar(halo, AA, size(halo))
Base.similar(halo::AbstractSingleHaloArray, dims::Dims{M}) where {M} = similar(halo, eltype(halo), dims)
Base.similar(halo::AbstractSingleHaloArray, dims::NTuple{M,<:Integer}) where {M} =
    similar(halo, eltype(halo), dims)

function Base.map!(f, dest::AbstractSingleHaloArray, src::Vararg{AbstractSingleHaloArray,Nsrc}) where {Nsrc}
    @views map!(f, interior_view(dest), map(interior_view, src)...)
    return dest
end

function Base.map(f, src::Vararg{AbstractSingleHaloArray,Nsrc}) where {Nsrc}
    dest = similar(src[1])
    map!(f, dest, src...)
    return dest
end

# ---- AbstractHaloCollection helpers (Group 3) -------------------------
# _first_field: the reference field used for geometry queries
# _fields:      all fields as an iterable (for operations like isactive)
# Concrete methods are defined in ArrayOfHaloArray.jl and multihaloarray.jl
# (those types don't exist at this point in the load order).
function _first_field end
function _fields end

@inline halo_backend(mha::AbstractHaloCollection) = halo_backend(_first_field(mha))
@inline halo_width(mha::AbstractHaloCollection)   = halo_width(_first_field(mha))
@inline tile_count(mha::AbstractHaloCollection)   = tile_count(_first_field(mha))
@inline tile_size(mha::AbstractHaloCollection)    = tile_size(_first_field(mha))
@inline tile_coordinates(mha::AbstractHaloCollection, tile_id::Integer) =
    tile_coordinates(_first_field(mha), tile_id)
@inline neighbor_tile_id(mha::AbstractHaloCollection, tile_id::Integer,
        dim::Integer, side::Integer) =
    neighbor_tile_id(_first_field(mha), tile_id, dim, side)
@inline is_root(mha::AbstractHaloCollection; root::Integer=0) =
    is_root(_first_field(mha); root=root)
@inline isactive(mha::AbstractHaloCollection) = all(isactive, _fields(mha))

function foreach_field!(f!, mha::AbstractHaloCollection)
    foreach(f!, _fields(mha))
    return nothing
end

@inline function _check_global_scalar_indices(halo::AbstractHaloArray, I::Tuple)
    length(I) == ndims(halo) || throw(BoundsError(halo, I))
    all(d -> first(axes(halo, d)) <= I[d] <= last(axes(halo, d)), eachindex(I)) ||
        throw(BoundsError(halo, I))
    return I
end

Base.getindex(halo::AbstractHaloArray, I::CartesianIndex) = getindex(halo, Tuple(I)...)
Base.setindex!(halo::AbstractHaloArray, value, I::CartesianIndex) =
    setindex!(halo, value, Tuple(I)...)
