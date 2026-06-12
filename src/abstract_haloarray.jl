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
Contrast with [`interior_size`](@ref) (ghost-free) and [`global_size`](@ref).
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
    interior_size(halo[, i])

Size of the interior (ghost-free) region this process owns. For `HaloArray`
this is the local MPI subdomain size, not the global distributed size; for
serial containers it equals the full logical size.
"""
@inline interior_size(halo::AbstractHaloArray, i::Int) = interior_size(halo)[i]

"""
    interior_axes(halo[, i])

Axes of the interior (ghost-free) region this process owns. Use `axes(halo)`
for the global logical axes and `interior_axes(halo)` when looping over data
this process can update directly.
"""
@inline interior_axes(halo::AbstractHaloArray) = map(Base.OneTo, interior_size(halo))
@inline interior_axes(halo::AbstractHaloArray, i::Int) = interior_axes(halo)[i]

function storage_size end
function interior_to_global_index end
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
# ThreadedHaloArray overrides fill!, copyto!,
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
    fill!(u, value)

Set every interior (ghost-free) cell of `u` to `value` — consistent with the
interior-only semantics of broadcast and reductions. Ghosts are left untouched;
call [`synchronize_halo!`](@ref) before a stencil reads them. Use
`fill!(parent(u), value)` to overwrite the raw storage including ghosts.
"""
function Base.fill!(halo::AbstractSingleHaloArray, value)
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


@inline interior_axes(halo::AbstractSingleHaloArray)         = axes(interior_view(halo))
@inline interior_axes(halo::AbstractSingleHaloArray, i::Int) = axes(interior_view(halo), i)

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

# ---- spatial geometry of the fields -----------------------------------
# Uniform across single halo arrays, collections, and raw AbstractArrays of
# fields: `_geometry_field` picks the reference field, and the `_spatial_*`
# accessors forward to it. Collections read their spatial dimension straight
# from the type parameter (AbstractHaloCollection{T,N,S}).
@inline _geometry_field(a::AbstractSingleHaloArray) = a
@inline _geometry_field(c::AbstractHaloCollection) = _first_field(c)
@inline _geometry_field(arr::AbstractArray{<:AbstractSingleHaloArray}) = first(arr)

@inline _spatial_ndims(x) = ndims(_geometry_field(x))
@inline _spatial_ndims(::AbstractHaloCollection{T,N,S}) where {T,N,S} = S
@inline _spatial_interior_range(x) = interior_range(_geometry_field(x))
@inline _spatial_interior_size(x)  = interior_size(_geometry_field(x))
@inline _spatial_global_size(x)    = global_size(_geometry_field(x))
@inline _spatial_storage_size(x)   = storage_size(_geometry_field(x))
@inline _spatial_axes(x)           = axes(_geometry_field(x))
@inline _spatial_interior_axes(x)     = interior_axes(_geometry_field(x))

# ---- collection shape / axes -------------------------------------------
# Every size/axes query on a collection is the field-axes prefix (field_shape:
# (n_field,) for named collections, the container Shape for indexed ones)
# followed by the shared spatial geometry of the fields. interior_size falls back
# to the generic interior_size = interior_size alias above.
@inline n_field(c::AbstractHaloCollection) = length(_fields(c))
@inline interior_size(c::AbstractHaloCollection) = (field_shape(c)..., _spatial_interior_size(c)...)
@inline global_size(c::AbstractHaloCollection)   = (field_shape(c)..., _spatial_global_size(c)...)
@inline storage_size(c::AbstractHaloCollection)  = (field_shape(c)..., _spatial_storage_size(c)...)
@inline storage_size(c::AbstractHaloCollection, i::Int) = storage_size(c)[i]
@inline Base.size(c::AbstractHaloCollection)         = global_size(c)
@inline Base.size(c::AbstractHaloCollection, i::Int) = size(c)[i]
@inline Base.length(c::AbstractHaloCollection)       = prod(size(c))
@inline Base.axes(c::AbstractHaloCollection) = (map(Base.OneTo, field_shape(c))..., _spatial_axes(c)...)
@inline Base.axes(c::AbstractHaloCollection, i::Int) = axes(c)[i]
@inline Base.eachindex(c::AbstractHaloCollection) = CartesianIndices(axes(c))
@inline interior_axes(c::AbstractHaloCollection) = (map(Base.OneTo, field_shape(c))..., _spatial_interior_axes(c)...)
@inline interior_axes(c::AbstractHaloCollection, i::Int) = interior_axes(c)[i]

# ---- whole-collection data ops via the field container -------------------
# _map_fields(g, c): apply g to every field and rebuild the same kind of
# collection (concrete hooks live next to each type's constructor).
# _check_same_fields(dest, src): layout compatibility for copyto!.
function _map_fields end
function _check_same_fields end

Base.copy(c::AbstractHaloCollection)    = _map_fields(copy, c)
Base.similar(c::AbstractHaloCollection) = _map_fields(similar, c)
Base.similar(c::AbstractHaloCollection, ::Type{T}) where {T} = _map_fields(a -> similar(a, T), c)

function Base.zero(c::AbstractHaloCollection)
    z = similar(c)
    fill!(z, zero(eltype(c)))
    return z
end

function Base.fill!(c::AbstractHaloCollection, value)
    foreach(f -> fill!(f, value), _fields(c))
    return c
end

function Base.copyto!(dest::C, src::C) where {C<:AbstractHaloCollection}
    _check_same_fields(dest, src)
    for (d, s) in zip(_fields(dest), _fields(src))
        copyto!(d, s)
    end
    return dest
end

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
