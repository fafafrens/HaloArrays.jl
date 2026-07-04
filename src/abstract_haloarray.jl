"""
    AbstractCartesianTopology{N}

Common supertype for N-dimensional Cartesian decomposition topologies.

Concrete subtypes:
- `CartesianTopology{N,C}` — MPI-backed distributed topology
- `ThreadedCartesianTopology{N}` — shared-memory tiled topology

Shared interface (all subtypes must implement):
- `Base.ndims(::AbstractCartesianTopology{N})` → N
- `is_active(topology)` → Bool
- `is_root(topology; root=0)` → Bool
- `topology.dims` → NTuple{N,Int}
- `topology.periodic_boundary_condition` → NTuple{N,Bool}
"""
abstract type AbstractCartesianTopology{N} end

"""
    AbstractHaloArray{T,N} <: AbstractArray{T,N}

Root of the halo-array type hierarchy: an `N`-dimensional array of `T` whose
storage carries ghost (halo) cells around an interior region. All halo arrays —
single backends and field collections alike — subtype this and present the
ghost-free interior through the standard `AbstractArray` interface, so generic
code (`broadcast`, `sum`, `similar`, …) sees only the owned data.

Hierarchy:
- [`AbstractSingleHaloArray`](@ref) — one field per array
  - [`AbstractDistributedHaloArray`](@ref) — MPI-backed (`HaloArray`)
  - [`AbstractSerialHaloArray`](@ref) — single-process (`LocalHaloArray`, `ThreadedHaloArray`)
- [`AbstractHaloCollection`](@ref) — a bundle of same-geometry fields
"""
abstract type AbstractHaloArray{T,N} <: AbstractArray{T,N} end

"""
    AbstractSingleHaloArray{T,N} <: AbstractHaloArray{T,N}

A halo array holding a single field in one backing `data` array (accessible via
`parent`). Supertype of the serial ([`AbstractSerialHaloArray`](@ref)) and
distributed ([`AbstractDistributedHaloArray`](@ref)) backends, as opposed to the
multi-field [`AbstractHaloCollection`](@ref).
"""
abstract type AbstractSingleHaloArray{T,N} <: AbstractHaloArray{T,N} end

"""
    AbstractDistributedHaloArray{T,N} <: AbstractSingleHaloArray{T,N}

A halo array whose interior is distributed across MPI ranks via a Cartesian
topology; each rank owns a local block and exchanges ghost cells with its
neighbors during [`synchronize_halo!`](@ref). The concrete subtype is
`HaloArray`.
"""
abstract type AbstractDistributedHaloArray{T,N} <: AbstractSingleHaloArray{T,N} end

"""
    AbstractSerialHaloArray{T,N} <: AbstractSingleHaloArray{T,N}

A halo array living entirely in one process: `LocalHaloArray` (a single padded
block) or `ThreadedHaloArray` (a shared-memory tiling). Serial backends are
always [`is_active`](@ref) and have no communicator (`communicator` returns
`nothing`).
"""
abstract type AbstractSerialHaloArray{T,N} <: AbstractSingleHaloArray{T,N} end

"""
    AbstractHaloCollection{T,N,S} <: AbstractHaloArray{T,N}

A bundle of same-geometry halo fields exposed as one `N`-dimensional array over
`S` spatial dimensions (so `N - S` index dimensions select the field). The
concrete type is `FieldCollection`, with the exported aliases `MultiHaloArray`
(fields in a `NamedTuple`, accessed by name) and `ArrayOfHaloArray` (fields in
an `AbstractArray`, accessed by index).
"""
abstract type AbstractHaloCollection{T,N,S} <: AbstractHaloArray{T,N} end

"""
    AbstractHaloBackend

Supertype of the singleton backend traits returned by [`halo_backend`](@ref):
[`MPIHaloBackend`](@ref), [`LocalHaloBackend`](@ref), and
[`ThreadedHaloBackend`](@ref). Dispatch on these to provide MPI-, local-, or
threaded-specific implementations while still accepting collection wrappers.
"""
abstract type AbstractHaloBackend end

"""
    MPIHaloBackend() <: AbstractHaloBackend

Backend trait for distributed `HaloArray`s. Returned by
[`halo_backend`](@ref) for MPI-backed storage.
"""
struct MPIHaloBackend <: AbstractHaloBackend end

"""
    LocalHaloBackend() <: AbstractHaloBackend

Backend trait for single-process `LocalHaloArray` storage. Returned by
[`halo_backend`](@ref).
"""
struct LocalHaloBackend <: AbstractHaloBackend end

"""
    ThreadedHaloBackend() <: AbstractHaloBackend

Backend trait for shared-memory, tiled `ThreadedHaloArray` storage. Returned by
[`halo_backend`](@ref).
"""
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
"""
    is_active(x) -> Bool

Whether this rank participates in the halo array / collection `x`. Always `true`
for serial backends; for a distributed `HaloArray` it is `false` on ranks that
fall outside the Cartesian topology (e.g. when the communicator has more ranks
than the decomposition uses), letting collective code skip inactive ranks.
"""
@inline is_active(::AbstractSerialHaloArray) = true

"""
    communicator(x) -> MPI.Comm or nothing

The MPI communicator backing `x`, or `nothing` for serial (local/threaded)
backends. For a distributed `HaloArray` this is the Cartesian communicator of
its topology.
"""
@inline communicator(::AbstractSerialHaloArray) = nothing

# Storage geometry for contiguous single arrays (one padded `data` array).
# ThreadedHaloArray is tiled and overrides all three with per-tile versions.
"""
    storage_size(u[, i]) -> dims

Size of the backing storage **including** ghost padding (`interior + 2*halo` per
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

The index ranges of the interior cells **within the padded storage** (`(halo+1) :
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

Size of the interior (ghost-free) region local to this process. For `HaloArray`
this is the local MPI subdomain size, not the global distributed size; for
serial containers it equals the full logical size.
"""
@inline interior_size(halo::AbstractHaloArray, i::Int) = interior_size(halo)[i]

"""
    interior_axes(halo[, i])

Axes of the interior (ghost-free) region local to this process. Use `axes(halo)`
for the global logical axes and `interior_axes(halo)` when looping over data
this process can update directly.
"""
@inline interior_axes(halo::AbstractHaloArray) = map(Base.OneTo, interior_size(halo))
@inline interior_axes(halo::AbstractHaloArray, i::Int) = interior_axes(halo)[i]

function storage_size end
function interior_to_global_index end
function global_to_storage_index end
function is_root end

# A non-tiled single array is a 1-tile decomposition: its sole "tile" is the
# whole padded storage (for a distributed HaloArray, this rank's local block).
# This makes per-tile kernels backend-agnostic — the same `for t in 1:tile_count(u);
# s = tile_parent(u, t); …` loop runs on a LocalHaloArray/HaloArray (one tile) and
# a ThreadedHaloArray (many tiles, which overrides these two methods).
@inline tile_count(::AbstractSingleHaloArray) = 1
@inline tile_parent(halo::AbstractSingleHaloArray, ::Integer) = parent(halo)

@inline halo_width(arr::AbstractArray{<:AbstractSingleHaloArray}) = halo_width(first(arr))
@inline tile_count(arr::AbstractArray{<:AbstractSingleHaloArray}) = tile_count(first(arr))
@inline tile_size(arr::AbstractArray{<:AbstractSingleHaloArray}) = tile_size(first(arr))
@inline tile_coordinates(arr::AbstractArray{<:AbstractSingleHaloArray}, tile_id::Integer) =
    tile_coordinates(first(arr), tile_id)
@inline neighbor_tile_id(arr::AbstractArray{<:AbstractSingleHaloArray}, tile_id::Integer,
        dim::Integer, side::Integer) =
    neighbor_tile_id(first(arr), tile_id, dim, side)

is_active(::AbstractCartesianTopology) = true   # default: subtypes may override

function validate_boundary_condition(topology::AbstractCartesianTopology, boundary_condition)
    is_active(topology) || return true
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
# and Base.foreach with tforeach variants.

@inline Base.size(halo::AbstractSingleHaloArray)         = global_size(halo)
# `size`/`axes` must return 1 / OneTo(1) for trailing dims `i > ndims` (the
# AbstractArray contract — e.g. ArrayInterface.zeromatrix does `u .* u'`, which
# asks for axes(u, 2) on a 1-D array).
@inline Base.size(halo::AbstractSingleHaloArray, i::Int) = i <= ndims(halo) ? size(halo)[i] : 1
@inline Base.axes(halo::AbstractSingleHaloArray)         = map(Base.OneTo, size(halo))
@inline Base.axes(halo::AbstractSingleHaloArray, i::Int) = Base.OneTo(size(halo, i))
@inline Base.length(halo::AbstractSingleHaloArray)       = prod(size(halo))

# Scalar `*`/`/`, `norm`, `dot`, and the in-place BLAS-1 updates (the
# vector-space / Hilbert-space interface) live in vector_space.jl.

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
# _fields:      all fields as an iterable (for operations like is_active)
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
@inline Base.size(c::AbstractHaloCollection, i::Int) = i <= ndims(c) ? size(c)[i] : 1
@inline Base.length(c::AbstractHaloCollection)       = prod(size(c))
@inline Base.axes(c::AbstractHaloCollection) = (map(Base.OneTo, field_shape(c))..., _spatial_axes(c)...)
@inline Base.axes(c::AbstractHaloCollection, i::Int) = i <= ndims(c) ? axes(c)[i] : Base.OneTo(1)
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
@inline is_active(mha::AbstractHaloCollection) = all(is_active, _fields(mha))

function foreach_field!(f!, mha::AbstractHaloCollection)
    foreach(f!, _fields(mha))
    return nothing
end

@inline function _check_global_scalar_indices(halo::AbstractHaloArray, I::Tuple)
    nd = ndims(halo)
    length(I) >= nd || throw(BoundsError(halo, I))
    # Trailing indices beyond ndims must be 1, per the AbstractArray contract:
    # `A[i, 1]` is valid for a vector. Generic LinearAlgebra code relies on this
    # (e.g. Diagonal's ldiv!, used as a preconditioner, indexes `B[i, 1]`).
    @inbounds for d in (nd + 1):length(I)
        isone(I[d]) || throw(BoundsError(halo, I))
    end
    idx = ntuple(d -> @inbounds(I[d]), Val(ndims(halo)))
    @inbounds for d in 1:nd
        first(axes(halo, d)) <= idx[d] <= last(axes(halo, d)) || throw(BoundsError(halo, I))
    end
    return idx
end

Base.getindex(halo::AbstractHaloArray, I::CartesianIndex) = getindex(halo, Tuple(I)...)
Base.setindex!(halo::AbstractHaloArray, value, I::CartesianIndex) =
    setindex!(halo, value, Tuple(I)...)
