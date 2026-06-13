# ============================================================
# FieldCollection — the single collection type behind MultiHaloArray and
# ArrayOfHaloArray.
#
# A collection of same-geometry halo fields is one concept with two field
# containers: a NamedTuple (access by name) or an AbstractArray (access by
# index). Both public names are parametric aliases of this struct, so all
# shared behaviour (geometry, reductions, broadcast, boundary conditions, data
# ops) is defined once on FieldCollection / AbstractHaloCollection, while the
# genuinely container-specific surface (getproperty vs shape indexing, HDF5
# layout, …) dispatches on the alias.
# ============================================================

"""
    FieldCollection{T,D,S,C} <: AbstractHaloCollection{T,D,S}

The common storage for multi-field halo collections: `arrays::C` is either a
`NamedTuple` of fields ([`MultiHaloArray`](@ref)) or an `AbstractArray` of
fields ([`ArrayOfHaloArray`](@ref)). `T` is the promoted element type, `D` the
total logical dimensionality (field axes + spatial axes), and `S` the spatial
dimensionality of the fields. Construct through the aliases.
"""
struct FieldCollection{T,D,S,C} <: AbstractHaloCollection{T,D,S}
    arrays::C
end

"""
    MultiHaloArray(HaloArray, T, owned_dims, halo[, topology]; boundary_conditions)
    MultiHaloArray(named_tuple_of_fields)

A collection of several **named** halo-array fields sharing the same geometry
(dimensionality, interior size, halo width, and backend). Access a field by
name (`state.rho`), refresh them all with one [`synchronize_halo!`](@ref)`(state)`,
and broadcast/reduce over all fields at once (`state .*= 2`).

`boundary_conditions` is a `NamedTuple` mapping each field name to its boundary
condition; the field names are taken from its keys. The backing fields are
[`HaloArray`](@ref)s (MPI) here; use [`LocalMultiHaloArray`](@ref) or
[`ThreadedMultiHaloArray`](@ref) for local/threaded fields, or pass a
`NamedTuple` of pre-built arrays.

Use this when a solver evolves several fields on one grid (e.g. `rho`, `u`, `v`,
`p`). For an integer/matrix-indexed collection instead of names, see
[`ArrayOfHaloArray`](@ref). Both are aliases of one underlying type
(`FieldCollection`), so they share all generic behaviour.

# Examples
```julia
state = LocalMultiHaloArray(Float64, (64, 64), 1; boundary_conditions=(
    rho = ((Periodic(), Periodic()), (Periodic(), Periodic())),
    p   = ((Reflecting(), Reflecting()), (Periodic(), Periodic())),
))
state.rho .= 1.0
synchronize_halo!(state)   # refreshes every field
```
"""
const MultiHaloArray{T,D,S,C<:NamedTuple} = FieldCollection{T,D,S,C}

"""
    ArrayOfHaloArray(FieldType, T, field_shape, owned_dims, halo; boundary_condition, …)
    ArrayOfHaloArray(array_of_fields)

A collection of halo-array fields stored in an `AbstractArray` and addressed by
**integer / Cartesian index** rather than by name — the counterpart to
[`MultiHaloArray`](@ref) (both are aliases of one underlying type). All fields
share the same geometry and backend.

Natural when the number of fields is decided at runtime, or when they form a
grid/tensor layout: a conserved state vector `q = (ρ, ρu, E)` as a `(3,)` array,
a velocity as `(2,)`, or a stress tensor as `(2, 2)`. `field_shape` is the shape
of the field container; `FieldType` is [`LocalHaloArray`](@ref) or
[`ThreadedHaloArray`](@ref) (MPI fields are built from a topology instead).

Index fields with `arr[i]` / `arr[i, j]`; [`synchronize_halo!`](@ref) and
broadcast act on every field at once.

# Examples
```julia
vel = ArrayOfHaloArray(LocalHaloArray, Float64, (2,), (16, 16), 1;
                       boundary_condition=:periodic)
interior_view(vel[1]) .= 1.0
synchronize_halo!(vel)
```
"""
const ArrayOfHaloArray{T,D,S,C<:AbstractArray} = FieldCollection{T,D,S,C}

const HaloArrayField = AbstractSingleHaloArray

# ---- field-compatibility checks ------------------------------------------
# Every field must match the first in spatial dimensionality, interior size,
# halo width, and backend. `labeled_fields` is an iterable of (label, field)
# pairs (the label only colours the error message — a field name for
# MultiHaloArray, an index for ArrayOfHaloArray).
function _check_fields_compatible(what::AbstractString, ref, labeled_fields)
    ref_ndims   = _spatial_ndims(ref)
    ref_size    = _spatial_interior_size(ref)
    ref_halo    = halo_width(ref)
    ref_backend = halo_backend(ref)
    for (label, a) in labeled_fields
        _spatial_ndims(a) == ref_ndims ||
            throw(ArgumentError("$what field `$label` has dimensionality $(_spatial_ndims(a)) != $ref_ndims"))
        _spatial_interior_size(a) == ref_size ||
            throw(DimensionMismatch("$what field `$label` has interior size $(_spatial_interior_size(a)) != $ref_size"))
        halo_width(a) == ref_halo ||
            throw(DimensionMismatch("$what field `$label` has halo width $(halo_width(a)) != $ref_halo"))
        halo_backend(a) isa typeof(ref_backend) ||
            throw(ArgumentError("$what field `$label` has backend $(typeof(halo_backend(a))) != $(typeof(ref_backend))"))
    end
    return nothing
end

function _check_multihaloarray_compatible(field_names, field_values)
    isempty(field_values) && throw(ArgumentError("MultiHaloArray requires at least one field"))
    _check_fields_compatible("MultiHaloArray", first(field_values),
        zip(field_names, field_values))
    return nothing
end

function _check_array_fields(arrays::AbstractArray)
    isempty(arrays) && throw(ArgumentError("ArrayOfHaloArray requires at least one field"))
    all(a -> a isa HaloArrayField, arrays) ||
        throw(ArgumentError("All fields must be HaloArray, LocalHaloArray, or ThreadedHaloArray"))
    return nothing
end

function _check_arrayofhaloarray_compatible(arrays::AbstractArray)
    _check_array_fields(arrays)
    _check_fields_compatible("ArrayOfHaloArray", first(arrays),
        ((I, arrays[I]) for I in CartesianIndices(arrays)))
    return nothing
end

# ---- ground-truth constructors --------------------------------------------

function MultiHaloArray(arrs::NamedTuple; check=nothing)
    field_names = keys(arrs)
    field_values = values(arrs)
    _check_multihaloarray_compatible(field_names, field_values)

    T = promote_type(map(eltype, field_values)...)
    S = _spatial_ndims(first(field_values))
    return FieldCollection{T, S + 1, S, typeof(arrs)}(arrs)
end

function ArrayOfHaloArray(arrays::AbstractArray; check=nothing)
    _check_arrayofhaloarray_compatible(arrays)

    T = promote_type(map(eltype, arrays)...)
    S = ndims(first(arrays))
    return FieldCollection{T, S + ndims(arrays), S, typeof(arrays)}(arrays)
end

# Rebuild the same kind of collection from a new field container (used by
# _map_fields and similar).
@inline _rebuild_collection(arrs::NamedTuple)    = MultiHaloArray(arrs)
@inline _rebuild_collection(arrs::AbstractArray) = ArrayOfHaloArray(arrs)

# Build one backing field of the requested type for a single boundary condition.
# This is the per-field core shared by the MultiHaloArray (named) and
# ArrayOfHaloArray (indexed) constructor families: `map`ping it over a bcs
# container (a NamedTuple or an array) yields the fields in the matching
# container, which the alias constructor then wraps. Adding a backend means
# adding one `_make_field` method, not editing both constructor families.
@inline _make_field(::Type{<:HaloArray}, ::Type{T}, owned_dims, halo, topology, bc) where {T} =
    HaloArray(T, owned_dims, halo, topology; boundary_condition=bc)
@inline _make_field(::Type{<:HaloArray}, ::Type{T}, owned_dims, halo, bc) where {T} =
    HaloArray(T, owned_dims, halo; boundary_condition=bc)
@inline _make_field(::Type{<:LocalHaloArray}, ::Type{T}, owned_dims, halo, bc) where {T} =
    LocalHaloArray(T, owned_dims, halo; boundary_condition=bc)
@inline _make_field(::Type{<:ThreadedHaloArray}, ::Type{T}, tile_size, halo, bc; dims) where {T} =
    ThreadedHaloArray(T, tile_size, halo; dims=dims, boundary_condition=bc)

# ---- container-generic methods ---------------------------------------------
# `values` is the identity on AbstractArrays and the field tuple on NamedTuples,
# and `keys`/`map` preserve the container kind — so one definition covers both
# flavors for everything below. (`parent` stays per-alias: for MultiHaloArray it
# is the NamedTuple of raw storages, for ArrayOfHaloArray the field array itself
# — a test-asserted contract.)

@inline _fields(c::FieldCollection)      = values(getfield(c, :arrays))
@inline _first_field(c::FieldCollection) = first(_fields(c))
@inline _map_fields(g, c::FieldCollection) = _rebuild_collection(map(g, getfield(c, :arrays)))
@inline _check_same_fields(dest::FieldCollection, src::FieldCollection) =
    keys(getfield(dest, :arrays)) == keys(getfield(src, :arrays)) ||
        throw(DimensionMismatch("collection copyto! requires matching field layout"))

to_tuple(c::FieldCollection) = (_fields(c)...,)

"""
    active_fields(c)

`isactive` of every field, in the same container kind (a `NamedTuple` of Bools
for [`MultiHaloArray`](@ref), an array of Bools for [`ArrayOfHaloArray`](@ref)).
"""
active_fields(c::FieldCollection) = map(isactive, getfield(c, :arrays))

# One tile accessor for both flavors: the result keeps the container kind.
@inline tile_parent(c::FieldCollection, tile_id::Integer) =
    map(a -> tile_parent(a, tile_id), getfield(c, :arrays))

# ---- indexing: field axes first, then spatial axes --------------------------
# `field_ndims = D - S` (1 for named collections). Short indexing with up to
# field_ndims indices returns the field; full-dims indexing reaches a cell.
# NamedTuples support integer indexing, so this covers both flavors.

function Base.getindex(c::FieldCollection{T,D,S}, I...) where {T,D,S}
    field_ndims = D - S

    if length(I) <= field_ndims
        return getindex(getfield(c, :arrays), I...)
    elseif length(I) == D
        field_idx = ntuple(d -> I[d], field_ndims)
        spatial_idx = ntuple(d -> I[field_ndims + d], S)
        return getindex(getfield(c, :arrays)[field_idx...], spatial_idx...)
    else
        throw(BoundsError(c, I))
    end
end

Base.getindex(c::FieldCollection, I::CartesianIndex) = getindex(c, Tuple(I)...)

function Base.setindex!(c::FieldCollection{T,D,S}, value, I...) where {T,D,S}
    field_ndims = D - S

    if length(I) == D
        field_idx = ntuple(d -> I[d], field_ndims)
        spatial_idx = ntuple(d -> I[field_ndims + d], S)
        setindex!(getfield(c, :arrays)[field_idx...], value, spatial_idx...)
        return c
    else
        throw(BoundsError(c, I))
    end
end

Base.setindex!(c::FieldCollection, value, I::CartesianIndex) =
    setindex!(c, value, Tuple(I)...)

# ---- similar with explicit dims ----------------------------------------------
# The field-shape prefix may only change for array containers; named collections
# cannot grow or shrink their field set.

@inline _reshape_field_container(arrs::AbstractArray, new_shape, prototype, build) =
    (out = similar(arrs, typeof(prototype), new_shape);
     for I in CartesianIndices(out); out[I] = build(); end; out)
@inline _reshape_field_container(::NamedTuple, new_shape, prototype, build) =
    throw(DimensionMismatch("cannot change the field count of a named collection (MultiHaloArray) via similar"))

function Base.similar(c::FieldCollection{AA,D,S}, ::Type{T}, dims::Dims{M}) where {AA,D,S,T,M}
    field_ndims = D - S
    M == D ||
        throw(DimensionMismatch("collection similar dims must have $D dimensions"))

    new_field_shape = ntuple(d -> Int(dims[d]), Val(field_ndims))
    spatial_dims = ntuple(d -> Int(dims[field_ndims + d]), Val(S))

    if new_field_shape == field_shape(c)
        return _map_fields(a -> similar(a, T, spatial_dims), c)
    else
        ref = _first_field(c)
        prototype = similar(ref, T, spatial_dims)
        arrs = _reshape_field_container(getfield(c, :arrays), new_field_shape,
            prototype, () -> similar(ref, T, spatial_dims))
        return _rebuild_collection(arrs)
    end
end

# Non-Int dims are normalized to Dims by Base's generic similar fallbacks.
Base.similar(c::FieldCollection, dims::Dims{M}) where {M} =
    similar(c, eltype(c), dims)
Base.similar(c::FieldCollection, dims::NTuple{M,<:Integer}) where {M} =
    similar(c, eltype(c), dims)

# ---- field-wise maps over the whole collection ------------------------------
# These all keep the container kind: a NamedTuple result for MultiHaloArray, an
# array result for ArrayOfHaloArray (`map`/`values` preserve the container).

"""
    map(f, c::FieldCollection)

Apply `f` elementwise to every field, returning a collection of the same kind
(a [`MultiHaloArray`](@ref) for named fields, an [`ArrayOfHaloArray`](@ref) for
indexed fields).
"""
Base.map(f, c::FieldCollection) = _map_fields(field -> map(f, field), c)

"""
    interior_view(c::FieldCollection[, tile_id])

The interior (ghost-free) view of every field, in the same container kind as the
collection (a `NamedTuple` for [`MultiHaloArray`](@ref), an array for
[`ArrayOfHaloArray`](@ref)). Pass `tile_id` for the per-tile interior of a
threaded collection.
"""
interior_view(c::FieldCollection) = map(interior_view, getfield(c, :arrays))
interior_view(c::FieldCollection, tile_id::Integer) =
    map(a -> interior_view(a, tile_id), getfield(c, :arrays))

"""
    map_over_field(f, c::FieldCollection)

Apply `f` to each **whole field** of `c` (not elementwise), returning the raw
field container — a `NamedTuple` of results for [`MultiHaloArray`](@ref), an
array of results for [`ArrayOfHaloArray`](@ref).
"""
map_over_field(f, c::FieldCollection) = map(f, getfield(c, :arrays))

# ---- the two unwrap levels --------------------------------------------------

"""
    parent(c::FieldCollection)

The collection's field container — the conventional one-level unwrap: a
`NamedTuple` of fields for a [`MultiHaloArray`](@ref), an array of fields for an
[`ArrayOfHaloArray`](@ref). For the raw padded backing array of every field
(e.g. to index with ghost offsets in a stencil), use [`field_storages`](@ref).
"""
@inline Base.parent(c::FieldCollection) = getfield(c, :arrays)

"""
    field_storages(c::FieldCollection)

The raw padded backing array of every field — `parent` pushed down to the
leaves — in the same container kind as the collection (a `NamedTuple` for
[`MultiHaloArray`](@ref), an array for [`ArrayOfHaloArray`](@ref)). Index these
with **storage** indices (ghost-inclusive), e.g. over [`interior_range`](@ref)
or a [`FaceRanges`](@ref) sweep. Contrast `parent` (the field container) and
[`interior_view`](@ref) (ghost-free views).
"""
@inline field_storages(c::FieldCollection) = map(parent, getfield(c, :arrays))
