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
    MultiHaloArray(Backend, T, dims, halo[, topology]; boundary_conditions)
    MultiHaloArray(Backend, T, dims, halo[, topology]; fields, boundary_condition)
    MultiHaloArray(named_tuple_of_fields)

A collection of several **named** halo-array fields sharing the same geometry
(dimensionality, interior size, halo width, and backend). Access a field by
name (`state.rho`), refresh them all with one [`synchronize_halo!`](@ref)`(state)`,
and broadcast/reduce over all fields at once (`state .*= 2`).

`Backend` is the field type: [`LocalHaloArray`](@ref), [`ThreadedHaloArray`](@ref)
(with `dims=` for the tile grid), or [`HaloArray`](@ref) (MPI, with a
`topology`). `boundary_conditions` is a `NamedTuple` mapping each field name to
its boundary condition, the field names being its keys; `fields=(:rho, :p)` with
one `boundary_condition` for all of them is the shorthand. Or pass a `NamedTuple`
of pre-built arrays, which must share geometry, halo width, backend, and (for
threaded fields) tiling.

Use this when a solver evolves several fields on one grid (e.g. `rho`, `u`, `v`,
`p`). For an integer/matrix-indexed collection instead of names, see
[`ArrayOfHaloArray`](@ref). Both are aliases of one underlying type
(`FieldCollection`), so they share all generic behaviour.

Fields may themselves be collections. When they share their field shape
(`MultiHaloArray((; a, b))` with `a`, `b` collections of 4 fields each) the
result has the outer field axis followed by the inner ones, size
`(2, 4, spatial...)`, indexed `c[outer, inner..., spatial...]`. When they do not
(a 4-field and a 6-field collection, or a single array next to a collection)
the result has one field axis over all its leaves in declaration
order, size `(10, spatial...)`, indexed `c[leaf, spatial...]`. Either way
`length(c)` is the number of elements and `siteview` gives every leaf.

# Examples
```julia
state = MultiHaloArray(LocalHaloArray, Float64, (64, 64), 1; boundary_conditions=(
    rho = ((Periodic(), Periodic()), (Periodic(), Periodic())),
    p   = ((Reflecting(), Reflecting()), (Periodic(), Periodic())),
))
state.rho .= 1.0
synchronize_halo!(state)   # refreshes every field
q = MultiHaloArray(ThreadedHaloArray, Float64, (32, 32), 1; dims=(2, 2),
                   fields=(:rho, :p), boundary_condition=:periodic)
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

# A field of a collection: a single halo array, or (nested) another collection.
const HaloArrayField = Union{AbstractSingleHaloArray,AbstractHaloCollection}

# ---- field-compatibility checks ------------------------------------------
# Every field must match the first in spatial dimensionality, interior size,
# halo width and backend. With `same_field_shape` (an ArrayOfHaloArray, whose
# fields form a grid) nested fields must also share their field shape, so the
# whole is one rectangular array; a MultiHaloArray may mix field shapes, and
# then lists its leaves along one field axis (see `_layout`).
# `labeled_fields` is an iterable of (label, field) pairs (the label only
# colours the error message — a field name for MultiHaloArray, an index for
# ArrayOfHaloArray).
function _check_fields_compatible(what::AbstractString, ref, labeled_fields; same_field_shape::Bool=true)
    ref_ndims   = _spatial_ndims(ref)
    ref_size    = _spatial_interior_size(ref)
    ref_halo    = halo_width(ref)
    ref_backend = halo_backend(ref)
    ref_fshape  = field_shape(ref)
    for (label, a) in labeled_fields
        _spatial_ndims(a) == ref_ndims ||
            throw(ArgumentError("$what field `$label` has dimensionality $(_spatial_ndims(a)) != $ref_ndims"))
        _spatial_interior_size(a) == ref_size ||
            throw(DimensionMismatch("$what field `$label` has interior size $(_spatial_interior_size(a)) != $ref_size"))
        (!same_field_shape || field_shape(a) == ref_fshape) ||
            throw(DimensionMismatch("$what field `$label` has field shape $(field_shape(a)) != $ref_fshape " *
                "(the fields of an ArrayOfHaloArray must share their field shape; a MultiHaloArray accepts different ones)"))
        halo_width(a) == ref_halo ||
            throw(DimensionMismatch("$what field `$label` has halo width $(halo_width(a)) != $ref_halo"))
        halo_backend(a) isa typeof(ref_backend) ||
            throw(ArgumentError("$what field `$label` has backend $(typeof(halo_backend(a))) != $(typeof(ref_backend))"))
        _check_same_layout(what, label, a, ref)
    end
    return nothing
end

# Threaded fields must also share the tiling: equal global sizes with different
# tile sizes or tile grids would let tile-indexed operations (siteview with a
# tile id, per-tile kernels) address different global cells in different fields.
@inline _check_same_layout(what, label, a, ref) = nothing
function _check_same_layout(what, label, a::ThreadedHaloArray, ref::ThreadedHaloArray)
    tile_size(a) == tile_size(ref) ||
        throw(DimensionMismatch("$what field `$label` has tile size $(tile_size(a)) != $(tile_size(ref))"))
    a.topology.dims == ref.topology.dims ||
        throw(DimensionMismatch("$what field `$label` has tile grid $(a.topology.dims) != $(ref.topology.dims)"))
    return nothing
end

function _check_multihaloarray_compatible(field_names, field_values)
    isempty(field_values) && throw(ArgumentError("MultiHaloArray requires at least one field"))
    _check_fields_compatible("MultiHaloArray", first(field_values),
        zip(field_names, field_values); same_field_shape=false)
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

# D counts the container's own axes plus the fields' (which, for a nested
# collection, include the inner field axes): D = ndims(field) + container axes.
# Fields of different field shapes cannot be stacked: the collection then has
# one field axis over all its leaves, D = S + 1.
function MultiHaloArray(arrs::NamedTuple)
    field_names = keys(arrs)
    field_values = values(arrs)
    _check_multihaloarray_compatible(field_names, field_values)

    T = promote_type(map(eltype, field_values)...)
    S = _spatial_ndims(first(field_values))
    ref_shape = field_shape(first(field_values))
    stacked = all(f -> field_shape(f) == ref_shape, field_values)
    D = stacked ? ndims(first(field_values)) + 1 : S + 1
    return FieldCollection{T, D, S, typeof(arrs)}(arrs)
end

function ArrayOfHaloArray(arrays::AbstractArray)
    _check_arrayofhaloarray_compatible(arrays)

    T = promote_type(map(eltype, arrays)...)
    S = _spatial_ndims(first(arrays))
    return FieldCollection{T, ndims(first(arrays)) + ndims(arrays), S, typeof(arrays)}(arrays)
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
# Number of axes of the field container itself (1 for a NamedTuple).
@inline _container_ndims(::MultiHaloArray) = 1
@inline _container_ndims(c::ArrayOfHaloArray) = ndims(getfield(c, :arrays))

# Field layout (a Holy trait). `Stacked`: the outer field axis followed by the
# fields' own (equal) field axes — every flat collection, every ArrayOfHaloArray.
# `LeafAxis`: a MultiHaloArray whose fields differ in field shape (a single
# array next to a collection, or collections of different field counts) has one
# field axis over all its leaves, in declaration order. Read back from the type
# the constructor chose: one field axis (D == S + 1) while some field is itself
# a collection. Code that differs by layout dispatches on `_layout(c)`; the two
# layout types are defined in abstract_haloarray.jl, which uses them first.
@inline _layout(::AbstractHaloArray) = Stacked()
@inline _layout(::FieldCollection{T,D,S,NamedTuple{N,Tup}}) where {T,D,S,N,Tup} =
    (D - S == 1 && _any_collection(Tup)) ? LeafAxis() : Stacked()
@inline _any_collection(::Type{Tuple{}}) = false
@inline _any_collection(::Type{Tup}) where {Tup<:Tuple} =
    fieldtype(Tup, 1) <: AbstractHaloCollection || _any_collection(Base.tuple_type_tail(Tup))

# Number of leaf arrays of a field.
@inline _leaf_count(::AbstractSingleHaloArray) = 1
@inline _leaf_count(c::AbstractHaloCollection) = n_field(c)
@inline _map_fields(g, c::FieldCollection) = _rebuild_collection(map(g, getfield(c, :arrays)))
@inline _check_same_fields(dest::FieldCollection, src::FieldCollection) =
    keys(getfield(dest, :arrays)) == keys(getfield(src, :arrays)) ||
        throw(DimensionMismatch("collection copyto! requires matching field layout"))

to_tuple(c::FieldCollection) = (_fields(c)...,)

"""
    active_fields(c)

`is_active` of every field, in the same container kind (a `NamedTuple` of Bools
for [`MultiHaloArray`](@ref), an array of Bools for [`ArrayOfHaloArray`](@ref)).
"""
active_fields(c::FieldCollection) = map(is_active, getfield(c, :arrays))

# One tile accessor for both flavors: the result keeps the container kind.
@inline tile_parent(c::FieldCollection, tile_id::Integer) =
    map(a -> tile_parent(a, tile_id), getfield(c, :arrays))

# ---- indexing: field axes first, then spatial axes --------------------------
# The container's own axes come first (1 for named collections); the rest of
# the indices go to the selected field, which recurses for a nested collection
# and reaches a cell on a leaf. Short indexing with up to the container's axes
# returns the field. NamedTuples support integer indexing, so this covers both
# flavors.

Base.getindex(c::FieldCollection, I...) = _getindex(_layout(c), c, I...)
Base.getindex(c::FieldCollection, I::CartesianIndex) = getindex(c, Tuple(I)...)
Base.setindex!(c::FieldCollection, value, I...) = (_setindex!(_layout(c), c, value, I...); c)

function _getindex(::Stacked, c::FieldCollection{T,D,S}, I...) where {T,D,S}
    cn = _container_ndims(c)
    if length(I) <= cn
        return getindex(getfield(c, :arrays), I...)
    elseif length(I) == D
        field_idx = ntuple(d -> I[d], cn)
        rest = ntuple(d -> I[cn + d], D - cn)
        return getindex(getfield(c, :arrays)[field_idx...], rest...)
    else
        throw(BoundsError(c, I))
    end
end
function _getindex(::LeafAxis, c::FieldCollection{T,D,S}, I...) where {T,D,S}
    (length(I) == 1 || length(I) == D) && 1 <= I[1] <= n_field(c) || throw(BoundsError(c, I))
    leaf = _leaf_field(c, I[1])
    return length(I) == 1 ? leaf : getindex(leaf, Base.tail(I)...)
end

function _setindex!(::Stacked, c::FieldCollection{T,D,S}, value, I...) where {T,D,S}
    length(I) == D || throw(BoundsError(c, I))
    cn = _container_ndims(c)
    field_idx = ntuple(d -> I[d], cn)
    rest = ntuple(d -> I[cn + d], D - cn)
    setindex!(getfield(c, :arrays)[field_idx...], value, rest...)
end
function _setindex!(::LeafAxis, c::FieldCollection{T,D,S}, value, I...) where {T,D,S}
    length(I) == D && 1 <= I[1] <= n_field(c) || throw(BoundsError(c, I))
    setindex!(_leaf_field(c, I[1]), value, Base.tail(I)...)
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

function Base.similar(c::FieldCollection{AA,D}, ::Type{T}, dims::Dims{M}) where {AA,D,T,M}
    M == D ||
        throw(DimensionMismatch("collection similar dims must have $D dimensions"))
    return _similar(_layout(c), c, T, dims)
end
# The container's own axes come first; the rest (inner field axes of a nested
# field, then spatial) are the dims of each field.
function _similar(::Stacked, c::FieldCollection{AA,D}, ::Type{T}, dims) where {AA,D,T}
    cn = _container_ndims(c)
    new_container_shape = ntuple(d -> Int(dims[d]), cn)
    field_dims = ntuple(d -> Int(dims[cn + d]), D - cn)
    new_container_shape == _container_shape(c) &&
        return _map_fields(a -> similar(a, T, field_dims), c)
    ref = _first_field(c)
    arrs = _reshape_field_container(getfield(c, :arrays), new_container_shape,
        similar(ref, T, field_dims), () -> similar(ref, T, field_dims))
    return _rebuild_collection(arrs)
end
# The leaf count is fixed by the named fields.
function _similar(::LeafAxis, c::FieldCollection{AA,D}, ::Type{T}, dims) where {AA,D,T}
    Int(dims[1]) == n_field(c) || throw(DimensionMismatch(
        "cannot change the field count of a named collection (MultiHaloArray) via similar"))
    spatial = ntuple(d -> Int(dims[1 + d]), D - 1)
    return _map_fields(a -> similar(a, T, (field_shape(a)..., spatial...)), c)
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
[`interior_view`](@ref) (ghost-free views). A nested collection gives nested
containers, one level per collection, down to the leaf storages.
"""
@inline field_storages(c::FieldCollection) = map(_leaf_storages, getfield(c, :arrays))
@inline _leaf_storages(a::AbstractSingleHaloArray) = parent(a)
@inline _leaf_storages(c::FieldCollection) = field_storages(c)

"""
    field_storages!(dest, c::FieldCollection)

Fill `dest` with the raw padded backing array of every field and return `dest`.

[`field_storages`](@ref) builds a fresh container on every call, which allocates
for an [`ArrayOfHaloArray`](@ref) (its field count is not part of its type, so
the result cannot be a stack-allocated tuple). A hot loop that needs the raw
storages can hoist a `dest` container out of the loop and refill it here instead,
staying allocation-free. `dest` must be indexable in the collection's field
order with `prod(field_shape(c))` entries (the leaf fields in column-major
[`field_shape`](@ref) order for a nested collection) — `similar(field_storages(c))`
gives a suitable one for a flat collection.

```julia
cache = similar(field_storages(u))
field_storages!(cache, u)          # no allocation
accumulate_flux_divergence!(field_storages!(cache2, du), cache, ranges, 1, inv(dx), flux, read, scatter!)
```
"""
function field_storages!(dest, c::FieldCollection)
    n = n_field(c)
    length(dest) == n ||
        throw(DimensionMismatch("dest must hold one entry per leaf field; got $(length(dest)) for $n fields"))
    @inbounds for (j, k) in zip(eachindex(dest), 1:n)
        dest[j] = parent(_leaf_field(c, k))
    end
    return dest
end
