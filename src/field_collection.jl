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
