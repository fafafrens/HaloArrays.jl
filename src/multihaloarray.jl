# The MultiHaloArray alias, its docstring, and the ground-truth
# MultiHaloArray(::NamedTuple) constructor live in field_collection.jl.


# One constructor for every backend. The boundary conditions are resolved into
# a NamedTuple shaped like the fields (`fields` + one `boundary_condition`, or
# `boundary_conditions` keyed by name) and one field is built per entry with
# the backend's own constructor `FT(T, geometry, halo[, topology]; …)`. What
# follows the halo (an MPI topology) and every keyword this method does not
# consume (`dims`, `thread_backend`, `storage`, …) goes to that constructor
# unchanged, so a backend option never needs a collection method of its own.
function MultiHaloArray(FT::Type{<:AbstractSingleHaloArray}, ::Type{T},
        geometry::NTuple{N,<:Integer}, halo::Integer, topology...;
        fields::Union{NTuple{<:Any,Symbol},Nothing}=nothing,
        boundary_condition=:repeating,
        boundary_conditions::Union{NamedTuple,Nothing}=nothing,
        kwargs...) where {T,N}
    bcs = _resolve_bcs(fields, boundary_condition, boundary_conditions)
    return MultiHaloArray(map(bc -> FT(T, geometry, halo, topology...; boundary_condition=bc, kwargs...), bcs))
end

MultiHaloArray(FT::Type{<:AbstractSingleHaloArray}, geometry::NTuple{N,<:Integer}, halo::Integer,
        args...; kwargs...) where {N} =
    MultiHaloArray(FT, Float64, geometry, halo, args...; kwargs...)


Base.getindex(mha::MultiHaloArray, name::Symbol) = mha.arrays[name]

# Named-field access: `state.rho` forwards to the backing field, while
# `state.arrays` still returns the underlying NamedTuple (`arrays` is the only
# real struct field). `getfield` is used internally to avoid recursion.
@inline Base.getproperty(mha::MultiHaloArray, name::Symbol) =
    name === :arrays ? getfield(mha, :arrays) : getfield(mha, :arrays)[name]
Base.propertynames(mha::MultiHaloArray) = keys(getfield(mha, :arrays))

# eltype/ndims come from AbstractArray{T,D} via FieldCollection{T,D,S,C,L}.

# size/axes/eachindex/length, n_field, interior/global/storage size, and
# interior_axes come from AbstractHaloCollection (field_shape prefix + _spatial_*).
"""
    field_shape(c) -> Dims

The shape of the field index of a halo collection `c`: the sizes of the index
dimensions that select a field, leaving the spatial dimensions out. For a
`MultiHaloArray` of `n` named fields this is `(n,)`; for an `ArrayOfHaloArray` it
is the `size` of the backing field array (e.g. `(4, 2)` for a `4×2` grid of
fields). For a nested collection the outer container's shape is followed by
the fields' own field shape; a `MultiHaloArray` whose fields differ in field
shape has the single axis `(number of leaves,)`. `prod(field_shape(c))` is the
number of leaf fields; a single halo array has field shape `()`.
"""
@inline field_shape(c::AbstractHaloCollection) =
    (_container_shape(c)..., field_shape(_first_field(c))...)
@inline field_shape(c::FieldCollection{T,D,S,C,LeafAxis}) where {T,D,S,C} =
    (sum(_leaf_count, _fields(c)),)
@inline field_shape(::AbstractSingleHaloArray) = ()
@inline _container_shape(mha::MultiHaloArray) = (length(getfield(mha, :arrays)),)
# parent (the field NamedTuple) and field_storages (the storage NamedTuple) are
# container-generic in field_collection.jl.

# Everything else MultiHaloArray needs is container-generic and defined once on
# FieldCollection (field_collection.jl) / AbstractHaloCollection
# (abstract_haloarray.jl): the _fields/_first_field/_map_fields/_check_same_fields
# hooks, tile_parent, integer+Cartesian getindex/
# setindex!, similar(c[, T][, dims]), copy/copyto!/fill!/zero, map, interior_view,
# map_over_field, all/any, and halo_backend/halo_width/tile_*/is_active/is_root.

# ============================================================
# Helpers for the uniform-BC shorthand (fields + boundary_condition)
# ============================================================

function _make_boundary_conditions(fields::NTuple{M,Symbol}, bc) where {M}
    return NamedTuple{fields}(ntuple(_ -> bc, Val(M)))
end

function _resolve_bcs(
        fields::Union{NTuple{<:Any,Symbol},Nothing},
        bc,
        boundary_conditions::Union{NamedTuple,Nothing})
    fields !== nothing && return _make_boundary_conditions(fields, bc)
    boundary_conditions !== nothing && return boundary_conditions
    throw(ArgumentError(
        "provide either `fields` (with `boundary_condition`) or `boundary_conditions`"))
end
