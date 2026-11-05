# The ArrayOfHaloArray alias, its docstring, the field-compatibility checks,
# and the ground-truth ArrayOfHaloArray(::AbstractArray) constructor live in
# field_collection.jl. _spatial_* geometry helpers live in abstract_haloarray.jl.

# One constructor for every backend, as for MultiHaloArray: the boundary
# conditions become an array of `field_shape` (one `boundary_condition` for
# all, or `boundary_conditions` already in that shape), and one field is built
# per entry with `FT(T, geometry, halo[, topology]; …)`; the topology and the
# remaining keywords (`dims`, `thread_backend`, `storage`, …) pass through.
# Given only a `boundary_conditions` array, its size is the field shape.
function ArrayOfHaloArray(FT::Type{<:AbstractSingleHaloArray}, ::Type{T},
        field_shape::NTuple{F,<:Integer}, geometry::NTuple{N,<:Integer}, halo::Integer, topology...;
        boundary_condition=:repeating, boundary_conditions=nothing, kwargs...) where {T,F,N}
    shape = ntuple(d -> Int(field_shape[d]), Val(F))
    bcs = _arrayofhaloarray_boundary_conditions(shape, boundary_condition, boundary_conditions)
    return ArrayOfHaloArray(map(bc -> FT(T, geometry, halo, topology...; boundary_condition=bc, kwargs...), bcs))
end

function ArrayOfHaloArray(FT::Type{<:AbstractSingleHaloArray}, ::Type{T},
        geometry::NTuple{N,<:Integer}, halo::Integer, topology...;
        boundary_conditions::AbstractArray, kwargs...) where {T,N}
    return ArrayOfHaloArray(FT, T, size(boundary_conditions), geometry, halo, topology...;
        boundary_conditions, kwargs...)
end

ArrayOfHaloArray(FT::Type{<:AbstractSingleHaloArray}, shape::NTuple{N,<:Integer}, args...; kwargs...) where {N} =
    ArrayOfHaloArray(FT, Float64, shape, args...; kwargs...)

function _arrayofhaloarray_boundary_conditions(field_shape::NTuple{F,Int},
        boundary_condition, boundary_conditions) where {F}
    boundary_conditions === nothing && return fill(boundary_condition, field_shape)
    size(boundary_conditions) == field_shape ||
        throw(DimensionMismatch("boundary_conditions shape $(size(boundary_conditions)) != field shape $field_shape"))
    return boundary_conditions
end

# eltype/ndims come from AbstractArray{T,D} via FieldCollection{T,D,S,C,L}.
@inline _container_shape(mha::ArrayOfHaloArray) = size(getfield(mha, :arrays))
# parent (the field array) and field_storages (the array of storages) are
# container-generic in field_collection.jl.

# Everything else ArrayOfHaloArray needs is container-generic and defined once on
# FieldCollection (field_collection.jl) / AbstractHaloCollection
# (abstract_haloarray.jl): the _fields/_first_field/_map_fields/_check_same_fields
# hooks, integer+Cartesian getindex/setindex!, similar(c[, T][, dims]),
# copy/copyto!/fill!/zero, map, interior_view, map_over_field, all/any (reduction.jl),
# and halo_backend/halo_width/tile_*/is_active/is_root.
