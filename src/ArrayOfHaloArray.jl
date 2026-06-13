# The ArrayOfHaloArray alias, its docstring, the field-compatibility checks,
# and the ground-truth ArrayOfHaloArray(::AbstractArray) constructor live in
# field_collection.jl. _spatial_* geometry helpers live in abstract_haloarray.jl.

# MPI-backed fields — field-type-first, like the LocalHaloArray/ThreadedHaloArray
# constructors below. The shape of `boundary_conditions` fixes the field shape.
# (The element-type-first forms were removed: a dual-meaning first argument
# caused method ambiguities against the specialized field-type constructors.)
function ArrayOfHaloArray(::Type{<:HaloArray}, ::Type{T}, owned_dims::NTuple{N,Int},
        halo::Int, topology::CartesianTopology{N};
        boundary_conditions::AbstractArray) where {T,N}
    arrays = map(boundary_conditions) do bc
        HaloArray(T, owned_dims, halo, topology; boundary_condition=bc)
    end
    return ArrayOfHaloArray(arrays)
end

function ArrayOfHaloArray(::Type{<:HaloArray}, ::Type{T}, owned_dims::NTuple{N,Int},
        halo::Int; boundary_conditions::AbstractArray) where {T,N}
    arrays = map(boundary_conditions) do bc
        HaloArray(T, owned_dims, halo; boundary_condition=bc)
    end
    return ArrayOfHaloArray(arrays)
end

function ArrayOfHaloArray(::Type{LocalHaloArray}, ::Type{T}, owned_dims::NTuple{N,Int},
        halo::Int; boundary_conditions::AbstractArray) where {T,N}
    arrays = map(boundary_conditions) do bc
        LocalHaloArray(T, owned_dims, halo; boundary_condition=bc)
    end
    return ArrayOfHaloArray(arrays)
end

function ArrayOfHaloArray(::Type{LocalHaloArray}, owned_dims::NTuple{N,Int}, halo::Int;
        boundary_conditions::AbstractArray) where {N}
    return ArrayOfHaloArray(LocalHaloArray, Float64, owned_dims, halo;
        boundary_conditions=boundary_conditions)
end

function _arrayofhaloarray_boundary_conditions(field_shape::NTuple{F,Int},
        boundary_condition, boundary_conditions) where {F}
    if boundary_conditions === nothing
        return fill(boundary_condition, field_shape)
    end

    size(boundary_conditions) == field_shape ||
        throw(DimensionMismatch("boundary_conditions shape $(size(boundary_conditions)) != field shape $field_shape"))
    return boundary_conditions
end

function _local_arrayofhaloarray_field(::Type{T}, owned_dims::NTuple{N,Int},
        halo::Int, boundary_condition, storage) where {T,N}
    fullsize = ntuple(d -> Int(owned_dims[d]) + 2 * halo, Val(N))
    data = storage(T, fullsize...)
    return LocalHaloArray(data, halo, boundary_condition)
end

function ArrayOfHaloArray(::Type{LocalHaloArray}, ::Type{T},
        field_shape::NTuple{F,<:Integer}, owned_dims::NTuple{N,<:Integer},
        halo::Integer;
        boundary_condition=:repeating,
        boundary_conditions=nothing,
        storage=zeros) where {T,F,N}
    shape = ntuple(d -> Int(field_shape[d]), Val(F))
    owned = ntuple(d -> Int(owned_dims[d]), Val(N))
    h = Int(halo)
    bcs = _arrayofhaloarray_boundary_conditions(shape, boundary_condition, boundary_conditions)
    arrays = map(bcs) do bc
        _local_arrayofhaloarray_field(T, owned, h, bc, storage)
    end
    return ArrayOfHaloArray(arrays)
end

function ArrayOfHaloArray(::Type{LocalHaloArray},
        field_shape::NTuple{F,<:Integer}, owned_dims::NTuple{N,<:Integer},
        halo::Integer; kwargs...) where {F,N}
    return ArrayOfHaloArray(LocalHaloArray, Float64, field_shape, owned_dims, halo; kwargs...)
end

function ArrayOfHaloArray(::Type{ThreadedHaloArray}, ::Type{T}, tile_size::NTuple{N,<:Integer},
        halo::Integer; dims::NTuple{N,<:Integer}, boundary_conditions::AbstractArray) where {T,N}
    arrays = map(boundary_conditions) do bc
        ThreadedHaloArray(T, tile_size, halo; dims=dims, boundary_condition=bc)
    end
    return ArrayOfHaloArray(arrays)
end

function ArrayOfHaloArray(::Type{ThreadedHaloArray}, tile_size::NTuple{N,<:Integer},
        halo::Integer; dims::NTuple{N,<:Integer}, boundary_conditions::AbstractArray) where {N}
    return ArrayOfHaloArray(ThreadedHaloArray, Float64, tile_size, halo;
        dims=dims, boundary_conditions=boundary_conditions)
end

function ArrayOfHaloArray(::Type{ThreadedHaloArray}, ::Type{T},
        field_shape::NTuple{F,<:Integer}, tile_size::NTuple{N,<:Integer},
        halo::Integer;
        dims::NTuple{N,<:Integer},
        boundary_condition=:repeating,
        boundary_conditions=nothing) where {T,F,N}
    shape = ntuple(d -> Int(field_shape[d]), Val(F))
    bcs = _arrayofhaloarray_boundary_conditions(shape, boundary_condition, boundary_conditions)
    arrays = map(bcs) do bc
        ThreadedHaloArray(T, tile_size, halo; dims=dims, boundary_condition=bc)
    end
    return ArrayOfHaloArray(arrays)
end

function ArrayOfHaloArray(::Type{ThreadedHaloArray},
        field_shape::NTuple{F,<:Integer}, tile_size::NTuple{N,<:Integer},
        halo::Integer; kwargs...) where {F,N}
    return ArrayOfHaloArray(ThreadedHaloArray, Float64, field_shape, tile_size, halo; kwargs...)
end

# eltype/ndims come from AbstractArray{T,D} via FieldCollection{T,D,S,C}.
@inline field_shape(mha::ArrayOfHaloArray) = size(getfield(mha, :arrays))
@inline Base.parent(mha::ArrayOfHaloArray) = mha.arrays

# Everything else ArrayOfHaloArray needs is container-generic and defined once on
# FieldCollection (field_collection.jl) / AbstractHaloCollection
# (abstract_haloarray.jl): the _fields/_first_field/_map_fields/_check_same_fields
# hooks, integer+Cartesian getindex/setindex!, similar(c[, T][, dims]),
# copy/copyto!/fill!/zero, map, interior_view, map_over_field, all/any (reduction.jl),
# and halo_backend/halo_width/tile_*/isactive/is_root.
