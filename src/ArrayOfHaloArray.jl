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

# AbstractHaloCollection helpers (concrete methods; stubs in abstract_haloarray.jl)
@inline _first_field(mha::ArrayOfHaloArray) = first(parent(mha))
@inline _fields(mha::ArrayOfHaloArray)      = parent(mha)
@inline _map_fields(g, mha::ArrayOfHaloArray) = ArrayOfHaloArray(map(g, mha.arrays))
@inline _check_same_fields(dest::ArrayOfHaloArray, src::ArrayOfHaloArray) =
    field_shape(dest) == field_shape(src) ||
        throw(DimensionMismatch("ArrayOfHaloArray field shapes must match"))

# halo_backend, halo_width, tile_count, tile_size, tile_coordinates, neighbor_tile_id,
# is_root, isactive inherited from AbstractHaloCollection (abstract_haloarray.jl)

# size/axes/eachindex/length, n_field, interior/owned/global/storage size, and
# interior_axes come from AbstractHaloCollection (field_shape prefix + _spatial_*).

function Base.getindex(mha::ArrayOfHaloArray{T,D,S}, I...) where {T,D,S}
    field_ndims = D - S

    if length(I) <= field_ndims
        return getindex(mha.arrays, I...)
    elseif length(I) == length(size(mha))
        field_idx = ntuple(d -> I[d], field_ndims)
        spatial_idx = ntuple(d -> I[field_ndims + d], S)
        return getindex(mha.arrays[field_idx...], spatial_idx...)
    else
        throw(BoundsError(mha, I))
    end
end

Base.getindex(mha::ArrayOfHaloArray, I::CartesianIndex) = getindex(mha, Tuple(I)...)

function Base.setindex!(mha::ArrayOfHaloArray{T,D,S}, value, I...) where {T,D,S}
    field_ndims = D - S

    if length(I) == length(size(mha))
        field_idx = ntuple(d -> I[d], field_ndims)
        spatial_idx = ntuple(d -> I[field_ndims + d], S)
        setindex!(mha.arrays[field_idx...], value, spatial_idx...)
        return mha
    else
        throw(BoundsError(mha, I))
    end
end

Base.setindex!(mha::ArrayOfHaloArray, value, I::CartesianIndex) =
    setindex!(mha, value, Tuple(I)...)

function Base.similar(mha::ArrayOfHaloArray{AA,D,S}, ::Type{T},
        dims::Dims{M}) where {AA,D,S,T,M}
    field_ndims = D - S
    M == D ||
        throw(DimensionMismatch("ArrayOfHaloArray similar dims must have $D dimensions"))

    new_field_shape = ntuple(d -> Int(dims[d]), Val(field_ndims))
    spatial_dims = ntuple(d -> Int(dims[field_ndims + d]), Val(S))

    if new_field_shape == field_shape(mha)
        arrs = map(a -> similar(a, T, spatial_dims), mha.arrays)
    else
        ref = first(mha.arrays)
        prototype = similar(ref, T, spatial_dims)
        arrs = similar(mha.arrays, typeof(prototype), new_field_shape)
        for I in CartesianIndices(arrs)
            arrs[I] = similar(ref, T, spatial_dims)
        end
    end

    return ArrayOfHaloArray(arrs)
end

# Non-Int dims are normalized to Dims by Base's generic similar fallbacks.

Base.similar(mha::ArrayOfHaloArray, dims::Dims{M}) where {M} =
    similar(mha, eltype(mha), dims)
Base.similar(mha::ArrayOfHaloArray, dims::NTuple{M,<:Integer}) where {M} =
    similar(mha, eltype(mha), dims)

# similar(mha[, T]) / copy / copyto! / fill! / zero come from
# AbstractHaloCollection via _map_fields / _fields / _check_same_fields.

function Base.map(f, mha::ArrayOfHaloArray)
    arrs = map(a -> map(f, a), mha.arrays)
    return ArrayOfHaloArray(arrs)
end

# foreach_field!(f!, ::AbstractHaloCollection) inherited from abstract_haloarray.jl

function foreach_field!(f!, mha::ArrayOfHaloArray, others::Vararg{ArrayOfHaloArray})
    all(other -> field_shape(other) == field_shape(mha), others) ||
        throw(DimensionMismatch("ArrayOfHaloArray field shapes must match"))
    for I in eachindex(mha.arrays)
        f!(mha.arrays[I], map(other -> other.arrays[I], others)...)
    end
    return nothing
end

function map_over_field(f, mha::ArrayOfHaloArray)
    return map(f, mha.arrays)
end

function map_over_field(f, mha::ArrayOfHaloArray, others::Vararg{ArrayOfHaloArray})
    all(other -> field_shape(other) == field_shape(mha), others) ||
        throw(DimensionMismatch("ArrayOfHaloArray field shapes must match"))
    return map(CartesianIndices(mha.arrays)) do I
        f(mha.arrays[I], map(other -> other.arrays[I], others)...)
    end
end

function interior_view(mha::ArrayOfHaloArray)
    return map(interior_view, mha.arrays)
end

# isactive, is_root, tile_count, tile_size, tile_coordinates, neighbor_tile_id
# inherited from AbstractHaloCollection (abstract_haloarray.jl)

function Base.all(f::F, mha::ArrayOfHaloArray) where {F<:Function}
    return all(field -> all(f, field), mha.arrays)
end

function Base.any(f::F, mha::ArrayOfHaloArray) where {F<:Function}
    return any(field -> any(f, field), mha.arrays)
end
