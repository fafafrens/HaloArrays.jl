# Multi-field halo container using an AbstractArray to store the fields.
mutable struct ArrayOfHaloArray{T,N,Shape,A}
    arrays::A
end

const HaloArrayField = Union{HaloArray,LocalHaloArray,ThreadedHaloArray}

@inline _spatial_ndims(x) = ndims(x)
@inline _spatial_size(x) = size(x)
@inline _spatial_interior_size(x) = interior_size(x)
@inline _spatial_full_size(x) = full_size(x)
@inline _spatial_axes(x) = axes(x)

@inline _spatial_ndims(::ArrayOfHaloArray{T,N}) where {T,N} = N
@inline _spatial_size(x::ArrayOfHaloArray) = size(first(parent(x)))
@inline _spatial_interior_size(x::ArrayOfHaloArray) = interior_size(first(parent(x)))
@inline _spatial_full_size(x::ArrayOfHaloArray) = full_size(first(parent(x)))
@inline _spatial_axes(x::ArrayOfHaloArray) = axes(first(parent(x)))

function _check_array_fields(arrays::AbstractArray)
    isempty(arrays) && throw(ArgumentError("ArrayOfHaloArray requires at least one field"))
    all(a -> a isa HaloArrayField, arrays) ||
        throw(ArgumentError("All fields must be HaloArray, LocalHaloArray, or ThreadedHaloArray"))
    return nothing
end

function _check_arrayofhaloarray_compatible(arrays::AbstractArray)
    _check_array_fields(arrays)

    ref = first(arrays)
    ref_ndims = ndims(ref)
    ref_interior_size = interior_size(ref)
    ref_halo_width = halo_width(ref)

    for I in CartesianIndices(arrays)
        a = arrays[I]
        ndims(a) == ref_ndims ||
            throw(ArgumentError("Field `$I` has dimensionality $(ndims(a)) != $ref_ndims"))
        interior_size(a) == ref_interior_size ||
            throw(DimensionMismatch("Field `$I` has interior size $(interior_size(a)) != $ref_interior_size"))
        halo_width(a) == ref_halo_width ||
            throw(DimensionMismatch("Field `$I` has halo width $(halo_width(a)) != $ref_halo_width"))
    end

    return nothing
end

function ArrayOfHaloArray(arrays::AbstractArray; check=nothing)
    _check_arrayofhaloarray_compatible(arrays)

    T = promote_type(map(eltype, arrays)...)
    N = ndims(first(arrays))
    Shape = size(arrays)
    return ArrayOfHaloArray{T,N,Shape,typeof(arrays)}(arrays)
end

function ArrayOfHaloArray(::Type{T}, local_size::NTuple{N,Int}, halo::Int,
        topology::CartesianTopology{N}; boundary_conditions::AbstractArray) where {T,N}
    arrays = map(boundary_conditions) do bc
        HaloArray(T, local_size, halo, topology; boundary_condition=bc)
    end
    return ArrayOfHaloArray(arrays)
end

function ArrayOfHaloArray(::Type{T}, local_size::NTuple{N,Int}, halo::Int,
        topology::CartesianTopology{N}, boundary_conditions::AbstractArray) where {T,N}
    return ArrayOfHaloArray(T, local_size, halo, topology; boundary_conditions=boundary_conditions)
end

function ArrayOfHaloArray(::Type{T}, local_size::NTuple{N,Int}, halo::Int;
        boundary_conditions::AbstractArray) where {T,N}
    arrays = map(boundary_conditions) do bc
        HaloArray(T, local_size, halo; boundary_condition=bc)
    end
    return ArrayOfHaloArray(arrays)
end

function ArrayOfHaloArray(::Type{T}, local_size::NTuple{N,Int}, halo::Int,
        boundary_conditions::AbstractArray) where {T,N}
    return ArrayOfHaloArray(T, local_size, halo; boundary_conditions=boundary_conditions)
end

function ArrayOfHaloArray(local_size::NTuple{N,Int}, halo::Int;
        boundary_conditions::AbstractArray) where {N}
    return ArrayOfHaloArray(Float64, local_size, halo; boundary_conditions=boundary_conditions)
end

function ArrayOfHaloArray(local_size::NTuple{N,Int}, halo::Int,
        boundary_conditions::AbstractArray) where {N}
    return ArrayOfHaloArray(Float64, local_size, halo; boundary_conditions=boundary_conditions)
end

function ArrayOfHaloArray(::Type{LocalHaloArray}, ::Type{T}, local_size::NTuple{N,Int},
        halo::Int; boundary_conditions::AbstractArray) where {T,N}
    arrays = map(boundary_conditions) do bc
        LocalHaloArray(T, local_size, halo; boundary_condition=bc)
    end
    return ArrayOfHaloArray(arrays)
end

function ArrayOfHaloArray(::Type{LocalHaloArray}, local_size::NTuple{N,Int}, halo::Int;
        boundary_conditions::AbstractArray) where {N}
    return ArrayOfHaloArray(LocalHaloArray, Float64, local_size, halo;
        boundary_conditions=boundary_conditions)
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

Base.eltype(::ArrayOfHaloArray{T}) where {T} = T
Base.ndims(::ArrayOfHaloArray{T,N,Shape}) where {T,N,Shape} = N + length(Shape)
Base.ndims(::Type{<:ArrayOfHaloArray{T,N,Shape}}) where {T,N,Shape} = N + length(Shape)
@inline field_shape(::ArrayOfHaloArray{T,N,Shape}) where {T,N,Shape} = Shape
@inline n_field(mha::ArrayOfHaloArray) = length(mha.arrays)
@inline Base.parent(mha::ArrayOfHaloArray) = mha.arrays

@inline function Base.size(mha::ArrayOfHaloArray)
    return (field_shape(mha)..., size(first(mha.arrays))...)
end

@inline Base.size(mha::ArrayOfHaloArray, i::Int) = size(mha)[i]
@inline Base.length(mha::ArrayOfHaloArray) = prod(size(mha))
@inline Base.axes(mha::ArrayOfHaloArray) = (map(Base.OneTo, field_shape(mha))..., axes(first(mha.arrays))...)
@inline Base.axes(mha::ArrayOfHaloArray, i::Int) = axes(mha)[i]
@inline Base.eachindex(mha::ArrayOfHaloArray) = CartesianIndices(axes(mha))

@inline function interior_size(mha::ArrayOfHaloArray)
    return (field_shape(mha)..., interior_size(first(mha.arrays))...)
end

@inline function full_size(mha::ArrayOfHaloArray)
    return (field_shape(mha)..., full_size(first(mha.arrays))...)
end

@inline full_size(mha::ArrayOfHaloArray, i::Int) = full_size(mha)[i]
@inline halo_width(mha::ArrayOfHaloArray) = halo_width(first(mha.arrays))
@inline halo_width(mha::ArrayOfHaloArray, i) = map(halo_width, mha.arrays)
@inline global_size(mha::ArrayOfHaloArray) = size(mha)

function Base.getindex(mha::ArrayOfHaloArray{T,N,Shape}, I...) where {T,N,Shape}
    field_ndims = length(Shape)

    if length(I) <= field_ndims
        return getindex(mha.arrays, I...)
    elseif length(I) == field_ndims + N
        field_index = ntuple(d -> I[d], Val(field_ndims))
        halo_index = ntuple(d -> I[field_ndims + d], Val(N))
        field = mha.arrays[field_index...]
        field isa ThreadedHaloArray &&
            throw(ArgumentError("global scalar indexing is not implemented for ThreadedHaloArray fields"))
        return getindex(interior_view(field), halo_index...)
    else
        throw(BoundsError(mha, I))
    end
end

function Base.setindex!(mha::ArrayOfHaloArray{T,N,Shape}, value, I...) where {T,N,Shape}
    field_ndims = length(Shape)

    if length(I) == field_ndims + N
        field_index = ntuple(d -> I[d], Val(field_ndims))
        halo_index = ntuple(d -> I[field_ndims + d], Val(N))
        field = mha.arrays[field_index...]
        field isa ThreadedHaloArray &&
            throw(ArgumentError("global scalar indexing is not implemented for ThreadedHaloArray fields"))
        setindex!(interior_view(field), value, halo_index...)
        return mha
    else
        throw(BoundsError(mha, I))
    end
end

function Base.similar(mha::ArrayOfHaloArray{AA,N,Shape,A}, ::Type{T},
        dims::NTuple{M,Int}) where {AA,N,Shape,A,T,M}
    arrs = map(a -> similar(a, T, dims), mha.arrays)
    return ArrayOfHaloArray{T,N,Shape,typeof(arrs)}(arrs)
end

function Base.similar(mha::ArrayOfHaloArray{AA,N,Shape,A}, ::Type{T}) where {AA,N,Shape,A,T}
    arrs = map(a -> similar(a, T), mha.arrays)
    return ArrayOfHaloArray{T,N,Shape,typeof(arrs)}(arrs)
end

function Base.similar(mha::ArrayOfHaloArray{AA,N,Shape,A}) where {AA,N,Shape,A}
    arrs = map(similar, mha.arrays)
    return ArrayOfHaloArray{AA,N,Shape,typeof(arrs)}(arrs)
end

function Base.copyto!(dest::ArrayOfHaloArray, src::ArrayOfHaloArray)
    field_shape(dest) == field_shape(src) ||
        throw(DimensionMismatch("ArrayOfHaloArray field shapes must match"))
    for I in eachindex(dest.arrays)
        copyto!(dest.arrays[I], src.arrays[I])
    end
    return dest
end

function Base.copy(mha::ArrayOfHaloArray)
    arrs = map(copy, mha.arrays)
    return ArrayOfHaloArray(arrs)
end

function Base.fill!(mha::ArrayOfHaloArray, value)
    foreach(a -> fill!(a, value), mha.arrays)
    return mha
end

function Base.map(f, mha::ArrayOfHaloArray)
    arrs = map(a -> map(f, a), mha.arrays)
    return ArrayOfHaloArray(arrs)
end

function foreach_field!(f!, mha::ArrayOfHaloArray)
    foreach(f!, mha.arrays)
    return nothing
end

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

function isactive(mha::ArrayOfHaloArray)
    return all(isactive, mha.arrays)
end

function Base.all(f, mha::ArrayOfHaloArray)
    return all(field -> all(f, field), mha.arrays)
end

function Base.any(f, mha::ArrayOfHaloArray)
    return any(field -> any(f, field), mha.arrays)
end
