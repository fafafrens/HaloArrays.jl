# Multi-field halo container using an AbstractArray to store the fields.
mutable struct ArrayOfHaloArray{T,N,Shape,A,D} <: AbstractHaloCollection{T,D}
    arrays::A
end

const HaloArrayField = AbstractSingleHaloArray

@inline _spatial_ndims(x) = ndims(x)
@inline _spatial_owned_size(x) = owned_size(x)
@inline _spatial_interior_size(x) = interior_size(x)
@inline _spatial_global_size(x) = global_size(x)
@inline _spatial_storage_size(x) = storage_size(x)
@inline _spatial_axes(x) = axes(x)
@inline _spatial_owned_axes(x) = owned_axes(x)

@inline _spatial_ndims(::ArrayOfHaloArray{T,N}) where {T,N} = N
@inline _spatial_owned_size(x::ArrayOfHaloArray) = owned_size(first(parent(x)))
@inline _spatial_interior_size(x::ArrayOfHaloArray) = interior_size(first(parent(x)))
@inline _spatial_global_size(x::ArrayOfHaloArray) = global_size(first(parent(x)))
@inline _spatial_storage_size(x::ArrayOfHaloArray) = storage_size(first(parent(x)))
@inline _spatial_axes(x::ArrayOfHaloArray) = axes(first(parent(x)))
@inline _spatial_owned_axes(x::ArrayOfHaloArray) = owned_axes(first(parent(x)))

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
    ref_backend = halo_backend(ref)

    for I in CartesianIndices(arrays)
        a = arrays[I]
        ndims(a) == ref_ndims ||
            throw(ArgumentError("Field `$I` has dimensionality $(ndims(a)) != $ref_ndims"))
        interior_size(a) == ref_interior_size ||
            throw(DimensionMismatch("Field `$I` has interior size $(interior_size(a)) != $ref_interior_size"))
        halo_width(a) == ref_halo_width ||
            throw(DimensionMismatch("Field `$I` has halo width $(halo_width(a)) != $ref_halo_width"))
        halo_backend(a) isa typeof(ref_backend) ||
            throw(ArgumentError("Field `$I` has backend $(typeof(halo_backend(a))) != $(typeof(ref_backend))"))
    end

    return nothing
end

function ArrayOfHaloArray(arrays::AbstractArray; check=nothing)
    _check_arrayofhaloarray_compatible(arrays)

    T = promote_type(map(eltype, arrays)...)
    N = ndims(first(arrays))
    Shape = size(arrays)
    D = N + length(Shape)
    return ArrayOfHaloArray{T,N,Shape,typeof(arrays),D}(arrays)
end

function ArrayOfHaloArray(::Type{T}, owned_dims::NTuple{N,Int}, halo::Int,
        topology::CartesianTopology{N}; boundary_conditions::AbstractArray) where {T,N}
    arrays = map(boundary_conditions) do bc
        HaloArray(T, owned_dims, halo, topology; boundary_condition=bc)
    end
    return ArrayOfHaloArray(arrays)
end

function ArrayOfHaloArray(::Type{T}, owned_dims::NTuple{N,Int}, halo::Int,
        topology::CartesianTopology{N}, boundary_conditions::AbstractArray) where {T,N}
    return ArrayOfHaloArray(T, owned_dims, halo, topology; boundary_conditions=boundary_conditions)
end

function ArrayOfHaloArray(::Type{T}, owned_dims::NTuple{N,Int}, halo::Int;
        boundary_conditions::AbstractArray) where {T,N}
    arrays = map(boundary_conditions) do bc
        HaloArray(T, owned_dims, halo; boundary_condition=bc)
    end
    return ArrayOfHaloArray(arrays)
end

function ArrayOfHaloArray(::Type{T}, owned_dims::NTuple{N,Int}, halo::Int,
        boundary_conditions::AbstractArray) where {T,N}
    return ArrayOfHaloArray(T, owned_dims, halo; boundary_conditions=boundary_conditions)
end

function ArrayOfHaloArray(owned_dims::NTuple{N,Int}, halo::Int;
        boundary_conditions::AbstractArray) where {N}
    return ArrayOfHaloArray(Float64, owned_dims, halo; boundary_conditions=boundary_conditions)
end

function ArrayOfHaloArray(owned_dims::NTuple{N,Int}, halo::Int,
        boundary_conditions::AbstractArray) where {N}
    return ArrayOfHaloArray(Float64, owned_dims, halo; boundary_conditions=boundary_conditions)
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

Base.eltype(::ArrayOfHaloArray{T}) where {T} = T
Base.eltype(::Type{<:ArrayOfHaloArray{T}}) where {T} = T
Base.ndims(::ArrayOfHaloArray{T,N,Shape,A,D}) where {T,N,Shape,A,D} = D
Base.ndims(::Type{<:ArrayOfHaloArray{T,N,Shape,A,D}}) where {T,N,Shape,A,D} = D
@inline field_shape(::ArrayOfHaloArray{T,N,Shape}) where {T,N,Shape} = Shape
@inline n_field(mha::ArrayOfHaloArray) = length(mha.arrays)
@inline Base.parent(mha::ArrayOfHaloArray) = mha.arrays
@inline halo_backend(mha::ArrayOfHaloArray) = halo_backend(first(parent(mha)))

@inline function Base.size(mha::ArrayOfHaloArray)
    return global_size(mha)
end

@inline Base.size(mha::ArrayOfHaloArray, i::Int) = size(mha)[i]
@inline Base.length(mha::ArrayOfHaloArray) = prod(size(mha))
@inline Base.axes(mha::ArrayOfHaloArray) = (map(Base.OneTo, field_shape(mha))..., axes(first(mha.arrays))...)
@inline Base.axes(mha::ArrayOfHaloArray, i::Int) = axes(mha)[i]
@inline Base.eachindex(mha::ArrayOfHaloArray) = CartesianIndices(axes(mha))
@inline owned_axes(mha::ArrayOfHaloArray) = (map(Base.OneTo, field_shape(mha))..., owned_axes(first(mha.arrays))...)
@inline owned_axes(mha::ArrayOfHaloArray, i::Int) = owned_axes(mha)[i]

@inline function interior_size(mha::ArrayOfHaloArray)
    return (field_shape(mha)..., interior_size(first(mha.arrays))...)
end

@inline function owned_size(mha::ArrayOfHaloArray)
    return (field_shape(mha)..., owned_size(first(mha.arrays))...)
end

@inline function storage_size(mha::ArrayOfHaloArray)
    return (field_shape(mha)..., storage_size(first(mha.arrays))...)
end

@inline storage_size(mha::ArrayOfHaloArray, i::Int) = storage_size(mha)[i]
@inline halo_width(mha::ArrayOfHaloArray) = halo_width(first(mha.arrays))
@inline halo_width(mha::ArrayOfHaloArray, i) = map(halo_width, mha.arrays)
@inline global_size(mha::ArrayOfHaloArray) = (field_shape(mha)..., global_size(first(mha.arrays))...)

function Base.getindex(mha::ArrayOfHaloArray{T,N,Shape}, I...) where {T,N,Shape}
    field_ndims = length(Shape)

    if length(I) <= field_ndims
        return getindex(mha.arrays, I...)
    elseif length(I) == length(size(mha))
        field_idx = ntuple(d -> I[d], field_ndims)
        spatial_idx = ntuple(d -> I[field_ndims + d], N)
        return getindex(mha.arrays[field_idx...], spatial_idx...)
    else
        throw(BoundsError(mha, I))
    end
end

Base.getindex(mha::ArrayOfHaloArray, I::CartesianIndex) = getindex(mha, Tuple(I)...)

function Base.setindex!(mha::ArrayOfHaloArray{T,N,Shape}, value, I...) where {T,N,Shape}
    field_ndims = length(Shape)

    if length(I) == length(size(mha))
        field_idx = ntuple(d -> I[d], field_ndims)
        spatial_idx = ntuple(d -> I[field_ndims + d], N)
        setindex!(mha.arrays[field_idx...], value, spatial_idx...)
        return mha
    else
        throw(BoundsError(mha, I))
    end
end

Base.setindex!(mha::ArrayOfHaloArray, value, I::CartesianIndex) =
    setindex!(mha, value, Tuple(I)...)

function Base.similar(mha::ArrayOfHaloArray{AA,N,Shape,A,D}, ::Type{T},
        dims::Dims{M}) where {AA,N,Shape,A,D,T,M}
    field_ndims = length(Shape)
    M == field_ndims + N ||
        throw(DimensionMismatch("ArrayOfHaloArray similar dims must have $(field_ndims + N) dimensions"))

    new_field_shape = ntuple(d -> Int(dims[d]), Val(field_ndims))
    spatial_dims = ntuple(d -> Int(dims[field_ndims + d]), Val(N))

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

Base.similar(mha::ArrayOfHaloArray{AA,N,Shape,A,D}, ::Type{T},
    dims::NTuple{M,<:Integer}) where {AA,N,Shape,A,D,T,M} =
    similar(mha, T, ntuple(d -> Int(dims[d]), Val(M)))

Base.similar(mha::ArrayOfHaloArray, dims::Dims{M}) where {M} =
    similar(mha, eltype(mha), dims)
Base.similar(mha::ArrayOfHaloArray, dims::NTuple{M,<:Integer}) where {M} =
    similar(mha, eltype(mha), dims)

function Base.similar(mha::ArrayOfHaloArray{AA,N,Shape,A,D}, ::Type{T}) where {AA,N,Shape,A,D,T}
    arrs = map(a -> similar(a, T), mha.arrays)
    return ArrayOfHaloArray{T,N,Shape,typeof(arrs),D}(arrs)
end

function Base.similar(mha::ArrayOfHaloArray{AA,N,Shape,A,D}) where {AA,N,Shape,A,D}
    arrs = map(similar, mha.arrays)
    return ArrayOfHaloArray{AA,N,Shape,typeof(arrs),D}(arrs)
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

function Base.zero(mha::ArrayOfHaloArray)
    z = similar(mha)
    fill!(z, zero(eltype(mha)))
    return z
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

function is_root(mha::ArrayOfHaloArray; root::Integer=0)
    return is_root(first(mha.arrays); root=root)
end

@inline tile_count(mha::ArrayOfHaloArray) = tile_count(first(parent(mha)))
@inline tile_size(mha::ArrayOfHaloArray) = tile_size(first(parent(mha)))
@inline tile_coordinates(mha::ArrayOfHaloArray, tile_id::Integer) =
    tile_coordinates(first(parent(mha)), tile_id)
@inline neighbor_tile_id(mha::ArrayOfHaloArray, tile_id::Integer, dim::Integer, side::Integer) =
    neighbor_tile_id(first(parent(mha)), tile_id, dim, side)

function Base.all(f::F, mha::ArrayOfHaloArray) where {F<:Function}
    return all(field -> all(f, field), mha.arrays)
end

function Base.any(f::F, mha::ArrayOfHaloArray) where {F<:Function}
    return any(field -> any(f, field), mha.arrays)
end
