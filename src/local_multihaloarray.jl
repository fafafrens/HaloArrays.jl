mutable struct LocalMultiHaloArray{T,N,A}
    arrays::A
end

function LocalMultiHaloArray(arrs::NamedTuple; check=false)
    field_names = keys(arrs)
    field_values = values(arrs)
    field_eltypes = map(eltype, field_values)
    TTypes = promote_type(field_eltypes...)
    N_ref = ndims(first(field_values))

    if check
        if !all(ndims(a) == N_ref for a in field_values)
            throw(ArgumentError("All LocalHaloArrays must have the same dimensionality N"))
        end

        ref_size = size(interior_view(first(field_values)))
        for (name, a) in zip(field_names, field_values)
            if size(interior_view(a)) != ref_size
                throw(DimensionMismatch("Field `$(name)` has interior size $(size(interior_view(a))) != $ref_size"))
            end
        end
    end

    return LocalMultiHaloArray{TTypes,N_ref,typeof(arrs)}(arrs)
end

function LocalMultiHaloArray(local_size::NTuple{N,Int}, halo::Int,
        bcs::NamedTuple{names,<:Tuple}) where {N,names}
    arrays = NamedTuple{names}(map(bcs) do bc
        LocalHaloArray(local_size, halo; boundary_condition=bc)
    end)
    return LocalMultiHaloArray(arrays)
end

Base.getindex(mha::LocalMultiHaloArray, name::Symbol) = mha.arrays[name]
Base.eltype(::LocalMultiHaloArray{T}) where {T} = T
Base.ndims(::LocalMultiHaloArray{T,N}) where {T,N} = N
Base.ndims(::Type{<:LocalMultiHaloArray{T,N}}) where {T,N} = N
@inline Base.size(mha::LocalMultiHaloArray) = (length(mha.arrays), size(first(mha.arrays))...)
@inline Base.size(mha::LocalMultiHaloArray, i::Int) = size(mha)[i]

n_field(halos::LocalMultiHaloArray) = length(halos.arrays)
@inline interior_size(halos::LocalMultiHaloArray) = (n_field(halos), map(interior_size, halos.arrays)...)
@inline full_size(halos::LocalMultiHaloArray) = (n_field(halos), map(full_size, halos.arrays)...)
@inline full_size(halos::LocalMultiHaloArray, i) = full_size(halos)[i]
@inline halo_width(halos::LocalMultiHaloArray, i) = map(halo_width, halos.arrays)
@inline Base.parent(halos::LocalMultiHaloArray) = map(parent, halos.arrays)

function Base.similar(mha::LocalMultiHaloArray{AA,N,A}, ::Type{T}, dims::NTuple{M,Int}) where {AA,N,A,T,M}
    arrs = map(a -> similar(a, T, dims), values(mha.arrays))
    return LocalMultiHaloArray(NamedTuple{keys(mha.arrays)}(arrs))
end

function Base.similar(mha::LocalMultiHaloArray{AA,N,A}, ::Type{T}) where {AA,N,A,T}
    arrs = map(a -> similar(a, T), values(mha.arrays))
    return LocalMultiHaloArray(NamedTuple{keys(mha.arrays)}(arrs))
end

function Base.similar(mha::LocalMultiHaloArray)
    arrs = map(similar, values(mha.arrays))
    return LocalMultiHaloArray(NamedTuple{keys(mha.arrays)}(arrs))
end

function Base.copy(mha::LocalMultiHaloArray)
    arrs = map(copy, values(mha.arrays))
    return LocalMultiHaloArray(NamedTuple{keys(mha.arrays)}(arrs))
end

function foreach_field!(f!, mha::LocalMultiHaloArray)
    for arr in values(mha.arrays)
        f!(arr)
    end
    return nothing
end

function foreach_field!(f!, mha::LocalMultiHaloArray, etc::Vararg{LocalMultiHaloArray})
    for (name, arr) in pairs(mha.arrays)
        f!(arr, map(x -> x.arrays[name], etc)...)
    end
    return nothing
end

function interior_view(mha::LocalMultiHaloArray)
    return NamedTuple{keys(mha.arrays)}(map(interior_view, values(mha.arrays)))
end

isactive(::LocalMultiHaloArray) = true
