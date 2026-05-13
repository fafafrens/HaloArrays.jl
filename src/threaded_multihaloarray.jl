mutable struct ThreadedMultiHaloArray{T,N,A}
    arrays::A
end

function ThreadedMultiHaloArray(arrs::NamedTuple; check=false)
    field_names = keys(arrs)
    field_values = values(arrs)
    field_eltypes = map(eltype, field_values)
    TTypes = promote_type(field_eltypes...)
    N_ref = ndims(first(field_values))

    if check
        if !all(a -> a isa ThreadedHaloArray, field_values)
            throw(ArgumentError("All fields must be ThreadedHaloArray"))
        end

        if !all(ndims(a) == N_ref for a in field_values)
            throw(ArgumentError("All ThreadedHaloArrays must have the same dimensionality N"))
        end

        ref = first(field_values)
        for (name, a) in zip(field_names, field_values)
            tile_size(a) == tile_size(ref) ||
                throw(DimensionMismatch("Field `$(name)` has tile_size $(tile_size(a)) != $(tile_size(ref))"))
            halo_width(a) == halo_width(ref) ||
                throw(DimensionMismatch("Field `$(name)` has halo width $(halo_width(a)) != $(halo_width(ref))"))
            a.topology.dims == ref.topology.dims ||
                throw(DimensionMismatch("Field `$(name)` has topology dims $(a.topology.dims) != $(ref.topology.dims)"))
            tile_count(a) == tile_count(ref) ||
                throw(DimensionMismatch("Field `$(name)` has tile_count $(tile_count(a)) != $(tile_count(ref))"))
        end
    end

    return ThreadedMultiHaloArray{TTypes,N_ref,typeof(arrs)}(arrs)
end

function ThreadedMultiHaloArray(::Type{T}, tile_size::NTuple{N,<:Integer}, halo::Integer;
        dims::NTuple{N,<:Integer},
        boundary_conditions::NamedTuple{names,<:Tuple}) where {T,N,names}
    arrays = NamedTuple{names}(map(boundary_conditions) do bc
        ThreadedHaloArray(T, tile_size, halo; dims=dims, boundary_condition=bc)
    end)
    return ThreadedMultiHaloArray(arrays; check=true)
end

ThreadedMultiHaloArray(tile_size::NTuple{N,<:Integer}, halo::Integer; kwargs...) where {N} =
    ThreadedMultiHaloArray(Float64, tile_size, halo; kwargs...)

Base.getindex(mha::ThreadedMultiHaloArray, name::Symbol) = mha.arrays[name]
Base.eltype(::ThreadedMultiHaloArray{T}) where {T} = T
Base.ndims(::ThreadedMultiHaloArray{T,N}) where {T,N} = N
Base.ndims(::Type{<:ThreadedMultiHaloArray{T,N}}) where {T,N} = N
@inline Base.size(mha::ThreadedMultiHaloArray) = (length(mha.arrays), size(first(mha.arrays))...)
@inline Base.size(mha::ThreadedMultiHaloArray, i::Int) = size(mha)[i]

n_field(halos::ThreadedMultiHaloArray) = length(halos.arrays)
@inline tile_size(halos::ThreadedMultiHaloArray) = tile_size(first(values(halos.arrays)))
@inline tile_count(halos::ThreadedMultiHaloArray) = tile_count(first(values(halos.arrays)))
@inline tile_parent(halos::ThreadedMultiHaloArray, tile_id::Integer) =
    NamedTuple{keys(halos.arrays)}(map(a -> tile_parent(a, tile_id), values(halos.arrays)))
@inline tile_coordinates(halos::ThreadedMultiHaloArray, tile_id::Integer) =
    tile_coordinates(first(values(halos.arrays)), tile_id)
@inline interior_size(halos::ThreadedMultiHaloArray) = (n_field(halos), map(interior_size, halos.arrays)...)
@inline full_size(halos::ThreadedMultiHaloArray) = (n_field(halos), map(full_size, halos.arrays)...)
@inline full_size(halos::ThreadedMultiHaloArray, i) = full_size(halos)[i]
@inline halo_width(halos::ThreadedMultiHaloArray) = halo_width(first(values(halos.arrays)))
@inline halo_width(halos::ThreadedMultiHaloArray, i) = map(halo_width, halos.arrays)
@inline Base.parent(halos::ThreadedMultiHaloArray) = map(parent, halos.arrays)
@inline global_size(halos::ThreadedMultiHaloArray) = size(halos)

function Base.similar(mha::ThreadedMultiHaloArray{AA,N,A}, ::Type{T}, dims::NTuple{M,Int}) where {AA,N,A,T,M}
    arrs = map(a -> similar(a, T, dims), values(mha.arrays))
    return ThreadedMultiHaloArray(NamedTuple{keys(mha.arrays)}(arrs))
end

function Base.similar(mha::ThreadedMultiHaloArray{AA,N,A}, ::Type{T}) where {AA,N,A,T}
    arrs = map(a -> similar(a, T), values(mha.arrays))
    return ThreadedMultiHaloArray(NamedTuple{keys(mha.arrays)}(arrs))
end

function Base.similar(mha::ThreadedMultiHaloArray)
    arrs = map(similar, values(mha.arrays))
    return ThreadedMultiHaloArray(NamedTuple{keys(mha.arrays)}(arrs))
end

function Base.copy(mha::ThreadedMultiHaloArray)
    arrs = map(copy, values(mha.arrays))
    return ThreadedMultiHaloArray(NamedTuple{keys(mha.arrays)}(arrs))
end

function foreach_field!(f!, mha::ThreadedMultiHaloArray)
    for arr in values(mha.arrays)
        f!(arr)
    end
    return nothing
end

function foreach_field!(f!, mha::ThreadedMultiHaloArray, etc::Vararg{ThreadedMultiHaloArray})
    for (name, arr) in pairs(mha.arrays)
        f!(arr, map(x -> x.arrays[name], etc)...)
    end
    return nothing
end

function interior_view(mha::ThreadedMultiHaloArray)
    return NamedTuple{keys(mha.arrays)}(map(interior_view, values(mha.arrays)))
end

function interior_view(mha::ThreadedMultiHaloArray, tile_id::Integer)
    return NamedTuple{keys(mha.arrays)}(map(a -> interior_view(a, tile_id), values(mha.arrays)))
end

function halo_exchange!(mha::ThreadedMultiHaloArray)
    foreach_field!(halo_exchange!, mha)
    return mha
end

function boundary_condition!(mha::ThreadedMultiHaloArray)
    foreach_field!(boundary_condition!, mha)
    return nothing
end

function synchronize_halo!(mha::ThreadedMultiHaloArray)
    foreach_field!(synchronize_halo!, mha)
    return mha
end

start_halo_exchange!(mha::ThreadedMultiHaloArray) = halo_exchange!(mha)
finish_halo_exchange!(mha::ThreadedMultiHaloArray) = mha
isactive(::ThreadedMultiHaloArray) = true
