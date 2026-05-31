mutable struct MultiHaloArray{T,N,A,D} <: AbstractHaloCollection{T,D}
    arrays::A
end

function _check_multihaloarray_compatible(field_names, field_values)
    isempty(field_values) && throw(ArgumentError("MultiHaloArray requires at least one field"))

    ref = first(field_values)
    ref_ndims = _spatial_ndims(ref)
    ref_size = _spatial_interior_size(ref)
    ref_halo_width = halo_width(ref)
    ref_backend = halo_backend(ref)

    if !all(_spatial_ndims(a) == ref_ndims for a in field_values)
        throw(ArgumentError("All MultiHaloArray fields must have the same spatial dimensionality"))
    end

    for (name, a) in zip(field_names, field_values)
        if _spatial_interior_size(a) != ref_size
            throw(DimensionMismatch("Field `$(name)` has interior size $(_spatial_interior_size(a)) != $ref_size"))
        end
        if halo_width(a) != ref_halo_width
            throw(DimensionMismatch("Field `$(name)` has halo width $(halo_width(a)) != $ref_halo_width"))
        end
        if !(halo_backend(a) isa typeof(ref_backend))
            throw(ArgumentError("Field `$(name)` has backend $(typeof(halo_backend(a))) != $(typeof(ref_backend))"))
        end
    end

    return nothing
end

function MultiHaloArray(arrs::NamedTuple; check=nothing)
    field_names = keys(arrs)
    field_values = values(arrs)
    _check_multihaloarray_compatible(field_names, field_values)

    field_eltypes = map(eltype, field_values)
   
    TTypes = promote_type(field_eltypes...)
    N_ref = _spatial_ndims(first(field_values))

    D = N_ref + 1
    return MultiHaloArray{TTypes,N_ref,typeof(arrs),D}(arrs)
end


function MultiHaloArray(::Type{<:HaloArray}, ::Type{T}, owned_dims::NTuple{N,Int},
        halo::Int, topology::CartesianTopology{N};
        boundary_conditions::NamedTuple{names,<:Tuple}) where {T,N,names}
    arrays = NamedTuple{names}(map(boundary_conditions) do bc
        HaloArray(T, owned_dims, halo, topology; boundary_condition=bc)
    end)
    return MultiHaloArray(arrays)
end

function MultiHaloArray(::Type{<:HaloArray}, ::Type{T}, owned_dims::NTuple{N,Int},
        halo::Int; boundary_conditions::NamedTuple{names,<:Tuple}) where {T,N,names}
    arrays = NamedTuple{names}(map(boundary_conditions) do bc
        HaloArray(T, owned_dims, halo; boundary_condition=bc)
    end)
    return MultiHaloArray(arrays)
end

function MultiHaloArray(::Type{<:HaloArray}, owned_dims::NTuple{N,Int},
        halo::Int; boundary_conditions::NamedTuple{names,<:Tuple}) where {N,names}
    return MultiHaloArray(HaloArray, Float64, owned_dims, halo;
        boundary_conditions=boundary_conditions)
end

function MultiHaloArray(::Type{T}, owned_dims::NTuple{N,Int}, halo::Int,
        topology::CartesianTopology{N}; boundary_conditions::NamedTuple{names,<:Tuple}) where {T,N,names}
    return MultiHaloArray(HaloArray, T, owned_dims, halo, topology;
        boundary_conditions=boundary_conditions)
end

function MultiHaloArray(::Type{T}, owned_dims::NTuple{N,Int}, halo::Int;
        boundary_conditions::NamedTuple{names,<:Tuple}) where {T,N,names}
    return MultiHaloArray(HaloArray, T, owned_dims, halo;
        boundary_conditions=boundary_conditions)
end

function MultiHaloArray(owned_dims::NTuple{N,Int}, halo::Int,
        boundary_conditions::NamedTuple{names,<:Tuple}) where {N,names}
    return MultiHaloArray(Float64, owned_dims, halo;
        boundary_conditions=boundary_conditions)
end

function MultiHaloArray(::Type{<:LocalHaloArray}, ::Type{T}, owned_dims::NTuple{N,Int},
        halo::Int; boundary_conditions::NamedTuple{names,<:Tuple}) where {T,N,names}
    arrays = NamedTuple{names}(map(boundary_conditions) do bc
        LocalHaloArray(T, owned_dims, halo; boundary_condition=bc)
    end)
    return MultiHaloArray(arrays)
end

function MultiHaloArray(::Type{<:LocalHaloArray}, owned_dims::NTuple{N,Int},
        halo::Int; boundary_conditions::NamedTuple{names,<:Tuple}) where {N,names}
    return MultiHaloArray(LocalHaloArray, Float64, owned_dims, halo;
        boundary_conditions=boundary_conditions)
end

function MultiHaloArray(::Type{<:ThreadedHaloArray}, ::Type{T},
        tile_size::NTuple{N,<:Integer}, halo::Integer;
        dims::NTuple{N,<:Integer},
        boundary_conditions::NamedTuple{names,<:Tuple}) where {T,N,names}
    arrays = NamedTuple{names}(map(boundary_conditions) do bc
        ThreadedHaloArray(T, tile_size, halo; dims=dims, boundary_condition=bc)
    end)
    return MultiHaloArray(arrays)
end

function MultiHaloArray(::Type{<:ThreadedHaloArray}, tile_size::NTuple{N,<:Integer},
        halo::Integer; dims::NTuple{N,<:Integer},
        boundary_conditions::NamedTuple{names,<:Tuple}) where {N,names}
    return MultiHaloArray(ThreadedHaloArray, Float64, tile_size, halo;
        dims=dims, boundary_conditions=boundary_conditions)
end


Base.getindex(mha::MultiHaloArray, name::Symbol) = mha.arrays[name]

Base.eltype(mha::MultiHaloArray{T,N,A,D}) where {T,N,A,D} = T
Base.eltype(::Type{<:MultiHaloArray{T,N,A,D}}) where {T,N,A,D} = T
Base.ndims(mha::MultiHaloArray{T,N,A,D}) where {T,N,A,D} = D
Base.ndims(::Type{<:MultiHaloArray{T,N,A,D}}) where {T,N,A,D} = D

@inline Base.size(mha::MultiHaloArray) = global_size(mha)

@inline Base.size(mha::MultiHaloArray, i::Int) = size(mha)[i]
@inline Base.length(mha::MultiHaloArray) = prod(size(mha))
@inline Base.eachindex(mha::MultiHaloArray) = CartesianIndices(axes(mha))

n_field(halos::MultiHaloArray{T,N,A,D}) where {T,N,A,D} = length(halos.arrays)

@inline interior_size(halos::MultiHaloArray) = (n_field(halos), _spatial_interior_size(first(values(halos.arrays)))...)
@inline owned_size(halos::MultiHaloArray) = (n_field(halos), _spatial_owned_size(first(values(halos.arrays)))...)
@inline global_size(halos::MultiHaloArray) = (n_field(halos), _spatial_global_size(first(values(halos.arrays)))...)
@inline storage_size(halos::MultiHaloArray) = (n_field(halos), _spatial_storage_size(first(values(halos.arrays)))...)
@inline storage_size(halos::MultiHaloArray,i) = storage_size(halos)[i]
@inline halo_width(halo::MultiHaloArray) = halo_width(first(values(halo.arrays)))
@inline halo_width(halo::MultiHaloArray,i)= map(halo_width,halo.arrays)
@inline Base.parent(halo::MultiHaloArray)  = map(parent,halo.arrays)
@inline Base.axes(x::MultiHaloArray) = (Base.OneTo(n_field(x)), _spatial_axes(first(values(x.arrays)))...)
@inline Base.axes(x::MultiHaloArray,i) = axes(x)[i]
@inline owned_axes(x::MultiHaloArray) = (Base.OneTo(n_field(x)), _spatial_owned_axes(first(values(x.arrays)))...)
@inline owned_axes(x::MultiHaloArray,i) = owned_axes(x)[i]
@inline tile_size(halos::MultiHaloArray) = tile_size(first(values(halos.arrays)))
@inline tile_count(halos::MultiHaloArray) = tile_count(first(values(halos.arrays)))
@inline tile_parent(halos::MultiHaloArray, tile_id::Integer) =
    NamedTuple{keys(halos.arrays)}(map(a -> tile_parent(a, tile_id), values(halos.arrays)))
@inline tile_coordinates(halos::MultiHaloArray, tile_id::Integer) =
    tile_coordinates(first(values(halos.arrays)), tile_id)
@inline neighbor_tile_id(halos::MultiHaloArray, tile_id::Integer, dim::Integer, side::Integer) =
    neighbor_tile_id(first(values(halos.arrays)), tile_id, dim, side)
@inline halo_backend(halos::MultiHaloArray) = halo_backend(first(values(halos.arrays)))

to_tuple(mha::MultiHaloArray) = (mha.arrays...,)

function Base.getindex(mha::MultiHaloArray, field_index::Integer)
    1 <= field_index <= n_field(mha) || throw(BoundsError(mha, (field_index,)))
    return values(mha.arrays)[field_index]
end

function Base.getindex(mha::MultiHaloArray, field_index::Integer, I::Vararg{Integer})
    return getindex(getindex(mha, field_index), I...)
end

function Base.setindex!(mha::MultiHaloArray, value, field_index::Integer, I::Vararg{Integer})
    setindex!(getindex(mha, field_index), value, I...)
    return mha
end

function Base.similar(mha::MultiHaloArray{AA,N,A,D}, ::Type{T}, dims::Dims{M}) where {AA,N,A,D,T,M}
    M == N + 1 || throw(DimensionMismatch("MultiHaloArray similar dims must have $(N + 1) dimensions"))
    Int(dims[1]) == n_field(mha) ||
        throw(DimensionMismatch("MultiHaloArray similar cannot change the named field count from $(n_field(mha)) to $(dims[1])"))
    spatial_dims = ntuple(d -> Int(dims[d + 1]), Val(N))

    arrs = map(a -> similar(a, T, spatial_dims), values(mha.arrays))
    names = keys(mha.arrays)
    nt = NamedTuple{names}(arrs)
    return MultiHaloArray(nt)
end

Base.similar(mha::MultiHaloArray{AA,N,A,D}, ::Type{T},
    dims::NTuple{M,<:Integer}) where {AA,N,A,D,T,M} =
    similar(mha, T, ntuple(d -> Int(dims[d]), Val(M)))

Base.similar(mha::MultiHaloArray, dims::Dims{M}) where {M} =
    similar(mha, eltype(mha), dims)
Base.similar(mha::MultiHaloArray, dims::NTuple{M,<:Integer}) where {M} =
    similar(mha, eltype(mha), dims)

function Base.similar(mha::MultiHaloArray{AA,N,A,D}, ::Type{T}) where {AA,N,A,D,T}
    arrs = map(a -> similar(a, T), values(mha.arrays))
    names = keys(mha.arrays)
    nt = NamedTuple{names}(arrs)
    return MultiHaloArray(nt)
end


function Base.similar(mha::MultiHaloArray{AA,N,A,D}) where {AA,N,A,D}
    arrs = map(a -> similar(a), values(mha.arrays))
    names = keys(mha.arrays)
    nt = NamedTuple{names}(arrs)
    return MultiHaloArray(nt)
end


function Base.copy(mha::MultiHaloArray)
    newfields = map(x -> copy(x), values(mha.arrays))
    new_ntuple = NamedTuple{keys(mha.arrays)}(newfields)
    return MultiHaloArray(new_ntuple)
end

function Base.copyto!(dest::MultiHaloArray, src::MultiHaloArray)
    keys(dest.arrays) == keys(src.arrays) ||
        throw(DimensionMismatch("MultiHaloArray copyto! requires matching field names"))
    for name in keys(dest.arrays)
        copyto!(dest.arrays[name], src.arrays[name])
    end
    return dest
end

function Base.fill!(mha::MultiHaloArray, value)
    foreach(field -> fill!(field, value), values(mha.arrays))
    return mha
end

function Base.zero(mha::MultiHaloArray)
    z = similar(mha)
    fill!(z, zero(eltype(mha)))
    return z
end

function Base.map(f, mha::MultiHaloArray)
    newfields = map(x -> map(f, x), values(mha.arrays))
    new_ntuple = NamedTuple{keys(mha.arrays)}(newfields)
    return MultiHaloArray(new_ntuple)
end

function foreach_field!(f!, mha::MultiHaloArray)
    for  arr in values(mha.arrays)
        f!(arr)
    end

    return nothing
end


function foreach_field!(f!, mha::MultiHaloArray, etc::Vararg{MultiHaloArray})
    for (name, arr) in pairs(mha.arrays)
        f!(arr, map(x -> x.arrays[name], etc)...)
    end
    return nothing
end

function map_over_field(f, mha::MultiHaloArray)

    return map(f, mha.arrays)

end

function map_over_field(f, mha::MultiHaloArray,etc::Vararg{MultiHaloArray})
    n_fields = n_field(mha)
    keyset = keys(mha.arrays)

    result = ntuple(n_fields) do n
        f(mha.arrays[n], map(x -> x.arrays[n], etc)...)
    end

    return NamedTuple{keyset}(result)
end

"""
    interior_view(mha::MultiHaloArray)

Return a `NamedTuple` with the interior view of each field.
"""
function interior_view(mha::MultiHaloArray)
    return NamedTuple{keys(mha.arrays)}(map(interior_view, values(mha.arrays)))
end

function interior_view(mha::MultiHaloArray, tile_id::Integer)
    return NamedTuple{keys(mha.arrays)}(map(a -> interior_view(a, tile_id), values(mha.arrays)))
end

function isactive(mha::MultiHaloArray)
    return all(isactive, values(mha.arrays))
end

function is_root(mha::MultiHaloArray; root::Integer=0)
    return is_root(first(values(mha.arrays)); root=root)
end

function active_fields(mha::MultiHaloArray)
    return (; (name => isactive(ha) for (name, ha) in mha.arrays)...)
end
