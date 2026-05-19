# MultiHaloArray using NamedTuple to store fields
mutable struct MultiHaloArray{T, N, A } <: AbstractHaloCollection
    arrays::A  # NamedTuple of HaloArrays
end

function _check_multihaloarray_compatible(field_names, field_values)
    isempty(field_values) && throw(ArgumentError("MultiHaloArray requires at least one field"))

    ref = first(field_values)
    ref_ndims = _spatial_ndims(ref)
    ref_size = _spatial_interior_size(ref)
    ref_halo_width = halo_width(ref)

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
    end

    return nothing
end

function MultiHaloArray(arrs::NamedTuple; check=nothing)
    field_names = keys(arrs)
    field_values = values(arrs)
    _check_multihaloarray_compatible(field_names, field_values)

    # Infer per-field eltype
    field_eltypes = map(eltype, field_values)
   
    TTypes = promote_type(field_eltypes...)
    N_ref = _spatial_ndims(first(field_values))

    #newarr=map(field_values) do data 
    #    HaloArray(convert.(TTypes,data.data),halo_width(data), data.topology,
    #data.boundary_condition)
    #end
    
    #ntup =NamedTuple{field_names}(newarr)
    #Len=length(field_values)
    return MultiHaloArray{TTypes, N_ref ,typeof(arrs)}(arrs)
end


function MultiHaloArray(::Type{<:HaloArray}, ::Type{T}, local_size::NTuple{N,Int},
        halo::Int, topology::CartesianTopology{N};
        boundary_conditions::NamedTuple{names,<:Tuple}) where {T,N,names}
    arrays = NamedTuple{names}(map(boundary_conditions) do bc
        HaloArray(T, local_size, halo, topology; boundary_condition=bc)
    end)
    return MultiHaloArray(arrays)
end

function MultiHaloArray(::Type{<:HaloArray}, ::Type{T}, local_size::NTuple{N,Int},
        halo::Int; boundary_conditions::NamedTuple{names,<:Tuple}) where {T,N,names}
    arrays = NamedTuple{names}(map(boundary_conditions) do bc
        HaloArray(T, local_size, halo; boundary_condition=bc)
    end)
    return MultiHaloArray(arrays)
end

function MultiHaloArray(::Type{<:HaloArray}, local_size::NTuple{N,Int},
        halo::Int; boundary_conditions::NamedTuple{names,<:Tuple}) where {N,names}
    return MultiHaloArray(HaloArray, Float64, local_size, halo;
        boundary_conditions=boundary_conditions)
end

function MultiHaloArray(::Type{T}, local_size::NTuple{N,Int}, halo::Int,
        topology::CartesianTopology{N}; boundary_conditions::NamedTuple{names,<:Tuple}) where {T,N,names}
    return MultiHaloArray(HaloArray, T, local_size, halo, topology;
        boundary_conditions=boundary_conditions)
end

function MultiHaloArray(::Type{T}, local_size::NTuple{N,Int}, halo::Int;
        boundary_conditions::NamedTuple{names,<:Tuple}) where {T,N,names}
    return MultiHaloArray(HaloArray, T, local_size, halo;
        boundary_conditions=boundary_conditions)
end

function MultiHaloArray(local_size::NTuple{N,Int}, halo::Int,
        boundary_conditions::NamedTuple{names,<:Tuple}) where {N,names}
    return MultiHaloArray(Float64, local_size, halo;
        boundary_conditions=boundary_conditions)
end

function MultiHaloArray(::Type{<:LocalHaloArray}, ::Type{T}, local_size::NTuple{N,Int},
        halo::Int; boundary_conditions::NamedTuple{names,<:Tuple}) where {T,N,names}
    arrays = NamedTuple{names}(map(boundary_conditions) do bc
        LocalHaloArray(T, local_size, halo; boundary_condition=bc)
    end)
    return MultiHaloArray(arrays)
end

function MultiHaloArray(::Type{<:LocalHaloArray}, local_size::NTuple{N,Int},
        halo::Int; boundary_conditions::NamedTuple{names,<:Tuple}) where {N,names}
    return MultiHaloArray(LocalHaloArray, Float64, local_size, halo;
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


# Access field by symbol name
Base.getindex(mha::MultiHaloArray, name::Symbol) = mha.arrays[name]


#Base.setproperty!(mha::MultiHaloArray, name::Symbol, value) = (mha.arrays[name] = value)



# Metadata helpers
Base.eltype(mha::MultiHaloArray{T, N, A}) where {T,N,A} = T
Base.ndims(mha::MultiHaloArray{T, N, A}) where {T,N,A} = N
Base.ndims(::Type{MultiHaloArray{T, N, A}}) where {T,N,A} = N

# Size includes field axis
@inline Base.size(mha::MultiHaloArray) = (length(mha.arrays), _spatial_size(first(values(mha.arrays)))...)

@inline Base.size(mha::MultiHaloArray, i::Int) = size(mha)[i]

n_field(halos::MultiHaloArray{T,N,A}) where {T,N,A} = length(halos.arrays)

#Base.length(halo::HaloArray) = length(halo.data)


@inline interior_size(halos::MultiHaloArray) = (n_field(halos), _spatial_interior_size(first(values(halos.arrays)))...)
@inline full_size(halos::MultiHaloArray) = (n_field(halos), _spatial_full_size(first(values(halos.arrays)))...)
@inline full_size(halos::MultiHaloArray,i) = full_size(halos)[i]
@inline halo_width(halo::MultiHaloArray) = halo_width(first(values(halo.arrays)))
@inline halo_width(halo::MultiHaloArray,i)= map(halo_width,halo.arrays)
@inline Base.parent(halo::MultiHaloArray)  = map(parent,halo.arrays)
@inline Base.axes(x::MultiHaloArray) = (Base.OneTo(n_field(x)), _spatial_axes(first(values(x.arrays)))...)
@inline Base.axes(x::MultiHaloArray,i) = axes(x)[i]
@inline tile_size(halos::MultiHaloArray) = tile_size(first(values(halos.arrays)))
@inline tile_count(halos::MultiHaloArray) = tile_count(first(values(halos.arrays)))
@inline tile_parent(halos::MultiHaloArray, tile_id::Integer) =
    NamedTuple{keys(halos.arrays)}(map(a -> tile_parent(a, tile_id), values(halos.arrays)))
@inline tile_coordinates(halos::MultiHaloArray, tile_id::Integer) =
    tile_coordinates(first(values(halos.arrays)), tile_id)

to_tuple(mha::MultiHaloArray) = (mha.arrays...,)

function Base.similar(mha::MultiHaloArray{AA, N, A}, ::Type{T},dims::NTuple{M,Int64}) where {AA,N, A,T,M}

    arrs = map(a -> similar(a, T, dims), values(mha.arrays))
    names = keys(mha.arrays)
    nt = NamedTuple{names}(arrs)
    return MultiHaloArray{T, N, typeof(nt)}(nt)
end

function Base.similar(mha::MultiHaloArray{AA, N, A}, ::Type{T}) where {AA,N, A, T}
    arrs = map(a -> similar(a, T), values(mha.arrays))
    names = keys(mha.arrays)
    nt = NamedTuple{names}(arrs)
    return MultiHaloArray{T, N, typeof(nt)}(nt)
end


function Base.similar(mha::MultiHaloArray{AA, N, A}) where {AA,N, A}
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

function Base.map(f, mha::MultiHaloArray)
    newfields = map(x -> map(f, x), values(mha.arrays))
    # newfields is a Vector of mapped HaloArrays, but you want a NamedTuple
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
    # Ensure all MultiHaloArrays have the same number of fields
    keyset = keys(mha.arrays)

    # Map over each field across all MultiHaloArrays
    result = ntuple(n_fields) do n
        f(mha.arrays[n], map(x -> x.arrays[n], etc)...)
    end

    return NamedTuple{keyset}(result)
end

"""
    interior_view(mha::MultiHaloArray)

Restituisce un NamedTuple con le interior view di ciascun campo del MultiHaloArray.
I campi del NamedTuple hanno gli stessi simboli del `mha.arrays`.
"""
function interior_view(mha::MultiHaloArray)
    return NamedTuple{keys(mha.arrays)}(map(interior_view, values(mha.arrays)))
end

function interior_view(mha::MultiHaloArray, tile_id::Integer)
    return NamedTuple{keys(mha.arrays)}(map(a -> interior_view(a, tile_id), values(mha.arrays)))
end

# isactive per MultiHaloArray: true se almeno un campo è active
# richiede che isactive sia definita per i singoli HaloArray
function isactive(mha::MultiHaloArray)
    return all(isactive, values(mha.arrays))
end

# utility: ritorna NamedTuple mapping field -> Bool (active per campo)
function active_fields(mha::MultiHaloArray)
    return (; (name => isactive(ha) for (name, ha) in mha.arrays)...)
end
