# MultiHaloArray using NamedTuple to store fields
struct MultiHaloArray{T, N, A , Len } #<: AbstractArray{T, N}
    arrays::A  # NamedTuple of HaloArrays
end


function MultiHaloArray(arrs::NamedTuple)
    field_names = keys(arrs)
    field_values = values(arrs)

    # Infer per-field eltype
    field_eltypes = map(eltype, field_values)
   
    TTypes = promote_type(field_eltypes...)
  

    # Check all have same dimensionality
    N_ref = ndims(first(field_values))
    if !all(ndims(a) == N_ref for a in field_values)
        throw(ArgumentError("All HaloArrays must have the same dimensionality N"))
    end

    # Check all have same interior size
    ref_size = size(interior_view(first(field_values)))
    for (name, a) in zip(field_names, field_values)
        if size(interior_view(a)) != ref_size
            throw(DimensionMismatch("Field `$(name)` has interior size $(size(interior_view(a))) ≠ $ref_size"))
        end
    end

    newarr=map(field_values) do data 
        HaloArray(convert.(TTypes,data.data),halo_width(data), data.topology,
    data.boundary_condition)
    end
    
    ntup =NamedTuple{field_names}(newarr)
    Len=length(field_values)
    return MultiHaloArray{TTypes, N_ref , typeof(ntup),Len}(ntup)
end


function MultiHaloArray(local_size::NTuple{N, Int}, halo::Int, 
                        bcs::NamedTuple{names, <:Tuple}) where {N, names}
    arrays = NamedTuple{names}(map(bcs) do bc
        HaloArray(local_size, halo, bc)
    end)
    return MultiHaloArray(arrays)
end


# Access field by symbol name
Base.getindex(mha::MultiHaloArray, name::Symbol) = mha.arrays[name]


#Base.setproperty!(mha::MultiHaloArray, name::Symbol, value) = (mha.arrays[name] = value)



# Metadata helpers
Base.eltype(mha::MultiHaloArray{T, N, A,Len}) where {T,N,A,Len} = T
Base.ndims(mha::MultiHaloArray{T, N, A,Len}) where {T,N,A,Len} = N 



# Size includes field axis
@inline Base.size(mha::MultiHaloArray) = (length(mha.arrays), size(first(mha.arrays))...)
@inline Base.size(mha::MultiHaloArray, i::Int) = size(mha)[i]

n_field(halos::MultiHaloArray{T,N,A,Len}) where {T,N,A,Len} = Len

#Base.length(halo::HaloArray) = length(halo.data)


@inline interior_size(halos::MultiHaloArray) =(n_field(halos), map(interior_size,halos.arrays)...)
@inline full_size(halos::MultiHaloArray) =(n_field(halos), map(full_size,halos.arrays)...)
@inline full_size(halos::MultiHaloArray,i) =(n_field(halos), map(full_size,halos.arrays)...)[i]
@inline halo_width(halo::MultiHaloArray,i)= map(halo_width,halo.arrays)
@inline Base.parent(halo::MultiHaloArray)  = map(parent,halo.arrays)
@inline Base.axes(x::MultiHaloArray,i)  = 
(1:n_field(x),first(map(axes,x.arrays))...)

to_tuple(mha::MultiHaloArray) = (mha.arrays...,)

function Base.similar(mha::MultiHaloArray{AA, N, A,Len}, ::Type{T},dims::NTuple{M,Inf64}) where {AA,N, A, Len,T,M}

    arrs = map(a -> similar(a, T, dims), values(mha.arrays))
    names = keys(mha.arrays)
    nt = NamedTuple{names}(arrs)
    return MultiHaloArray{T, N, typeof(nt),length(nt)}(nt)
end


function Base.similar(mha::MultiHaloArray{AA, N, A,Len}, ::Type{T}) where {AA,N, A, Len,T}
    arrs = map(a -> similar(a, T), values(mha.arrays))
    names = keys(mha.arrays)
    nt = NamedTuple{names}(arrs)
    return MultiHaloArray{T, N, typeof(nt),length(nt)}(nt)
end


function Base.similar(mha::MultiHaloArray{AA, N, A,Len}) where {AA,N, A,Len}
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


function foreach_field!(f!, mha::MultiHaloArray,etc::Vararg{MultiHaloArray})
   
    for (n,arr) in enumerate(mha.arrays)
        f!(arr, getindex(etc.arrays,n)...)
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
    result= ntuple(1:n_fields) do n
        f(mha.arrays[n], getindex(etc.arrays,n)...)
    end

    return NamedTuple{keyset}(result)
end


