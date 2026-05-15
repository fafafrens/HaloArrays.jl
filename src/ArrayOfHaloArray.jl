# MultiHaloArray using NamedTuple to store fields
mutable struct ArrayOfHaloArray{T, N, Shape ,A } #<: AbstractArray{T, N}
    arrays::A  # Arra of HaloArrays
end

function ArrayOfHaloArray(array::AbstractArray{T, N}) where {T,N}
    shape=size(array)
    TTtype=promote_type(array...)
    N_ref = ndims(first(field_values))+N
    
    ArrayOfHaloArray{TTtype, N_ref, shape ,typeof(array) }(arrays)
end 

function ArrayOfHaloArray(local_size::NTuple{N, Int}, halo::Int, 
                        bcs::AbstractArray{T, N}) where {T,N}
    
    arrays= map(bcs) do  bc
        HaloArray(local_size, halo, bc)
    end 
                       
    return ArrayOfHaloArray(arrays)
end

# Metadata helpers
Base.eltype(mha::ArrayOfHaloArray{T, N ,S, A}) where {T,N,S,A} = T
Base.ndims(mha::ArrayOfHaloArray{T, N ,S, A}) where {T,N,S,A} =N +prod(S)
Base.ndims(::Type{ArrayOfHaloArray{T, N ,S, A}}) where {T,N,S,A} =N +prod(S)

# Size includes field axis
@inline Base.size(mha::ArrayOfHaloArray) = (size(mha.arrays)..., size(first(mha.arrays))...)

@inline Base.size(mha::ArrayOfHaloArray, i::Int) = size(mha)[i]

@inline field_shape(halos::ArrayOfHaloArray{T, N ,S, A}) where {T, N ,S, A} = S
n_field(halos::ArrayOfHaloArray)  = prod(field_shape(S))

#Base.length(halo::HaloArray) = length(halo.data)


@inline interior_size(halos::ArrayOfHaloArray) =(field_shape(halos)..., map(interior_size,halos.arrays)...)
@inline full_size(halos::ArrayOfHaloArray) =(field_shape(halos)..., map(full_size,halos.arrays)...)
@inline full_size(halos::ArrayOfHaloArray,i) =(field_shape(halos)..., map(full_size,halos.arrays)...)[i]
@inline halo_width(halo::ArrayOfHaloArray,i)= map(halo_width,halo.arrays)
@inline Base.parent(halo::ArrayOfHaloArray)  = halo.arrays
function Base.axes(x::ArrayOfHaloArray)  
    shape=field_shape(x)
return (ntuple(j->1:shape[j],lenght(shape))...,first(map(axes,parent(x)))...)
end 

function Base.similar(mha::ArrayOfHaloArray{AA, N,S, A}, ::Type{T},dims::NTuple{M,Int64}) where {AA,N, S,A,T,M}

    arrs = map(a -> similar(a, T, dims), parent(mha))

    return ArrayOfHaloArray{T, N,S, typeof(arrs )}(arrs)
end

function Base.similar(mha::ArrayOfHaloArray{AA, N,S, A}, ::Type{T}) where {AA,N,S, A, T}
    
    arrs = map(a -> similar(a, T), parent(mha))
    
    return ArrayOfHaloArray{T, N, S,typeof(arrs )}(arrs)
end


function Base.similar(mha::ArrayOfHaloArray{AA, N,S, A}) where {AA,N,S, A}
    arrs = map(a -> similar(a), parent(mha))
    
    return ArrayOfHaloArray{AA, N, S,typeof(arrs )}(arrs)
end


function Base.copy(mha::ArrayOfHaloArray)
    arrs = map(a -> copy(a), parent(mha))
    return ArrayOfHaloArray(arrs)
end

# I ma not sure about this semantic
function Base.map(f, mha::ArrayOfHaloArray)
    arrs  = map(x -> map(f, x),parent(mha))
    # newfields is a Vector of mapped HaloArrays, but you want a NamedTuple
    
    return ArrayOfHaloArray(arrs)
end

function foreach_field!(f!, mha::ArrayOfHaloArray)

    foreach(f!,parent(mha)) 

end


function map_over_field(f, mha::ArrayOfHaloArray)

    return map(f, parent(mha))

end


"""
    interior_view(mha::MultiHaloArray)

Restituisce un NamedTuple con le interior view di ciascun campo del MultiHaloArray.
I campi del NamedTuple hanno gli stessi simboli del `mha.arrays`.
"""
function interior_view(mha::ArrayOfHaloArray)

    return map(interior_view,mha)

end

# isactive per MultiHaloArray: true se almeno un campo è active
# richiede che isactive sia definita per i singoli HaloArray
function isactive(mha::ArrayOfHaloArray)
    return all(isactive, parent(mha))
end

