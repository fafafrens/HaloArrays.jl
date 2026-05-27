mutable struct LocalHaloArray{T,N,A,Halo,BCondition} <: AbstractSerialHaloArray{T,N}
    data::A
    boundary_condition::BCondition
end

@inline halo_backend(::Type{<:LocalHaloArray}) = LocalHaloBackend()

function LocalHaloArray(data::AbstractArray{T,N}, halo::Int, boundary_condition) where {T,N}
    bc = normalize_boundary_condition(boundary_condition, N)
    return LocalHaloArray{T,N,typeof(data),halo,typeof(bc)}(data, bc)
end

function LocalHaloArray(::Type{T}, owned_dims::NTuple{N,Int}, halo::Int; boundary_condition=:repeating) where {T,N}
    fullsize = ntuple(i -> owned_dims[i] + 2 * halo, Val(N))
    data = zeros(T, fullsize...)
    return LocalHaloArray(data, halo, boundary_condition)
end

function LocalHaloArray(owned_dims::NTuple{N,Int}, halo::Int; boundary_condition=:repeating) where {N}
    return LocalHaloArray(Float64, owned_dims, halo; boundary_condition)
end

@inline Base.length(halo::LocalHaloArray) = length(interior_view(halo))
@inline Base.size(halo::LocalHaloArray) = global_size(halo)
@inline Base.size(halo::LocalHaloArray, i::Int) = size(halo)[i]
@inline Base.eltype(::LocalHaloArray{T}) where {T} = T
@inline Base.eltype(::Type{<:LocalHaloArray{T}}) where {T} = T
@inline Base.ndims(::LocalHaloArray{T,N}) where {T,N} = N
@inline Base.ndims(::Type{<:LocalHaloArray{T,N}}) where {T,N} = N
@inline Base.parent(halo::LocalHaloArray) = halo.data
@inline Base.axes(halo::LocalHaloArray) = map(Base.OneTo, size(halo))
@inline Base.axes(halo::LocalHaloArray, d::Int) = Base.OneTo(size(halo, d))
@inline owned_axes(halo::LocalHaloArray) = axes(interior_view(halo))
@inline owned_axes(halo::LocalHaloArray, d::Int) = axes(interior_view(halo), d)
@inline Base.eachindex(halo::LocalHaloArray) = eachindex(interior_view(halo))
@inline Base.iterate(halo::LocalHaloArray) = iterate(interior_view(halo))
@inline Base.iterate(halo::LocalHaloArray, state) = iterate(interior_view(halo), state)

isactive(::LocalHaloArray) = true
is_root(::LocalHaloArray; root::Integer=0) = (root == 0)
get_comm(::LocalHaloArray) = nothing

@inline halo_width(::Type{<:LocalHaloArray{T,N,A,Halo}}) where {T,N,A,Halo} = Halo
@inline halo_width(::LocalHaloArray{T,N,A,Halo}) where {T,N,A,Halo} = Halo

@inline function interior_size(halo::LocalHaloArray{T,N}) where {T,N}
    h = halo_width(halo)
    return ntuple(i -> size(halo.data, i) - 2 * h, Val(N))
end

@inline storage_size(halo::LocalHaloArray) = size(halo.data)
@inline storage_size(halo::LocalHaloArray, i::Int) = size(halo.data, i)

@inline function interior_range(halo::LocalHaloArray)
    h = halo_width(halo)
    N = ndims(halo)
    return ntuple(i -> (h + 1):(storage_size(halo, i) - h), Val(N))
end

@inline function full_range(halo::LocalHaloArray)
    N = ndims(halo)
    return ntuple(i -> 1:storage_size(halo, i), Val(N))
end

@inline function interior_view(halo::LocalHaloArray)
    ranges = interior_range(halo)
    @views return halo.data[ranges...]
end

function full_view(halo::LocalHaloArray)
    ranges = full_range(halo)
    @views return halo.data[ranges...]
end

@inline get_send_view(s, d, array::LocalHaloArray) = get_send_view(s, d, parent(array), halo_width(array))
@inline get_recv_view(s, d, array::LocalHaloArray) = get_recv_view(s, d, parent(array), halo_width(array))
@inline versors(::LocalHaloArray{T,N}) where {T,N} = versors(Val(N))

function Base.similar(halo::LocalHaloArray{T,N,A,Halo,BCondition}, ::Type{AA},
        dims::Dims{M}) where {T,N,A,Halo,BCondition,AA,M}
    M == N || throw(DimensionMismatch("LocalHaloArray similar dims must have $N dimensions"))
    fullsize = ntuple(i -> Int(dims[i]) + 2 * halo_width(halo), Val(N))
    data = similar(parent(halo), AA, fullsize)
    return LocalHaloArray(data, halo_width(halo), halo.boundary_condition)
end

Base.similar(halo::LocalHaloArray{T,N,A,Halo,BCondition}, ::Type{AA},
    dims::NTuple{M,<:Integer}) where {T,N,A,Halo,BCondition,AA,M} =
    similar(halo, AA, ntuple(d -> Int(dims[d]), Val(M)))

Base.similar(halo::LocalHaloArray) = similar(halo, eltype(halo), size(halo))
Base.similar(halo::LocalHaloArray, ::Type{AA}) where {AA} = similar(halo, AA, size(halo))
Base.similar(halo::LocalHaloArray, dims::Dims{M}) where {M} = similar(halo, eltype(halo), dims)
Base.similar(halo::LocalHaloArray, dims::NTuple{M,<:Integer}) where {M} =
    similar(halo, eltype(halo), dims)

function Base.copyto!(dest::LocalHaloArray, src::LocalHaloArray)
    copyto!(parent(dest), parent(src))
    return dest
end

function Base.copy(src::LocalHaloArray)
    dest = similar(src)
    copyto!(dest, src)
    return dest
end

function Base.zero(halo::LocalHaloArray)
    z = similar(halo)
    fill!(z, zero(eltype(halo)))
    return z
end

function Base.fill!(halo::LocalHaloArray, value)
    fill!(parent(halo), value)
    return halo
end

function fill_interior(halo::LocalHaloArray, value)
    fill!(interior_view(halo), value)
    return halo
end

function Base.map!(f, dest::LocalHaloArray, src::Vararg{LocalHaloArray,N}) where {N}
    dest_interior = interior_view(dest)
    src_interiors = map(interior_view, src)
    map!(f, dest_interior, src_interiors...)
    return dest
end

function Base.map(f, src::Vararg{LocalHaloArray,N}) where {N}
    dest = similar(src[1])
    map!(f, dest, src...)
    return dest
end

Base.:/(halo::LocalHaloArray, x::Number) = halo ./ x
Base.:*(halo::LocalHaloArray, x::Number) = halo .* x
Base.:*(x::Number, halo::LocalHaloArray) = x .* halo

function LinearAlgebra.norm(halo::LocalHaloArray, p::Real=2)
    if p == 2
        return sqrt(mapreduce(abs2, +, interior_view(halo)))
    elseif p == Inf
        return mapreduce(abs, max, interior_view(halo))
    else
        return mapreduce(x -> abs(x)^p, +, interior_view(halo))^(1 / p)
    end
end

function Base.foreach(f, halo::LocalHaloArray)
    foreach(f, interior_view(halo))
    return nothing
end

function fill_from_local_indices!(f, halo::LocalHaloArray)
    interior = interior_view(halo)
    for I in CartesianIndices(interior)
        interior[I] = f(Tuple(I)...)
    end
    return nothing
end

function fill_from_global_indices!(f, halo::LocalHaloArray)
    interior = interior_view(halo)
    for I in CartesianIndices(interior)
        interior[I] = f(Tuple(I))
    end
    return halo
end

owned_to_global_index(::LocalHaloArray, owned_idx::NTuple{N,<:Integer}) where {N} = owned_idx
global_to_storage_index(halo::LocalHaloArray, global_idx::NTuple{N,<:Integer}) where {N} =
    all(i -> 1 <= global_idx[i] <= owned_size(halo, i), 1:N) ? ntuple(i -> global_idx[i] + halo_width(halo), Val(N)) : nothing
global_size(halo::LocalHaloArray) = owned_size(halo)

function Base.getindex(halo::LocalHaloArray, I::Vararg{Integer})
    idx = _check_global_scalar_indices(halo, I)
    storage_idx = global_to_storage_index(halo, idx)
    @inbounds return parent(halo)[storage_idx...]
end

function Base.setindex!(halo::LocalHaloArray, value, I::Vararg{Integer})
    idx = _check_global_scalar_indices(halo, I)
    storage_idx = global_to_storage_index(halo, idx)
    @inbounds parent(halo)[storage_idx...] = value
    return halo
end

function Base.show(io::IO, obj::LocalHaloArray)
    print(io, "LocalHaloArray of global size ", size(obj), " (owned size: ", owned_size(obj), ", storage size: ", storage_size(obj), "), halo width: ", halo_width(obj), "\n")
    print(io, "  eltype: ", eltype(obj), "\n")
    print(io, "  boundary_condition: ", obj.boundary_condition, "\n")
end

function Base.show(io::IO, mime::MIME"text/plain", obj::LocalHaloArray)
    println(io, "LocalHaloArray (storage size: ", storage_size(obj), ", halo width: ", halo_width(obj), ")")
    println(io, "  eltype: ", eltype(obj))
    println(io, "  boundary_condition: ", obj.boundary_condition)
    println(io, "  interior data preview:")
    show(io, mime, interior_view(obj))
end
