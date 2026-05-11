mutable struct LocalHaloArray{T,N,A,Halo,BCondition}
    data::A
    boundary_condition::BCondition
end

function LocalHaloArray(data::AbstractArray{T,N}, halo::Int, boundary_condition) where {T,N}
    bc = normalize_boundary_condition(boundary_condition, N)
    return LocalHaloArray{T,N,typeof(data),halo,typeof(bc)}(data, bc)
end

function LocalHaloArray(::Type{T}, local_inner_size::NTuple{N,Int}, halo::Int; boundary_condition=:repeating) where {T,N}
    fullsize = ntuple(i -> local_inner_size[i] + 2 * halo, Val(N))
    data = zeros(T, fullsize...)
    return LocalHaloArray(data, halo, boundary_condition)
end

function LocalHaloArray(local_inner_size::NTuple{N,Int}, halo::Int; boundary_condition=:repeating) where {N}
    return LocalHaloArray(Float64, local_inner_size, halo; boundary_condition)
end

@inline Base.length(halo::LocalHaloArray) = length(interior_view(halo))
@inline Base.size(halo::LocalHaloArray) = interior_size(halo)
@inline Base.size(halo::LocalHaloArray, i::Int) = interior_size(halo)[i]
@inline Base.eltype(::LocalHaloArray{T}) where {T} = T
@inline Base.ndims(::LocalHaloArray{T,N}) where {T,N} = N
@inline Base.ndims(::Type{<:LocalHaloArray{T,N}}) where {T,N} = N
@inline Base.parent(halo::LocalHaloArray) = halo.data
@inline Base.axes(halo::LocalHaloArray) = axes(interior_view(halo))
@inline Base.axes(halo::LocalHaloArray, d::Int) = axes(interior_view(halo), d)
@inline Base.eachindex(halo::LocalHaloArray) = eachindex(interior_view(halo))
@inline Base.iterate(halo::LocalHaloArray) = iterate(interior_view(halo))
@inline Base.iterate(halo::LocalHaloArray, state) = iterate(interior_view(halo), state)
@inline Base.getindex(halo::LocalHaloArray, I...) = getindex(interior_view(halo), I...)
@inline Base.setindex!(halo::LocalHaloArray, value, I...) = setindex!(interior_view(halo), value, I...)

isactive(::LocalHaloArray) = true
get_comm(::LocalHaloArray) = nothing

@inline halo_width(::Type{<:LocalHaloArray{T,N,A,Halo}}) where {T,N,A,Halo} = Halo
@inline halo_width(::LocalHaloArray{T,N,A,Halo}) where {T,N,A,Halo} = Halo

@inline function interior_size(halo::LocalHaloArray{T,N}) where {T,N}
    h = halo_width(halo)
    return ntuple(i -> size(halo.data, i) - 2 * h, Val(N))
end

@inline full_size(halo::LocalHaloArray) = size(halo.data)
@inline full_size(halo::LocalHaloArray, i::Int) = size(halo.data, i)

@inline function interior_range(halo::LocalHaloArray)
    h = halo_width(halo)
    N = ndims(halo)
    return ntuple(i -> (h + 1):(full_size(halo, i) - h), Val(N))
end

@inline function full_range(halo::LocalHaloArray)
    N = ndims(halo)
    return ntuple(i -> 1:full_size(halo, i), Val(N))
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

function Base.similar(halo::LocalHaloArray{T,N,A,Halo,BCondition}, element_type=eltype(halo),
        dims::NTuple{M,Int}=interior_size(halo)) where {T,N,A,Halo,BCondition,M}
    M == N || throw(DimensionMismatch("LocalHaloArray similar dims must have $N dimensions"))
    fullsize = ntuple(i -> dims[i] + 2 * halo_width(halo), Val(N))
    data = zeros(element_type, fullsize...)
    return LocalHaloArray(data, halo_width(halo), halo.boundary_condition)
end

function Base.copyto!(dest::LocalHaloArray, src::LocalHaloArray)
    @assert size(dest) == size(src) "Incompatible array sizes"
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
    @assert all(size(dest_interior) == size(s) for s in src_interiors) "Incompatible array sizes"
    map!(f, dest_interior, src_interiors...)
    boundary_condition!(dest)
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

local_to_global_index(::LocalHaloArray, local_idx::NTuple{N,Int}) where {N} = local_idx
global_to_local_index(halo::LocalHaloArray, global_idx::NTuple{N,Int}) where {N} =
    all(i -> 1 <= global_idx[i] <= size(halo, i), 1:N) ? ntuple(i -> global_idx[i] + halo_width(halo), Val(N)) : nothing
global_size(halo::LocalHaloArray) = interior_size(halo)

function Base.show(io::IO, obj::LocalHaloArray)
    print(io, "LocalHaloArray of size ", size(obj), " (full size: ", full_size(obj), "), halo width: ", halo_width(obj), "\n")
    print(io, "  eltype: ", eltype(obj), "\n")
    print(io, "  boundary_condition: ", obj.boundary_condition, "\n")
end

function Base.show(io::IO, mime::MIME"text/plain", obj::LocalHaloArray)
    println(io, "LocalHaloArray (full size: ", full_size(obj), ", halo width: ", halo_width(obj), ")")
    println(io, "  eltype: ", eltype(obj))
    println(io, "  boundary_condition: ", obj.boundary_condition)
    println(io, "  interior data preview:")
    show(io, mime, interior_view(obj))
end
