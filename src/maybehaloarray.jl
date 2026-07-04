"""
    MaybeHaloArray(a::AbstractHaloArray)

A halo array that may be *inactive* — a wrapper carrying an `active` flag.

This handles the MPI case where a rank ends up owning no data (an empty patch):
the array is still a valid object to pass around and call collectively, but it
reports `length 0`, iterates over nothing, and errors on scalar indexing.
Reductions and broadcasts skip inactive values. The activity is taken from
`is_active(a)` at construction.
"""
struct MaybeHaloArray{T,N,A<:AbstractHaloArray{T,N}} <: AbstractHaloArray{T,N}
    data::A
    active::Bool
end


MaybeHaloArray(a::A) where {T,N,A<:AbstractHaloArray{T,N}} =
    MaybeHaloArray{T,N,A}(a, is_active(a))



Base.size(m::MaybeHaloArray) = global_size(m)
interior_axes(m::MaybeHaloArray) = interior_axes(m.data)
interior_size(m::MaybeHaloArray) = interior_size(m.data)
global_size(m::MaybeHaloArray) = global_size(m.data)
storage_size(m::MaybeHaloArray) = storage_size(m.data)
Base.parent(m::MaybeHaloArray) = m.data
Base.axes(m::MaybeHaloArray) = axes(m.data)
Base.axes(m::MaybeHaloArray, i::Int) = axes(m.data, i)

Base.ndims(::Type{<:MaybeHaloArray{T,N,A}}) where {T,N,A} = N

Base.ndims(m::MaybeHaloArray{T,N,A}) where {T,N,A} = N

@inline halo_width(::Type{<:MaybeHaloArray{T,N,A}}) where {T,N,A} = halo_width(A)

@inline halo_width(m::MaybeHaloArray) = halo_width(m.data)

Base.eltype(::Type{<:MaybeHaloArray{T,N,A}}) where {T,N,A} = T
Base.eltype(::MaybeHaloArray{T,N,A}) where {T,N,A} = T
Base.length(m::MaybeHaloArray) = m.active ? length(m.data) : 0
Base.eachindex(m::MaybeHaloArray{T,N,A}) where {T,N,A} =
    is_active(m) ? eachindex(m.data) : CartesianIndices(ntuple(_ -> 1:0, Val(N)))

function Base.getindex(m::MaybeHaloArray, I::Vararg{Integer})
    is_active(m) || throw(ErrorException("MaybeHaloArray: attempt to index inactive value"))
    return getindex(m.data, I...)
end

function Base.setindex!(m::MaybeHaloArray, value, I::Vararg{Integer})
    is_active(m) || throw(ErrorException("MaybeHaloArray: attempt to index inactive value"))
    setindex!(m.data, value, I...)
    return m
end

function Base.show(io::IO, m::MaybeHaloArray)
    if m.active
        print(io, "MaybeHaloArray(active) -> ")
        show(io, m.data)
    else
        print(io, "MaybeHaloArray(inactive, placeholder=", typeof(m.data), ")")
    end
end

is_active(m::MaybeHaloArray) = m.active
is_root(m::MaybeHaloArray; root::Integer=0) =
    is_active(m) && is_root(getdata(m); root=root)
halo_backend(m::MaybeHaloArray) = halo_backend(getdata(m))
getdata(m::MaybeHaloArray) = m.data
active(x::AbstractHaloArray)   = MaybeHaloArray(x, true)
inactive(x::AbstractHaloArray) = MaybeHaloArray(x, false)

function unwrap(m::MaybeHaloArray)
    if m.active
        return m.data
    else
        throw(ErrorException("MaybeHaloArray: attempt to unwrap inactive value"))
    end
end

function apply_if_active(f::Function, m::MaybeHaloArray, args...; kwargs...)
    m.active ? f(m.data, args...; kwargs...) : nothing
end

function apply_if_active!(f::Function, m::MaybeHaloArray, args...; kwargs...)
    if is_active(m)
        f(m.data, args...; kwargs...)
    end
    return m
end

function setactive(m::MaybeHaloArray, flag::Bool)
    MaybeHaloArray{eltype(m),ndims(m),typeof(m.data)}(m.data, flag)
end

macro maybe_delegate(funs...)
    bodies = Expr[]
    for fn in funs
        push!(bodies, :(function $(esc(fn))(m::MaybeHaloArray, args...; kwargs...)
            if m.active
                return $(esc(fn))(m.data, args...; kwargs...)
            else
                return nothing
            end
        end))
    end
    return Expr(:block, bodies...)
end

function Base.similar(m::MaybeHaloArray, ::Type{T}) where {T}
    inner_sim = similar(m.data, T)
    return MaybeHaloArray(inner_sim, m.active)
end

function Base.similar(m::MaybeHaloArray, ::Type{T}, dims::Dims{N}) where {T,N}
    inner_sim = similar(m.data, T, dims)
    return MaybeHaloArray(inner_sim, m.active)
end

# Non-Int dims are normalized to Dims by Base's generic similar fallbacks.

Base.similar(m::MaybeHaloArray, dims::Dims{N}) where {N} = similar(m, eltype(m), dims)
Base.similar(m::MaybeHaloArray, dims::NTuple{N,<:Integer}) where {N} =
    similar(m, eltype(m), dims)

function Base.similar(m::MaybeHaloArray)
    inner_sim = similar(m.data)
    return MaybeHaloArray(inner_sim, m.active)
end

function Base.copy(m::MaybeHaloArray)
    newdata = copy(m.data)
    return MaybeHaloArray(newdata, m.active)
end

function Base.copyto!(dest::MaybeHaloArray, src::MaybeHaloArray)
    is_active(dest) == is_active(src) ||
        throw(ArgumentError("MaybeHaloArray copyto! requires matching active states"))
    if is_active(dest)
        copyto!(dest.data, src.data)
    end
    return dest
end

function Base.fill!(m::MaybeHaloArray, value)
    if is_active(m)
        fill!(m.data, value)
    end
    return m
end

function Base.zero(m::MaybeHaloArray)
    return MaybeHaloArray(zero(m.data), m.active)
end
