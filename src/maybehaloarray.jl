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



# An inactive value holds no data, so it reports an empty shape — keeping the
# `AbstractArray` invariant `length == prod(size)` (active: global shape; a
# global-shaped `size` with `length == 0` would make `collect` return
# uninitialized garbage). Use `is_active` to gate before touching the data.
Base.size(m::MaybeHaloArray{T,N}) where {T,N} =
    is_active(m) ? global_size(m) : ntuple(_ -> 0, Val(N))
interior_axes(m::MaybeHaloArray) = interior_axes(m.data)
interior_size(m::MaybeHaloArray) = interior_size(m.data)
global_size(m::MaybeHaloArray) = global_size(m.data)
storage_size(m::MaybeHaloArray) = storage_size(m.data)
Base.parent(m::MaybeHaloArray) = m.data
# axes must agree with `size` (empty when inactive), or `collect`/`similar` —
# which allocate from `axes` — would rebuild the inner shape and fill garbage.
Base.axes(m::MaybeHaloArray{T,N}) where {T,N} =
    is_active(m) ? axes(m.data) : ntuple(_ -> Base.OneTo(0), Val(N))
# Trailing dims beyond ndims are OneTo(1) (the AbstractArray contract, e.g.
# `A[i, 1]` on a vector) — same as the AbstractSingleHaloArray `axes(u, i)`.
Base.axes(m::MaybeHaloArray, i::Int) = i <= ndims(m) ? axes(m)[i] : Base.OneTo(1)

Base.ndims(::Type{<:MaybeHaloArray{T,N,A}}) where {T,N,A} = N

Base.ndims(m::MaybeHaloArray{T,N,A}) where {T,N,A} = N

@inline halo_width(::Type{<:MaybeHaloArray{T,N,A}}) where {T,N,A} = halo_width(A)

@inline halo_width(m::MaybeHaloArray) = halo_width(m.data)

Base.eltype(::Type{<:MaybeHaloArray{T,N,A}}) where {T,N,A} = T
Base.eltype(::MaybeHaloArray{T,N,A}) where {T,N,A} = T
Base.length(m::MaybeHaloArray) = m.active ? length(m.data) : 0
Base.eachindex(m::MaybeHaloArray{T,N,A}) where {T,N,A} =
    is_active(m) ? eachindex(m.data) : CartesianIndices(ntuple(_ -> 1:0, Val(N)))

# The interior accessors pass through (guarded like getindex), so consumers of
# a reduced result never unwrap: `is_active(r) && interior_view(r)` works
# whether `r` is a bare serial array or a Maybe-wrapped distributed one. Both
# guard on activity so they agree — an inactive result has no addressable
# interior, so returning ranges into the placeholder would silently read zeros.
function interior_view(m::MaybeHaloArray, args...)
    is_active(m) || throw(ErrorException("MaybeHaloArray: attempt to view inactive value"))
    return interior_view(m.data, args...)
end
function interior_range(m::MaybeHaloArray, args...)
    is_active(m) || throw(ErrorException("MaybeHaloArray: attempt to range inactive value"))
    return interior_range(m.data, args...)
end

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
getdata(x::AbstractHaloArray) = x   # identity on bare arrays: unwrap uniformly
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
