struct MaybeHaloArray{T,N,A<:AbstractHaloArray{T,N}} <: AbstractHaloArray{T,N}
    data::A
    active::Bool
end


MaybeHaloArray(a::A) where {T,N,A<:AbstractHaloArray{T,N}} =
    MaybeHaloArray{T,N,A}(a, isactive(a))



# delegazioni di funzioni comunemente usate (inoltrate a data quando active)
Base.size(m::MaybeHaloArray) = global_size(m)
owned_size(m::MaybeHaloArray) = owned_size(m.data)
owned_axes(m::MaybeHaloArray) = owned_axes(m.data)
interior_size(m::MaybeHaloArray) = interior_size(m.data)
global_size(m::MaybeHaloArray) = global_size(m.data)
storage_size(m::MaybeHaloArray) = storage_size(m.data)
Base.axes(m::MaybeHaloArray) = axes(m.data)
Base.axes(m::MaybeHaloArray, i::Int) = axes(m.data, i)

# ndims sul tipo MaybeHaloArray: delega al tipo interno A
Base.ndims(::Type{<:MaybeHaloArray{T,N,A}}) where {T,N,A} = N

# ndims sull'istanza MaybeHaloArray delega al tipo
Base.ndims(m::MaybeHaloArray{T,N,A}) where {T,N,A} = N

@inline halo_width(::Type{<:MaybeHaloArray{T,N,A}}) where {T,N,A} = halo_width(A)

@inline halo_width(m::MaybeHaloArray) = halo_width(m.data)

Base.eltype(::Type{<:MaybeHaloArray{T,N,A}}) where {T,N,A} = T
Base.length(m::MaybeHaloArray) = m.active ? length(m.data) : 0

function Base.getindex(m::MaybeHaloArray, I::Vararg{Integer})
    isactive(m) || throw(ErrorException("MaybeHaloArray: attempt to index inactive value"))
    return getindex(m.data, I...)
end

function Base.setindex!(m::MaybeHaloArray, value, I::Vararg{Integer})
    isactive(m) || throw(ErrorException("MaybeHaloArray: attempt to index inactive value"))
    setindex!(m.data, value, I...)
    return m
end

# show
function Base.show(io::IO, m::MaybeHaloArray)
    if m.active
        print(io, "MaybeHaloArray(active) -> ")
        show(io, m.data)
    else
        print(io, "MaybeHaloArray(inactive, placeholder=", typeof(m.data), ")")
    end
end

# utility helpers
isactive(m::MaybeHaloArray) = m.active
getdata(m::MaybeHaloArray) = m.data
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
    if isactive(m)
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

# similar(m, T): crea un MaybeHaloArray il cui inner è `similar(m.data, T)`,
# preservando il flag `active` del wrapper.
function Base.similar(m::MaybeHaloArray, ::Type{T}) where {T}
    inner_sim = similar(m.data, T)
    return MaybeHaloArray(inner_sim, m.active)
end

function Base.similar(m::MaybeHaloArray, ::Type{T}, dims::Dims{N}) where {T,N}
    inner_sim = similar(m.data, T, dims)
    return MaybeHaloArray(inner_sim, m.active)
end

Base.similar(m::MaybeHaloArray, ::Type{T}, dims::NTuple{N,<:Integer}) where {T,N} =
    similar(m, T, ntuple(d -> Int(dims[d]), Val(N)))

Base.similar(m::MaybeHaloArray, dims::Dims{N}) where {N} = similar(m, eltype(m), dims)
Base.similar(m::MaybeHaloArray, dims::NTuple{N,<:Integer}) where {N} =
    similar(m, eltype(m), dims)

# similar(m): crea un MaybeHaloArray con inner = similar(m.data)
# preserva lo stato `active`
function Base.similar(m::MaybeHaloArray)
    inner_sim = similar(m.data)
    return MaybeHaloArray(inner_sim, m.active)
end

# copy(m): copia profonda del wrapper; preserva `active`
function Base.copy(m::MaybeHaloArray)
    newdata = copy(m.data)
    return MaybeHaloArray(newdata, m.active)
end
