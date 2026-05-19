struct MaybeHaloArray{A<:AbstractHaloArray} <: AbstractHaloArray
    data::A
    active::Bool
end


MaybeHaloArray(a) = MaybeHaloArray{typeof(a)}(a, isactive(a))



# delegazioni di funzioni comunemente usate (inoltrate a data quando active)
Base.size(m::MaybeHaloArray) = size(m.data)

# ndims sul tipo MaybeHaloArray: delega al tipo interno A
Base.ndims(::Type{<:MaybeHaloArray{A}}) where {A} = ndims(A)

# ndims sull'istanza MaybeHaloArray delega al tipo
Base.ndims(m::MaybeHaloArray{A}) where {A} = ndims(A)

@inline halo_width(::Type{<:MaybeHaloArray{A}}) where {A} = halo_width(A)

@inline halo_width(m::MaybeHaloArray) = halo_width(m.data)

Base.eltype(::Type{MaybeHaloArray{A}}) where {A} = eltype(A)
Base.length(m::MaybeHaloArray) = m.active ? length(m.data) : 0

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
    MaybeHaloArray{typeof(m.data)}(m.data, flag)
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






