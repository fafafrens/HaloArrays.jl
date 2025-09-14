struct MaybeHaloArray{A}
    data::A
    active::Bool
end


MaybeHaloArray(a::HaloArray) = MaybeHaloArray{typeof(a)}(a, isactive(a))
# convenience constructor per MultiHaloArray: active = true se almeno un campo è active
MaybeHaloArray(a::MultiHaloArray) = MaybeHaloArray{typeof(a)}(a, isactive(a))



# delegazioni di funzioni comunemente usate (inoltrate a data quando active)
Base.size(m::MaybeHaloArray) = size(m.data)

# ndims sul tipo MaybeHaloArray: delega al tipo interno A
Base.ndims(::Type{<:MaybeHaloArray{A}}) where {A} = ndims(A)

# ndims sull'istanza MaybeHaloArray delega al tipo
Base.ndims(m::MaybeHaloArray{A}) where {A} = ndims(A)

@inline halo_width(::Type{<:MaybeHaloArray{A}}) where {A} = halo_width(A)

@inline halo_width(m::MaybeHaloArray{A}) where A = halo_width(A)

Base.eltype(::Type{MaybeHaloArray{A}}) where {A} = eltype(A)
#Base.getindex(m::MaybeHaloArray, inds...) = m.active ? getindex(m.data, inds...) : throw(ErrorException("MaybeHaloArray: getindex on inactive"))
#Base.setindex!(m::MaybeHaloArray, v, inds...) = m.active ? setindex!(m.data, v, inds...) : throw(ErrorException("MaybeHaloArray: setindex! on inactive"))
Base.length(m::MaybeHaloArray) = m.active ? length(m.data) : 0
#Base.first(m::MaybeHaloArray, args...) = m.active ? first(m.data, args...) : throw(ErrorException("MaybeHaloArray: first on inactive"))
#Base.last(m::MaybeHaloArray, args...) = m.active ? last(m.data, args...) : throw(ErrorException("MaybeHaloArray: last on inactive"))

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









