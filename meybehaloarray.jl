struct MaybeHaloArray{A}
    data::A
    active::Bool
end


MaybeHaloArray(a::HaloArray) = MaybeHaloArray{typeof(a)}(a, isactive(a))

# Forwarding wrapper for MaybeHaloArray: inoltra le chiamate/field a .data quando active,
# altrimenti fallisce in modo esplicito.
function Base.getproperty(m::MaybeHaloArray, name::Symbol)
    if name === :data || name === :active
        return getfield(m, name)
    end
    if m.active
        return getproperty(m.data, name)
    else
        throw(ErrorException("MaybeHaloArray: attempted access of property $(name) on inactive MaybeHaloArray"))
    end
end

function Base.setproperty!(m::MaybeHaloArray, name::Symbol, val)
    if name === :data || name === :active
        return setfield!(m, name, val)
    end
    if m.active
        setproperty!(m.data, name, val)
        return m
    else
        throw(ErrorException("MaybeHaloArray: attempted set of property $(name) on inactive MaybeHaloArray"))
    end
end

# delegazioni di funzioni comunemente usate (inoltrate a data quando active)
Base.size(m::MaybeHaloArray) = m.active ? size(m.data) : ()
Base.ndims(m::MaybeHaloArray) = m.active ? ndims(m.data) : 0
Base.eltype(::Type{MaybeHaloArray{A}}) where {A} = eltype(A)
Base.getindex(m::MaybeHaloArray, inds...) = m.active ? getindex(m.data, inds...) : throw(ErrorException("MaybeHaloArray: getindex on inactive"))
Base.setindex!(m::MaybeHaloArray, v, inds...) = m.active ? setindex!(m.data, v, inds...) : throw(ErrorException("MaybeHaloArray: setindex! on inactive"))
Base.length(m::MaybeHaloArray) = m.active ? length(m.data) : 0
Base.first(m::MaybeHaloArray, args...) = m.active ? first(m.data, args...) : throw(ErrorException("MaybeHaloArray: first on inactive"))
Base.last(m::MaybeHaloArray, args...) = m.active ? last(m.data, args...) : throw(ErrorException("MaybeHaloArray: last on inactive"))

# Some domain-specific helpers forwarded if present on inner type
function topology(m::MaybeHaloArray)
    m.active ? getproperty(m.data, :topology) : throw(ErrorException("MaybeHaloArray: topology on inactive"))
end
function receive_bufs(m::MaybeHaloArray)
    m.active ? getproperty(m.data, :receive_bufs) : throw(ErrorException("MaybeHaloArray: receive_bufs on inactive"))
end
function send_bufs(m::MaybeHaloArray)
    m.active ? getproperty(m.data, :send_bufs) : throw(ErrorException("MaybeHaloArray: send_bufs on inactive"))
end
function boundary_condition(m::MaybeHaloArray)
    m.active ? getproperty(m.data, :boundary_condition) : throw(ErrorException("MaybeHaloArray: boundary_condition on inactive"))
end
function comm_state(m::MaybeHaloArray)
    m.active ? getproperty(m.data, :comm_state) : throw(ErrorException("MaybeHaloArray: comm_state on inactive"))
end
function halo_width(m::MaybeHaloArray)
    if m.active
        if hasproperty(m.data, :halo) return getproperty(m.data, :halo) end
        # fallback: try infer from receive_bufs shapes (best effort)
        bufs = getproperty(m.data, :receive_bufs)
        return length(size(bufs[1][1])) >= 1 ? size(bufs[1][1],1) : 0
    else
        throw(ErrorException("MaybeHaloArray: halo_width on inactive"))
    end
end

# Forward domain functions if defined for inner type (best-effort)
function get_recv_view(m::MaybeHaloArray, side::Side, dim::Dim, args...; kwargs...)
    m.active ? get_recv_view(m.data, side, dim, args...; kwargs...) : throw(ErrorException("MaybeHaloArray: get_recv_view on inactive"))
end
function get_send_view(m::MaybeHaloArray, side::Side, dim::Dim, args...; kwargs...)
    m.active ? get_send_view(m.data, side, dim, args...; kwargs...) : throw(ErrorException("MaybeHaloArray: get_send_view on inactive"))
end
function fill_interior!(m::MaybeHaloArray, args...; kwargs...)
    m.active ? fill_interior!(m.data, args...; kwargs...) : nothing
end
function halo_exchange!(m::MaybeHaloArray, args...; kwargs...)
    m.active ? halo_exchange!(m.data, args...; kwargs...) : nothing
end
function cart_coords(m::MaybeHaloArray)
    m.active ? getproperty(m.data, :topology) |> (t-> getproperty(t, :cart_coords)) : throw(ErrorException("MaybeHaloArray: cart_coords on inactive"))
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

function apply_if_active(m::MaybeHaloArray, f::Function, args...; kwargs...)
    m.active ? f(m.data, args...; kwargs...) : nothing
end

function apply_if_active!(m::MaybeHaloArray, f::Function, args...; kwargs...)
    if m.active
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







