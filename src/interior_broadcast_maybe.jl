using Base.Broadcast: Broadcasted, broadcastable, BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle

struct MaybeHaloArrayStyle{N} <: AbstractArrayStyle{N} end
MaybeHaloArrayStyle(::Val{N}) where {N} = MaybeHaloArrayStyle{N}()

Broadcast.BroadcastStyle(a::MaybeHaloArrayStyle, ::Base.Broadcast.DefaultArrayStyle{0}) = a

function Broadcast.BroadcastStyle(::Type{T}) where {T<:MaybeHaloArray}
    Ndim = ndims(T)
    MaybeHaloArrayStyle{Ndim}()
end

function Broadcast.BroadcastStyle(::MaybeHaloArrayStyle{Ndim},
        a::Base.Broadcast.DefaultArrayStyle{M}) where {Ndim,M}
    Base.Broadcast.DefaultArrayStyle(Val(max(M, Ndim)))
end
function Broadcast.BroadcastStyle(::MaybeHaloArrayStyle{Ndim},
        a::Base.Broadcast.AbstractArrayStyle{M}) where {Ndim,M}
    typeof(a)(Val(max(M,Ndim)))
end
Broadcast.BroadcastStyle(::MaybeHaloArrayStyle{Ndim}, ::MaybeHaloArrayStyle{Mdim}) where {Ndim,Mdim} =
    MaybeHaloArrayStyle(Val(max(Ndim, Mdim)))

_mixed_maybe_broadcast_error() =
    throw(ArgumentError("broadcast between MaybeHaloArray and non-Maybe halo containers is not supported; wrap all halo operands in MaybeHaloArray or unwrap first"))

Broadcast.BroadcastStyle(::MaybeHaloArrayStyle, ::HaloArrayStyle) =
    _mixed_maybe_broadcast_error()
Broadcast.BroadcastStyle(::HaloArrayStyle, ::MaybeHaloArrayStyle) =
    _mixed_maybe_broadcast_error()
Broadcast.BroadcastStyle(::MaybeHaloArrayStyle, ::ThreadedHaloArrayStyle) =
    _mixed_maybe_broadcast_error()
Broadcast.BroadcastStyle(::ThreadedHaloArrayStyle, ::MaybeHaloArrayStyle) =
    _mixed_maybe_broadcast_error()
Broadcast.BroadcastStyle(::MaybeHaloArrayStyle, ::MultiHaloArrayStyle) =
    _mixed_maybe_broadcast_error()
Broadcast.BroadcastStyle(::MultiHaloArrayStyle, ::MaybeHaloArrayStyle) =
    _mixed_maybe_broadcast_error()

Broadcast.broadcastable(x::MaybeHaloArray) = x

find_maybe(bc::Broadcasted) = find_maybe(bc.args)
find_maybe(args::Tuple) = find_maybe(find_maybe(args[1]), Base.tail(args))
find_maybe(x) = x
find_maybe(x, rest) = find_maybe(rest)
find_maybe(m::MaybeHaloArray, rest) =m 

unpack_maybe(bc::Broadcast.Broadcasted{Style}) where {Style} =
    Broadcast.Broadcasted{Style}(bc.f, unpack_args_maybe(bc.args))
unpack_maybe(bc::Broadcast.Broadcasted) =
    Broadcast.Broadcasted(bc.f, unpack_args_maybe(bc.args))

unpack_maybe(x::MaybeHaloArray) = x.data
unpack_maybe(x) = x

function unpack_args_maybe(args::Tuple)
    (unpack_maybe(args[1]), unpack_args_maybe(Base.tail(args))...)
end
unpack_args_maybe(args::Tuple{Any}) = (unpack_maybe(args[1]),)
unpack_args_maybe(::Tuple{}) = ()

@inline function Base.copyto!(dest::MaybeHaloArray, bc::Broadcasted{<:MaybeHaloArrayStyle{Ndim}}) where {Ndim}
    if !isactive(dest)
        return dest
    end

    bc_flat = Broadcast.flatten(bc)
    copyto!(dest.data, unpack_maybe(bc_flat))

    return dest
end


@inline function Base.copy(bc::Broadcast.Broadcasted{<:MaybeHaloArrayStyle{Ndim}}) where {Ndim}
    dest = similar(bc)
    if !isactive(dest)
        return dest
    end
    bc_flat = Broadcast.flatten(bc)
    copyto!(dest.data, unpack_maybe(bc_flat))
    return dest
end

function Broadcast.materialize!(dest::MaybeHaloArray, bc::Broadcasted)
    if !isactive(dest)
        return dest
    end

    bc_flat = Broadcast.flatten(bc)
    Broadcast.materialize!(dest.data, unpack_maybe(bc_flat))
    return dest
end

function Base.similar(bc::Broadcasted{<:MaybeHaloArrayStyle}, ::Type{T}) where {T}
    ha = find_maybe(bc)
    return similar(ha, T)
end

function Base.similar(bc::Broadcasted{<:MaybeHaloArrayStyle})
    ha = find_maybe(bc)
    return similar(ha)
end

