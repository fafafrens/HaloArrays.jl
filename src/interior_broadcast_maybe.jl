using Base.Broadcast: Broadcasted, broadcastable, BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle

# Broadcasting over a MaybeHaloArray: the style and its shared precedence rules
# live in interior_broadcast.jl. Inactive destinations are untouched; active
# ones broadcast on the wrapped array, with Maybe operands unwrapped.

Broadcast.BroadcastStyle(::Type{T}) where {T<:MaybeHaloArray} = MaybeHaloArrayStyle{ndims(T)}()

_mixed_maybe_broadcast_error() =
    throw(ArgumentError("broadcast between MaybeHaloArray and non-Maybe halo containers is not supported; wrap all halo operands in MaybeHaloArray or unwrap first"))
for S in (:HaloArrayStyle, :ThreadedHaloArrayStyle, :MultiHaloArrayStyle)
    @eval begin
        Broadcast.BroadcastStyle(::MaybeHaloArrayStyle, ::$S) = _mixed_maybe_broadcast_error()
        Broadcast.BroadcastStyle(::$S, ::MaybeHaloArrayStyle) = _mixed_maybe_broadcast_error()
    end
end

Broadcast.broadcastable(x::MaybeHaloArray) = x

_maybe_leaf(x::MaybeHaloArray) = x.data
_maybe_leaf(x) = x
# Every node loses its style: the unwrapped tree is a plain halo-array broadcast.
unpack_maybe(bc::Broadcasted) = _map_operands(_maybe_leaf, bc, Any)

@inline function Base.copyto!(dest::MaybeHaloArray, bc::Broadcasted{<:MaybeHaloArrayStyle})
    is_active(dest) || return dest
    copyto!(dest.data, unpack_maybe(Broadcast.flatten(bc)))
    return dest
end

@inline function Base.copy(bc::Broadcasted{<:MaybeHaloArrayStyle})
    dest = similar(bc)
    is_active(dest) || return dest
    copyto!(dest.data, unpack_maybe(Broadcast.flatten(bc)))
    return dest
end

function Broadcast.materialize!(dest::MaybeHaloArray, bc::Broadcasted)
    is_active(dest) || return dest
    Broadcast.materialize!(dest.data, unpack_maybe(Broadcast.flatten(bc)))
    return dest
end

Base.similar(bc::Broadcasted{<:MaybeHaloArrayStyle}, ::Type{T}) where {T} =
    similar(_find_operand(MaybeHaloArray, bc), T)
Base.similar(bc::Broadcasted{<:MaybeHaloArrayStyle}) = similar(_find_operand(MaybeHaloArray, bc))
