using Base.Broadcast: Broadcasted, broadcastable, BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle

# Broadcasting over a FieldCollection (MultiHaloArray / ArrayOfHaloArray): the
# style and its shared precedence rules live in interior_broadcast.jl; each
# field is broadcast on its own, with collection operands replaced by their
# i-th field and single-block arrays by their interior views.

Broadcast.BroadcastStyle(::Type{<:FieldCollection{T,D}}) where {T,D} = MultiHaloArrayStyle{D}()
# A single halo array mixed with a collection: the single-array style wins
# (both argument orders, or Base reports conflicting rules) and the broadcast
# runs field by field with the array applied to every field.
Broadcast.BroadcastStyle(::HaloArrayStyle{M}, ::MultiHaloArrayStyle{Ndim}) where {Ndim,M} =
    HaloArrayStyle(Val(max(M, Ndim)))
Broadcast.BroadcastStyle(::ThreadedHaloArrayStyle{M}, ::MultiHaloArrayStyle{Ndim}) where {Ndim,M} =
    ThreadedHaloArrayStyle(Val(max(M, Ndim)))

Broadcast.broadcastable(x::FieldCollection) = x

_field_leaf(x::FieldCollection, i) = _fields(x)[i]
_field_leaf(x::Union{HaloArray,LocalHaloArray}, i) = interior_view(x)
_field_leaf(x, i) = x

# Run `op!(field, broadcast_for_that_field)` for every field of `dest`.
@inline function _each_field!(op!::F, dest::FieldCollection, bc::Broadcasted) where {F}
    bc_flat = Broadcast.flatten(bc)
    out = _fields(dest)
    for x in bc_flat.args
        x isa FieldCollection && length(_fields(x)) != length(out) && throw(DimensionMismatch(
            "collection broadcast: operand has $(length(_fields(x))) fields, destination has $(length(out))"))
    end
    for i in eachindex(out)
        op!(out[i], _map_operands(x -> _field_leaf(x, i), bc_flat, MultiHaloArrayStyle))
    end
    return dest
end

@inline Base.copyto!(dest::FieldCollection, bc::Broadcasted{<:MultiHaloArrayStyle}) =
    _each_field!(copyto!, dest, bc)
@inline Base.copy(bc::Broadcasted{<:MultiHaloArrayStyle}) = _each_field!(copyto!, _broadcast_dest(bc), bc)
Broadcast.materialize!(dest::FieldCollection, bc::Broadcasted) =
    _each_field!(Broadcast.materialize!, dest, bc)

Base.similar(bc::Broadcasted{<:MultiHaloArrayStyle}, ::Type{T}) where {T} =
    similar(_find_operand(FieldCollection, bc), T)
Base.similar(bc::Broadcasted{<:MultiHaloArrayStyle}) = similar(_find_operand(FieldCollection, bc))
