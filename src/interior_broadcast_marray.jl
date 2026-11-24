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
_field_leaf(x::_SingleBlock, i) = interior_view(x)
_field_leaf(x, i) = x

# Every collection operand must have the destination's number of fields.
function _check_field_count(x::FieldCollection, n)
    nf = length(_fields(x))
    nf == n || throw(DimensionMismatch("collection broadcast: operand has $nf fields, destination has $n"))
    return nothing
end
_check_field_count(x, n) = nothing

# The broadcast for field `i`: every collection operand replaced by its field.
@inline _field_broadcast(bc_flat::Broadcasted, i) =
    _map_operands(x -> _field_leaf(x, i), bc_flat, MultiHaloArrayStyle)

# Run `op!(field, broadcast_for_that_field)` for every field of `dest`; the
# backend of the fields decides the order of the two loops.
@inline function _each_field!(op!::F, dest::FieldCollection, bc::Broadcasted) where {F}
    bc_flat = Broadcast.flatten(bc)
    _check_field_counts(dest, bc_flat)
    _run_fields!(halo_backend(dest), op!, dest, bc_flat)
    return dest
end
# Every level of a nested destination is checked before any cell is written.
function _check_field_counts(dest::FieldCollection, bc_flat::Broadcasted)
    out = _fields(dest)
    foreach(x -> _check_field_count(x, length(out)), bc_flat.args)
    for i in eachindex(out)
        _check_field_counts(out[i], _field_broadcast(bc_flat, i))
    end
    return nothing
end
_check_field_counts(dest, bc) = nothing

# Single-block fields have one tile each: field by field.
@inline function _run_fields!(::AbstractHaloBackend, op!::F, dest::FieldCollection, bc_flat::Broadcasted) where {F}
    out = _fields(dest)
    for i in eachindex(out)
        op!(out[i], _field_broadcast(bc_flat, i))
    end
    return nothing
end
# Threaded fields: ONE tile sweep with the fields inside each tile, so a state
# of n fields starts the thread backend once per broadcast instead of n times.
@inline _run_fields!(::ThreadedHaloBackend, op!::F, dest::FieldCollection, bc_flat::Broadcasted) where {F} =
    _foreach_tile(t -> _tile_fields!(op!, dest, bc_flat, t), dest)

@inline function _tile_fields!(op!::F, dest::FieldCollection, bc_flat::Broadcasted, t) where {F}
    out = _fields(dest)
    for i in eachindex(out)
        _tile_fields!(op!, out[i], _field_broadcast(bc_flat, i), t)
    end
    return nothing
end
@inline _tile_fields!(op!::F, dest::ThreadedHaloArray, bc_flat::Broadcasted, t) where {F} =
    (op!(interior_view(dest, t), unpack_ha_tile(bc_flat, t, dest)); nothing)

@inline Base.copyto!(dest::FieldCollection, bc::Broadcasted{<:MultiHaloArrayStyle}) =
    _each_field!(copyto!, dest, bc)
@inline Base.copy(bc::Broadcasted{<:MultiHaloArrayStyle}) = _each_field!(copyto!, similar(bc, Broadcast.combine_eltypes(bc.f, bc.args)), bc)
Broadcast.materialize!(dest::FieldCollection, bc::Broadcasted) =
    _each_field!(Broadcast.materialize!, dest, bc)

Base.similar(bc::Broadcasted{<:MultiHaloArrayStyle}, ::Type{T}) where {T} =
    similar(_find_operand(FieldCollection, bc), T)
Base.similar(bc::Broadcasted{<:MultiHaloArrayStyle}) = similar(_find_operand(FieldCollection, bc))
