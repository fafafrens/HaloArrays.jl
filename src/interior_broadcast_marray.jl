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
_field_leaf(x::FieldCollection, ::Val{i}) where {i} = _fields(x)[i]   # static index: a field tuple of mixed types
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

# `g(field, broadcast_for_that_field)` for every field. A field tuple is walked
# with `first`/`tail` and its position carried in a `Val`, so each field and
# its operands are picked statically: fields of different types (boundary
# conditions are type parameters) stay inferred and allocation-free. A field
# array is homogeneous and keeps the plain loop.
@inline _each_field_broadcast(g::G, out::Tuple, bc_flat::Broadcasted) where {G} =
    _each_field_broadcast(g, out, bc_flat, Val(1))
@inline _each_field_broadcast(g::G, out::Tuple, bc_flat::Broadcasted, v::Val{i}) where {G,i} =
    (g(first(out), _field_broadcast(bc_flat, v)); _each_field_broadcast(g, Base.tail(out), bc_flat, Val(i + 1)))
@inline _each_field_broadcast(::G, ::Tuple{}, ::Broadcasted, ::Val) where {G} = nothing
@inline function _each_field_broadcast(g::G, out::AbstractArray, bc_flat::Broadcasted) where {G}
    for i in eachindex(out)
        g(out[i], _field_broadcast(bc_flat, i))
    end
    return nothing
end

# Run `op!(field, broadcast_for_that_field)` for every field of `dest`; the
# backend of the fields decides the order of the two loops.
@inline function _each_field!(op!::F, dest::FieldCollection, bc::Broadcasted) where {F}
    _run_fields!(halo_backend(dest), op!, dest, Broadcast.flatten(bc))
    return dest
end

# Single-block fields have one tile each: field by field. A nested field checks
# its own operands when `op!` reaches it.
@inline function _run_fields!(::AbstractHaloBackend, op!::F, dest::FieldCollection, bc_flat::Broadcasted) where {F}
    out = _fields(dest)
    foreach(x -> _check_field_count(x, length(out)), bc_flat.args)
    _each_field_broadcast(op!, out, bc_flat)
    return nothing
end
# Threaded fields: ONE tile sweep with the fields inside each tile, so a state
# of n fields starts the thread backend once per broadcast instead of n times.
# The tiles run in tasks, so every nesting level is checked before the sweep.
@inline function _run_fields!(::ThreadedHaloBackend, op!::F, dest::FieldCollection, bc_flat::Broadcasted) where {F}
    _check_field_counts(dest, bc_flat)
    _foreach_tile(t -> _tile_fields!(op!, dest, bc_flat, t), dest)
    return nothing
end

function _check_field_counts(dest::FieldCollection, bc_flat::Broadcasted)
    out = _fields(dest)
    foreach(x -> _check_field_count(x, length(out)), bc_flat.args)
    _check_nested_counts(out, bc_flat)
    return nothing
end
_check_field_counts(dest, bc) = nothing
# Only fields that are collections have operands of their own to count.
_check_nested_counts(out, bc_flat) = _each_field_broadcast(_check_field_counts, out, bc_flat)
_check_nested_counts(::Tuple{Vararg{AbstractSingleHaloArray}}, bc_flat) = nothing
_check_nested_counts(::AbstractArray{<:AbstractSingleHaloArray}, bc_flat) = nothing

@inline _tile_fields!(op!::F, dest::FieldCollection, bc_flat::Broadcasted, t) where {F} =
    _each_field_broadcast((field, bc) -> _tile_fields!(op!, field, bc, t), _fields(dest), bc_flat)
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
