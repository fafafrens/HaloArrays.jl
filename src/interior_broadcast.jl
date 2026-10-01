using Base.Broadcast: Broadcasted, broadcastable, BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle

# ------------------------------------------------------------------------------
# The halo broadcast styles
#
# One style per container kind. The precedence rules are shared by all of
# them: a scalar keeps the halo style; a plain array turns the broadcast into
# plain-array semantics (DefaultArrayStyle); any other AbstractArrayStyle
# wins; two of the same kind keep the larger dimensionality. Cross-kind pairs
# are ruled on individually (below, and in the collection / Maybe files).
# ------------------------------------------------------------------------------

struct HaloArrayStyle{N}         <: AbstractArrayStyle{N} end   # HaloArray, LocalHaloArray
struct ThreadedHaloArrayStyle{N} <: AbstractArrayStyle{N} end   # ThreadedHaloArray
struct MultiHaloArrayStyle{N}    <: AbstractArrayStyle{N} end   # MultiHaloArray, ArrayOfHaloArray
struct MaybeHaloArrayStyle{N}    <: AbstractArrayStyle{N} end   # MaybeHaloArray

const _HaloStyle{N} = Union{HaloArrayStyle{N},ThreadedHaloArrayStyle{N},
                            MultiHaloArrayStyle{N},MaybeHaloArrayStyle{N}}

for S in (:HaloArrayStyle, :ThreadedHaloArrayStyle, :MultiHaloArrayStyle, :MaybeHaloArrayStyle)
    @eval begin
        $S(::Val{N}) where {N} = $S{N}()
        $S{M}(::Val{N}) where {M,N} = $S{N}()   # Base's convention: Style{M}(Val(N)) is Style{N}
        Broadcast.BroadcastStyle(::$S{N}, ::$S{M}) where {N,M} = $S(Val(max(N, M)))
    end
end
Broadcast.BroadcastStyle(s::_HaloStyle, ::DefaultArrayStyle{0}) = s
Broadcast.BroadcastStyle(::_HaloStyle{N}, ::DefaultArrayStyle{M}) where {N,M} =
    DefaultArrayStyle(Val(max(M, N)))
Broadcast.BroadcastStyle(::_HaloStyle{N}, a::AbstractArrayStyle{M}) where {N,M} =
    typeof(a)(Val(max(M, N)))

Broadcast.BroadcastStyle(::Type{<:HaloArray{T,N}}) where {T,N} = HaloArrayStyle{N}()
Broadcast.BroadcastStyle(::Type{<:LocalHaloArray{T,N}}) where {T,N} = HaloArrayStyle{N}()
Broadcast.BroadcastStyle(::Type{<:ThreadedHaloArray{T,N}}) where {T,N} = ThreadedHaloArrayStyle{N}()

_mixed_halo_backend_broadcast_error() =
    throw(ArgumentError("broadcast between threaded and non-threaded halo containers is not supported"))

Broadcast.BroadcastStyle(::ThreadedHaloArrayStyle, ::HaloArrayStyle) =
    _mixed_halo_backend_broadcast_error()

Broadcast.BroadcastStyle(::HaloArrayStyle, ::ThreadedHaloArrayStyle) =
    _mixed_halo_backend_broadcast_error()


# ------------------------------------------------------------------------------
# Broadcast setup for HaloArray
# ------------------------------------------------------------------------------

Broadcast.broadcastable(x::HaloArray) = x
Broadcast.broadcastable(x::LocalHaloArray) = x
Broadcast.broadcastable(x::ThreadedHaloArray) = x

# ---- Shared tree walkers -----------------------------------------------------
# First operand of type `T` in a broadcast tree: the prototype `similar` builds
# the destination from.
_find_operand(::Type{T}, bc::Broadcasted) where {T} = _find_operand(T, bc.args)
_find_operand(::Type{T}, args::Tuple) where {T} =
    _find_operand(T, _find_operand(T, args[1]), Base.tail(args))
_find_operand(::Type{T}, x) where {T} = x
_find_operand(::Type{T}, x::T, rest) where {T} = x
_find_operand(::Type{T}, x, rest) where {T} = _find_operand(T, rest)

# Rebuild a broadcast tree with `leaf` applied to every operand. Nodes whose
# style is a `Drop` lose it (it is recomputed from the unpacked operands);
# other nodes keep theirs.
@inline _map_operands(leaf::F, bc::Broadcasted{S}, ::Type{Drop}) where {F,S,Drop} =
    Broadcasted{S}(bc.f, _map_args(leaf, bc.args, Drop))
@inline _map_operands(leaf::F, bc::Broadcasted{<:Drop}, ::Type{Drop}) where {F,Drop} =
    Broadcasted(bc.f, _map_args(leaf, bc.args, Drop))
@inline _map_operands(leaf::F, x, ::Type{Drop}) where {F,Drop} = leaf(x)
@inline _map_args(leaf::F, args::Tuple, ::Type{Drop}) where {F,Drop} =
    (_map_operands(leaf, args[1], Drop), _map_args(leaf, Base.tail(args), Drop)...)
_map_args(::F, ::Tuple{}, ::Type{Drop}) where {F,Drop} = ()

# The destination of an out-of-place broadcast: Base's rule, the result
# element type combined from `f` and the operands (so `Float32.(u)`, `u .> v`
# or `u .+ 1im` give Float32, Bool, ComplexF64 arrays), falling back to the
# prototype's element type when inference gives nothing concrete.
@inline function _broadcast_dest(bc::Broadcasted)
    ElType = Broadcast.combine_eltypes(bc.f, bc.args)
    return isconcretetype(ElType) ? similar(bc, ElType) : similar(bc)
end

# Single-block arrays broadcast over their interior views.
_interior_leaf(x::AbstractSingleHaloArray) = interior_view(x)
_interior_leaf(x) = x
@inline unpack_ha(bc::Broadcasted) = _map_operands(_interior_leaf, bc, HaloArrayStyle)

# A plain array (or a single-block halo array's interior) in a THREADED
# broadcast is indexed by GLOBAL interior coordinates, so each tile must see
# its own global window of it — passing the whole operand to every tile
# mismatches the tile-sized destination. `ref` (the destination) supplies the
# tile geometry; dims where the operand has size 1 keep Base's broadcast
# expansion, anything else must span the global interior.
@inline function _tile_global_view(x::AbstractArray, ref::ThreadedHaloArray{T,N},
        tile_id::Integer) where {T,N}
    Base.require_one_based_indexing(x)
    gsz = _global_size(ref)
    # the tile's global window, from the shared tile-origin helper
    gr  = _tile_interior_range(tile_coordinates(ref, tile_id), tile_size(ref))
    rngs = ntuple(Val(N)) do d
        szd = size(x, d)
        szd == 1      ? (1:1) :
        szd == gsz[d] ? gr[d] :
        throw(DimensionMismatch(
            "broadcast operand of size $(size(x)) is incompatible with the " *
            "global interior $(gsz) of the threaded destination (each " *
            "dimension must match the global extent or be 1)"))
    end
    return view(x, rngs...)
end

unpack_ha_tile(x::ThreadedHaloArray, tile_id, ref) = interior_view(x, tile_id)
# A SERIAL single-block halo array's interior spans the same GLOBAL grid:
# slice this tile's window of it, like any global-shaped operand. (Only serial
# backends: a distributed HaloArray's interior is rank-LOCAL, so slicing it by
# global tile windows would read the wrong cells — refuse it explicitly.)
unpack_ha_tile(x::AbstractSerialHaloArray, tile_id, ref) =
    _tile_global_view(interior_view(x), ref, tile_id)
unpack_ha_tile(::AbstractDistributedHaloArray, tile_id, ref) = throw(ArgumentError(
    "a distributed HaloArray cannot be an operand of a threaded broadcast: its " *
    "interior is rank-local, not global"))
unpack_ha_tile(x::AbstractArray, tile_id, ref) = _tile_global_view(x, ref, tile_id)
unpack_ha_tile(x::AbstractArray{<:Any,0}, tile_id, ref) = x   # 0-d wrapper: scalar-like
unpack_ha_tile(x, tile_id, ref) = x                            # scalars, Ref, types, …

@inline function unpack_ha_tile(bc::Broadcasted{Style}, tile_id, ref) where {Style}
    Broadcasted{Style}(bc.f, unpack_args_ha_tile(tile_id, ref, bc.args))
end

@inline function unpack_ha_tile(
        bc::Broadcasted{<:Union{HaloArrayStyle,ThreadedHaloArrayStyle}}, tile_id, ref)
    Broadcasted(bc.f, unpack_args_ha_tile(tile_id, ref, bc.args))
end

@inline function unpack_args_ha_tile(tile_id, ref, args::Tuple)
    (unpack_ha_tile(args[1], tile_id, ref), unpack_args_ha_tile(tile_id, ref, Base.tail(args))...)
end
unpack_args_ha_tile(tile_id, ref, args::Tuple{Any}) = (unpack_ha_tile(args[1], tile_id, ref),)
unpack_args_ha_tile(tile_id, ref, args::Tuple{}) = ()


# ------------------------------------------------------------------------------
# Broadcast execution
# ------------------------------------------------------------------------------

@inline function Base.copyto!(dest::HaloArray, bc::Broadcasted{<:HaloArrayStyle})
    bc_flat = Broadcast.flatten(bc)
    copyto!(interior_view(dest), unpack_ha(bc_flat))
    return dest
end

@inline function Base.copyto!(dest::LocalHaloArray, bc::Broadcasted{<:HaloArrayStyle})
    bc_flat = Broadcast.flatten(bc)
    copyto!(interior_view(dest), unpack_ha(bc_flat))
    return dest
end

@inline function Base.copyto!(dest::ThreadedHaloArray, bc::Broadcasted{<:ThreadedHaloArrayStyle})
    bc_flat = Broadcast.flatten(bc)
    _foreach_tile(tile_id -> _copyto_threaded_broadcast_tile!(dest, bc_flat, tile_id), dest)
    return dest
end

@inline function Base.copy(bc::Broadcast.Broadcasted{<:HaloArrayStyle})
    bc_flat = Broadcast.flatten(bc)
    dest = _broadcast_dest(bc)
    copyto!(interior_view(dest), unpack_ha(bc_flat))
    return dest
end

@inline function Base.copy(bc::Broadcast.Broadcasted{<:ThreadedHaloArrayStyle})
    bc_flat = Broadcast.flatten(bc)
    dest = _broadcast_dest(bc)
    copyto!(dest, bc_flat)
    return dest
end

function Broadcast.materialize!(dest::HaloArray, bc::Broadcasted)
    bc_flat = Broadcast.flatten(bc)
    Broadcast.materialize!(interior_view(dest),unpack_ha(bc_flat))
    return dest
end

function Broadcast.materialize!(dest::LocalHaloArray, bc::Broadcasted)
    bc_flat = Broadcast.flatten(bc)
    Broadcast.materialize!(interior_view(dest), unpack_ha(bc_flat))
    return dest
end

function Broadcast.materialize!(dest::ThreadedHaloArray, bc::Broadcasted)
    bc_flat = Broadcast.flatten(bc)
    _foreach_tile(tile_id -> _materialize_threaded_broadcast_tile!(dest, bc_flat, tile_id), dest)
    return dest
end

@inline function _copyto_threaded_broadcast_tile!(dest::ThreadedHaloArray, bc_flat, tile_id)
    copyto!(interior_view(dest, tile_id), unpack_ha_tile(bc_flat, tile_id, dest))
    return nothing
end

@inline function _materialize_threaded_broadcast_tile!(dest::ThreadedHaloArray, bc_flat, tile_id)
    Broadcast.materialize!(interior_view(dest, tile_id), unpack_ha_tile(bc_flat, tile_id, dest))
    return nothing
end

# ------------------------------------------------------------------------------
# Allocation
# ------------------------------------------------------------------------------

function Base.similar(bc::Broadcasted{<:HaloArrayStyle}, ::Type{T}) where {T}
    return similar(_find_operand(Union{HaloArray,LocalHaloArray}, bc), T)
end

function Base.similar(bc::Broadcasted{<:HaloArrayStyle})
    return similar(_find_operand(Union{HaloArray,LocalHaloArray}, bc))
end

function Base.similar(bc::Broadcasted{<:ThreadedHaloArrayStyle}, ::Type{T}) where {T}
    return similar(_find_operand(ThreadedHaloArray, bc), T)
end

function Base.similar(bc::Broadcasted{<:ThreadedHaloArrayStyle})
    return similar(_find_operand(ThreadedHaloArray, bc))
end
