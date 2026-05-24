using Base.Broadcast: Broadcasted, broadcastable, BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle

# ------------------------------------------------------------------------------
# Broadcast style marker for HaloArray
# ------------------------------------------------------------------------------

struct HaloArrayStyle{N} <: AbstractArrayStyle{N} end
HaloArrayStyle(::Val{N}) where {N} = HaloArrayStyle{N}()
HaloArrayStyle{N}(::Val{N}) where {N} = HaloArrayStyle{N}()

struct ThreadedHaloArrayStyle{N} <: AbstractArrayStyle{N} end
ThreadedHaloArrayStyle(::Val{N}) where {N} = ThreadedHaloArrayStyle{N}()
ThreadedHaloArrayStyle{N}(::Val{N}) where {N} = ThreadedHaloArrayStyle{N}()

# This lets DefaultArrayStyle broadcast correctly with HaloArray
Broadcast.BroadcastStyle(a::HaloArrayStyle, ::Base.Broadcast.DefaultArrayStyle{0}) = a
Broadcast.BroadcastStyle(a::ThreadedHaloArrayStyle, ::Base.Broadcast.DefaultArrayStyle{0}) = a

# BroadcastStyle inference for HaloArray types
Broadcast.BroadcastStyle(::Type{<:HaloArray{T,N}}) where {T,N} = HaloArrayStyle{N}()
Broadcast.BroadcastStyle(::Type{<:LocalHaloArray{T,N}}) where {T,N} = HaloArrayStyle{N}()
Broadcast.BroadcastStyle(::Type{<:ThreadedHaloArray{T,N}}) where {T,N} = ThreadedHaloArrayStyle{N}()

function Broadcast.BroadcastStyle(::HaloArrayStyle{N}, a::Base.Broadcast.DefaultArrayStyle{M}) where {N,M}
    Base.Broadcast.DefaultArrayStyle(Val(max(M, N)))
end

function Broadcast.BroadcastStyle(::HaloArrayStyle{N},
        a::Base.Broadcast.AbstractArrayStyle{M}) where {M, N}
    typeof(a)(Val(max(M, N)))
end

function Broadcast.BroadcastStyle(::HaloArrayStyle{N}, ::HaloArrayStyle{M}) where {N,M}
    HaloArrayStyle(Val(max(N,M)))
end

function Broadcast.BroadcastStyle(::ThreadedHaloArrayStyle{N}, a::Base.Broadcast.DefaultArrayStyle{M}) where {N,M}
    Base.Broadcast.DefaultArrayStyle(Val(max(M, N)))
end

function Broadcast.BroadcastStyle(::ThreadedHaloArrayStyle{N},
        a::Base.Broadcast.AbstractArrayStyle{M}) where {M,N}
    typeof(a)(Val(max(M, N)))
end

function Broadcast.BroadcastStyle(::ThreadedHaloArrayStyle{N}, ::ThreadedHaloArrayStyle{M}) where {N,M}
    ThreadedHaloArrayStyle(Val(max(N,M)))
end

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

# Find first HaloArray in a broadcast expression
find_ha(bc::Broadcasted) = find_ha(bc.args)
find_ha(args::Tuple) = find_ha(find_ha(args[1]), Base.tail(args))
find_ha(x::HaloArray, rest) = x
find_ha(x::LocalHaloArray, rest) = x
find_ha(x, rest) = find_ha(rest)
find_ha(x) = x

find_threaded_ha(bc::Broadcasted) = find_threaded_ha(bc.args)
find_threaded_ha(args::Tuple) = find_threaded_ha(find_threaded_ha(args[1]), Base.tail(args))
find_threaded_ha(x::ThreadedHaloArray, rest) = x
find_threaded_ha(x, rest) = find_threaded_ha(rest)
find_threaded_ha(x) = x

# Unpack broadcast args per field
unpack_ha(x::HaloArray) = interior_view(x)
unpack_ha(x::LocalHaloArray) = interior_view(x)
unpack_ha(x) = x 
@inline function unpack_ha(bc::Broadcasted{Style}) where {Style}
    Broadcasted{Style}(bc.f, unpack_args_ha(bc.args))
end

@inline function unpack_ha(bc::Broadcasted{<:HaloArrayStyle}) 
    Broadcasted(bc.f, unpack_args_ha(bc.args))
end


@inline function unpack_args_ha( args::Tuple)
    (unpack_ha(args[1]), unpack_args_ha( Base.tail(args))...)
end
unpack_args_ha( args::Tuple{Any}) = (unpack_ha(args[1]),)

unpack_ha_tile(x::ThreadedHaloArray, tile_id) = interior_view(x, tile_id)
unpack_ha_tile(x::HaloArray, tile_id) = interior_view(x)
unpack_ha_tile(x::LocalHaloArray, tile_id) = interior_view(x)
unpack_ha_tile(x, tile_id) = x

@inline function unpack_ha_tile(bc::Broadcasted{Style}, tile_id) where {Style}
    Broadcasted{Style}(bc.f, unpack_args_ha_tile(tile_id, bc.args))
end

@inline function unpack_ha_tile(bc::Broadcasted{<:HaloArrayStyle}, tile_id)
    Broadcasted(bc.f, unpack_args_ha_tile(tile_id, bc.args))
end

@inline function unpack_ha_tile(bc::Broadcasted{<:ThreadedHaloArrayStyle}, tile_id)
    Broadcasted(bc.f, unpack_args_ha_tile(tile_id, bc.args))
end

@inline function unpack_args_ha_tile(tile_id, args::Tuple)
    (unpack_ha_tile(args[1], tile_id), unpack_args_ha_tile(tile_id, Base.tail(args))...)
end
unpack_args_ha_tile(tile_id, args::Tuple{Any}) = (unpack_ha_tile(args[1], tile_id),)
unpack_args_ha_tile(tile_id, args::Tuple{}) = ()


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
    tforeach(tile_id -> _copyto_threaded_broadcast_tile!(dest, bc_flat, tile_id),
        eachindex(parent(dest)); scheduler=:static)
    return dest
end

@inline function Base.copy(bc::Broadcast.Broadcasted{<:HaloArrayStyle})
    bc_flat = Broadcast.flatten(bc)
    dest = similar(bc)
    copyto!(interior_view(dest), unpack_ha(bc_flat))
    return dest
end

@inline function Base.copy(bc::Broadcast.Broadcasted{<:ThreadedHaloArrayStyle})
    bc_flat = Broadcast.flatten(bc)
    dest = similar(bc)
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
    tforeach(tile_id -> _materialize_threaded_broadcast_tile!(dest, bc_flat, tile_id),
        eachindex(parent(dest)); scheduler=:static)
    return dest
end

@inline function _copyto_threaded_broadcast_tile!(dest::ThreadedHaloArray, bc_flat, tile_id)
    copyto!(interior_view(dest, tile_id), unpack_ha_tile(bc_flat, tile_id))
    return nothing
end

@inline function _materialize_threaded_broadcast_tile!(dest::ThreadedHaloArray, bc_flat, tile_id)
    Broadcast.materialize!(interior_view(dest, tile_id), unpack_ha_tile(bc_flat, tile_id))
    return nothing
end

# ------------------------------------------------------------------------------
# Allocation
# ------------------------------------------------------------------------------

function Base.similar(bc::Broadcasted{<:HaloArrayStyle}, ::Type{T}) where {T}
    ha = find_ha(bc)
    return similar(ha, T)
end

function Base.similar(bc::Broadcasted{<:HaloArrayStyle})
    ha = find_ha(bc)
    return similar(ha)
end

function Base.similar(bc::Broadcasted{<:ThreadedHaloArrayStyle}, ::Type{T}) where {T}
    ha = find_threaded_ha(bc)
    return similar(ha, T)
end

function Base.similar(bc::Broadcasted{<:ThreadedHaloArrayStyle})
    ha = find_threaded_ha(bc)
    return similar(ha)
end
