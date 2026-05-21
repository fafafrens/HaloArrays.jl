using Base.Broadcast: Broadcasted, broadcastable, BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle

# Broadcast style marker for HaloArray
struct HaloArrayStyle{N} <: AbstractArrayStyle{N} end

# Wrapper for broadcasting
struct HaloArrayBroadcastable{T,N,A,Halo,Size,B,C,BC}
    data::HaloArray{T,N,A,Halo,Size,B,C,BC}
end

# Tell Julia how to broadcast HaloArray: wrap in HaloArrayBroadcastable
Broadcast.broadcastable(ha::HaloArray{T,N,A,Halo,Size,B,C,BC}) where {T,N,A,Halo,Size,B,C,BC} =
    HaloArrayBroadcastable{T,N,A,Halo,Size,B,C,BC}(ha)

# Define the broadcast style
Broadcast.BroadcastStyle(::Type{<:HaloArrayBroadcastable{T,N,A,Halo,Size,B,C,BC}}) where {T,N,A,Halo,Size,B,C,BC} =
    HaloArrayStyle{N}()

Broadcast.BroadcastStyle(::HaloArrayStyle{N}, ::AbstractArrayStyle) where {N} = HaloArrayStyle{N}()
Broadcast.BroadcastStyle(::HaloArrayStyle{N}, ::DefaultArrayStyle) where {N} = HaloArrayStyle{N}()

# Restrict broadcast shape to interior
Base.size(bc::HaloArrayBroadcastable) = size(interior_view(bc.data))
Base.axes(bc::HaloArrayBroadcastable) = axes(interior_view(bc.data))

# Type info
Base.eltype(::Type{<:HaloArrayBroadcastable{T,N,A,H,S,B,C,BC}}) where {T,N,A,H,S,B,C,BC} = T
Base.eltype(::Type{<:HaloArray{T,N,A,H,S,B,C,BC}}) where {T,N,A,H,S,B,C,BC} = T

# Unwrap for recursive broadcast lowering
#_unwrap_ha(bc::Broadcasted{Style}) where {Style} = Broadcasted(Style, map(_unwrap_ha, bc.args), axes(bc))
function _unwrap_ha(bc::Broadcasted{Style}) where {Style}
    args = map(_unwrap_ha, bc.args)
    axs = axes(bc)
    if Style === Nothing
        Broadcasted{Nothing}(bc.f, args, axs)
    else
        Broadcasted(bc.f, args, axs)
    end
end
_unwrap_ha(ha::HaloArrayBroadcastable) = interior_view(ha.data)
_unwrap_ha(x) = x

# Materialize into interior
function Broadcast.materialize!(dest::HaloArray, bc_in::Broadcasted)
    bc = _unwrap_ha(bc_in)
    Broadcast.materialize!(interior_view(dest), bc)
    return dest
end

# Copyto! into interior
function Base.copyto!(dest::HaloArray, bc_in::Broadcasted{Nothing})
    bc = _unwrap_ha(bc_in)
    copyto!(interior_view(dest), bc)
    return dest
end

# Copyto! from broadcastable wrapper
Base.copyto!(dest::HaloArrayBroadcastable, bc::Broadcasted{Nothing}) = copyto!(dest.data, bc)

# Copy (returns new HaloArray with interior broadcast result)
function Base.copy(bc::Broadcasted{<:HaloArrayStyle})
    bc_unwrapped = _unwrap_ha(bc)
    dest = Base.similar(bc)
    copyto!(interior_view(dest), bc_unwrapped)
    return dest
end

# similar fallback (used when allocating a new result in broadcast)
function Base.similar(bc::Broadcasted{<:HaloArrayStyle}, ::Type{T}) where {T}
    ha = find_ha(bc)::HaloArrayBroadcastable
    return similar(ha.data, T)
end

function Base.similar(bc::Broadcasted{<:HaloArrayStyle})
    ha = find_ha(bc)::HaloArrayBroadcastable
    return similar(ha.data)
end

# Find a HaloArray in broadcast tree
find_ha(bc::Broadcasted) = find_ha(bc.args)
find_ha(args::Tuple) = find_ha(find_ha(args[1]), Base.tail(args))
find_ha(x) = x
find_ha(::Any, rest) = find_ha(rest)
find_ha(halo::HaloArrayBroadcastable, rest) = halo