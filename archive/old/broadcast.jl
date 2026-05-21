using Base.Broadcast: Broadcasted, broadcastable, BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle

# Broadcast style marker for HaloArray
struct HaloArrayStyle{N} <: AbstractArrayStyle{N} end

# Wrapper for broadcasting, needed to implement custom rules
struct HaloArrayBroadcastable{T,N, A , Halo, Size, B, C, BCondition}
    data::HaloArray{T, N, A, Halo, Size, B, C, BCondition}
end

# Tell Julia how to broadcast HaloArray: wrap in HaloArrayBroadcastable
Broadcast.broadcastable(ha::HaloArray{T, N, A, Halo, Size, B, C, BCondition}) where {T, N, A, Halo, Size, B, C, BCondition} = HaloArrayBroadcastable{T,N,A,Halo,Size,B,C,BCondition}(ha)

# Broadcast style for HaloArrayBroadcastable
Broadcast.BroadcastStyle(::Type{<:HaloArrayBroadcastable{T, N, A, Halo, Size, B, C, BCondition}}) where {T,N,A,Halo,Size,B,C,BCondition} = HaloArrayStyle{N}()

# Make HaloArrayStyle win over other styles
Broadcast.BroadcastStyle(::HaloArrayStyle{N}, ::AbstractArrayStyle) where {N} = HaloArrayStyle{N}()
Broadcast.BroadcastStyle(::HaloArrayStyle{N}, ::DefaultArrayStyle) where {N} = HaloArrayStyle{N}()

# Size of the broadcasted HaloArray is the size of the underlying data array (including halo)
Base.size(bc::HaloArrayBroadcastable{T, N, A, Halo, Size, B, C, BCondition}) where {T, N, A, Halo, Size, B, C, BCondition} = size(bc.data)

Base.eltype(::Type{<:HaloArrayBroadcastable{T, N, A, Halo, Size, B, C, BCondition}}) where {T,N,A,Halo,Size,B,C,BCondition} = T

# Unwrap function to get underlying array for broadcasting operations
function _unwrap_ha(bc::Broadcasted{Style}) where {Style}
    args = map(_unwrap_ha, bc.args)
    axs = axes(bc)
    if Style === Nothing
        Broadcasted{Nothing}(bc.f, args, axs)
    else
        Broadcasted(bc.f, args, axs)
    end
end

_unwrap_ha(ha::HaloArrayBroadcastable) = ha.data.data
_unwrap_ha(x) = x



# materialize! to fill an existing HaloArray with broadcasted values
function Broadcast.materialize!(dest::HaloArray, bc_in::Broadcasted)
    # Unwrap all HaloArrayBroadcastables so broadcast falls back on data arrays
    bc = _unwrap_ha(bc_in)
    # Broadcast materialize into the underlying data array (including halos)
    Broadcast.materialize!(dest.data, bc)
    return dest
end

# copyto! to fill existing HaloArray with broadcasted values
function Base.copyto!(dest::HaloArray, bc_in::Broadcasted{Nothing})
    bc = _unwrap_ha(bc_in)
    copyto!(dest.data, bc)
    return dest
end

# copyto! when dest is HaloArrayBroadcastable, redirect to underlying HaloArray.data
function Base.copyto!(dest::HaloArrayBroadcastable, bc::Broadcasted{Nothing})
    copyto!(dest.data, bc)
    return dest
end

function Base.copy(bc::Broadcasted{<:HaloArrayStyle})
    bc_unwrapped = _unwrap_ha(bc)
    dest = Base.similar(bc)
    Base.copyto!(dest.data, bc_unwrapped)
    return dest
end


# Optionally, define eltype and size to make broadcasting more transparent
Base.eltype(::Type{<:HaloArray{T, N, A,H,S,B,C,BC}}) where {T, N, A,H,S,B,C,BC} = T


# Optional: similar() fallback (if you use broadcast to allocate new HaloArrays)
function Base.similar(bc::Broadcasted{<:HaloArrayStyle}, ::Type{T}) where {T}
    ha = find_ha(bc)::HaloArrayBroadcastable
    A = ha.data
    axes(bc) == axes(A.data) || throw(DimensionMismatch("axes mismatch in broadcast"))
    return similar(A, T)
end

function Base.similar(bc::Broadcasted{<:HaloArrayStyle})
    ha = find_ha(bc)::HaloArrayBroadcastable
    A = ha.data
    axes(bc) == axes(A.data) || throw(DimensionMismatch("axes mismatch in broadcast"))
    return similar(A)
end

# Find HaloArray inside broadcast tree
find_ha(bc::Broadcasted) = find_ha(bc.args)
find_ha(args::Tuple) = find_ha(find_ha(args[1]), Base.tail(args))
find_ha(x) = x
find_ha(::Any, rest) = find_ha(rest)
find_ha(halo::HaloArrayBroadcastable, rest) = halo