using Base.Broadcast: Broadcasted, broadcastable, BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle

# ------------------------------------------------------------------------------
# Broadcast style marker for HaloArray
# ------------------------------------------------------------------------------

struct HaloArrayStyle{N} <: AbstractArrayStyle{N} end
HaloArrayStyle(::Val{N}) where {N} = HaloArrayStyle{N}()
HaloArrayStyle{N}(::Val{N}) where {N} = HaloArrayStyle{N}()

# This lets DefaultArrayStyle broadcast correctly with HaloArray
Broadcast.BroadcastStyle(a::HaloArrayStyle, ::Base.Broadcast.DefaultArrayStyle{0}) = a

# BroadcastStyle inference for HaloArray types
Broadcast.BroadcastStyle(::Type{<:HaloArray{T,N}}) where {T,N} = HaloArrayStyle{N}()

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



# ------------------------------------------------------------------------------
# Broadcast setup for HaloArray
# ------------------------------------------------------------------------------

Broadcast.broadcastable(x::HaloArray) = x

# Find first HaloArray in a broadcast expression
find_ha(bc::Broadcasted) = find_ha(bc.args)
find_ha(args::Tuple) = find_ha(find_ha(args[1]), Base.tail(args))
find_ha(x::HaloArray, rest) = x
find_ha(x, rest) = find_ha(rest)
find_ha(x) = x

# Unpack broadcast args per field
unpack_ha(x::HaloArray) = interior_view(x)
unpack_ha(x) = x 
@inline function unpack_ha(bc::Broadcasted{Style}) where {Style}
    Broadcasted{Style}(bc.f, unpack_args_ha(bc.args))
end

@inline function unpack_ha(bc::Broadcasted{<:HaloArrayStyle}) 
    Broadcasted(bc.f, unpack_args_ha(bc.args))
end
unpack_ha(x) = x 
unpack_ha(x::HaloArray) = interior_view(x)


@inline function unpack_args_ha( args::Tuple)
    (unpack_ha(args[1]), unpack_args_ha( Base.tail(args))...)
end
unpack_args_ha( args::Tuple{Any}) = (unpack_ha(args[1]),)






# ------------------------------------------------------------------------------
# Broadcast execution
# ------------------------------------------------------------------------------

@inline function Base.copyto!(dest::HaloArray, bc::Broadcasted{<:HaloArrayStyle})
    bc_flat = Broadcast.flatten(bc)
    copyto!(interior_view(dest), unpack_ha(bc_flat))
    return dest
end


@inline function Base.copy(bc::Broadcast.Broadcasted{<:HaloArrayStyle})
    bc_flat = Broadcast.flatten(bc)
    dest = similar(bc)
    copyto!(interior_view(dest), unpack_ha(bc_flat))
    return dest
end

function Broadcast.materialize!(dest::HaloArray, bc::Broadcasted)
    bc_flat = Broadcast.flatten(bc)
    Broadcast.materialize!(interior_view(dest),unpack_ha(bc_flat))
    return dest
end

# ------------------------------------------------------------------------------
# Allocation
# ------------------------------------------------------------------------------

function Base.similar(bc::Broadcasted{<:HaloArrayStyle}, ::Type{T}) where {T}
    ha = find_ha(bc)::HaloArray
    return similar(ha, T)
end

function Base.similar(bc::Broadcasted{<:HaloArrayStyle})
    ha = find_ha(bc)::HaloArray
    return similar(ha)
end