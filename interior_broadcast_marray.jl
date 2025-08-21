using Base.Broadcast: Broadcasted, broadcastable, BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle

# Broadcast style marker for MultiHaloArray
struct MultiHaloArrayStyle{Ndim,N} <: AbstractArrayStyle{Ndim} end

MultiHaloArrayStyle{Ndim,N}(::Val{Ndim},::Val{N}) where {Ndim,N} = MultiHaloArrayStyle{Ndim,N}()

# The order is important here. We want to override Base.Broadcast.DefaultArrayStyle to return another Base.Broadcast.DefaultArrayStyle.
Broadcast.BroadcastStyle(a::MultiHaloArrayStyle, ::Base.Broadcast.DefaultArrayStyle{0}) = a
Broadcast.BroadcastStyle(::Type{<:MultiHaloArray{T,Ndim,A,N}}) where {T,Ndim,A,N} =
    MultiHaloArrayStyle{Ndim,N}()

    function Broadcast.BroadcastStyle(::MultiHaloArrayStyle{Ndim,N},
        a::Base.Broadcast.DefaultArrayStyle{M}) where { Ndim,N ,M}
    Base.Broadcast.DefaultArrayStyle(Val(max(M, Ndim)))
end
function Broadcast.BroadcastStyle(::MultiHaloArrayStyle{Ndim,N},
        a::Base.Broadcast.AbstractArrayStyle{M}) where {Ndim,N,M}
        typeof(a)(Val(max(M,Ndim)))
end

function Broadcast.BroadcastStyle(::HaloArrayStyle{M},::MultiHaloArrayStyle{Ndim,N}
        ) where {Ndim,N,M}
        HaloArrayStyle(Val(max(M, Ndim)))
end


function Broadcast.BroadcastStyle(::MultiHaloArrayStyle{Ndim,N},
        ::MultiHaloArrayStyle{Mdim,N}) where {Mdim, Ndim,N}
    MultiHaloArrayStyle(Val(max(Mdim, Ndim)),Val(N))
end

# make vectorofarrays broadcastable so they aren't collected
Broadcast.broadcastable(x::MultiHaloArray) = x

# Find MultiHaloArrayBroadcastable in tree
find_mha(bc::Broadcasted) = find_mha(bc.args)
find_mha(args::Tuple) = find_mha(find_mha(args[1]), Base.tail(args))
find_mha(x) = x
find_mha(::Any, rest) = find_mha(rest)
find_mha(mha::MultiHaloArray, rest) = mha


# drop axes because it is easier to recompute
@inline function unpack_mha(bc::Broadcast.Broadcasted{Style}, i) where {Style}
    Broadcast.Broadcasted{Style}(bc.f, unpack_args_mha(i, bc.args))
end
@inline function unpack_mha(bc::Broadcast.Broadcasted{<:MultiHaloArrayStyle}, i)
    Broadcast.Broadcasted(bc.f, unpack_args_mha(i, bc.args))
end
unpack_mha(x, ::Any) = x
unpack_mha(x::MultiHaloArray, i) = values(x.arrays)[i]

function unpack_mha(x::AbstractArray{T, N}, i) where {T, N}
   x
end

function unpack_mha(x::HaloArray, i) 
    interior_view(x)
end

@inline function unpack_args_mha(i, args::Tuple)
    (unpack_mha(args[1], i), unpack_args_mha(i, Base.tail(args))...)
end
unpack_args_mha(i, args::Tuple{Any}) = (unpack_mha(args[1], i),)
unpack_args_mha(::Any, args::Tuple{}) = ()





@inline function Base.copyto!(dest::MultiHaloArray,bc::Broadcast.Broadcasted{<:MultiHaloArrayStyle{Ndim,N}}) where {Ndim,N}
        bc = Broadcast.flatten(bc)
        out=values(dest.arrays)
for (d, i) in zip(out, 1:N)
    copyto!(d, unpack_mha(bc, i))
end
        dest
end

@inline function Base.copy(bc::Broadcasted{<:MultiHaloArrayStyle{Ndim,N}}) where {Ndim,N}
    bc_args = Broadcast.flatten(bc)
    

    dest = similar(bc)
    out=values(dest.arrays)

for (d, i) in zip(out, 1:N)
    copyto!(d, unpack_mha(bc, i))
end
   
    return dest
end

function Broadcast.materialize!(dest::MultiHaloArray{T, N, A, Len}, bc::Broadcasted) where {T, N, A, Len}
    bc_flat = Broadcast.flatten(bc)
    out=values(dest.arrays)
    for (d, i) in zip(out, 1:Len)
        Broadcast.materialize!(d, unpack_mha(bc_flat, i))
    end
    return dest
end


# Similar: allocate new MultiHaloArray during broadcast
function Base.similar(bc::Broadcasted{<:MultiHaloArrayStyle}, ::Type{T}) where {T}
    mha = find_mha(bc)::MultiHaloArray
    return similar(mha, T)
end

function Base.similar(bc::Broadcasted{<:MultiHaloArrayStyle})
    mha = find_mha(bc)::MultiHaloArray
    return similar(mha)
end


