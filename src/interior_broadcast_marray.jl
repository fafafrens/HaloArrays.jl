using Base.Broadcast: Broadcasted, broadcastable, BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle

# Broadcast style marker for MultiHaloArray
struct MultiHaloArrayStyle{Ndim} <: AbstractArrayStyle{Ndim} end

MultiHaloArrayStyle{Ndim}(::Val{Ndim}) where {Ndim} = MultiHaloArrayStyle{Ndim}()
# Both collection flavors are aliases of FieldCollection; one method covers them
# everywhere below, and the per-field accessor is the shared _fields.
const MultiHaloArrayLike = FieldCollection

# The order is important here. We want to override Base.Broadcast.DefaultArrayStyle to return another Base.Broadcast.DefaultArrayStyle.
Broadcast.BroadcastStyle(a::MultiHaloArrayStyle, ::Base.Broadcast.DefaultArrayStyle{0}) = a
Broadcast.BroadcastStyle(::Type{<:FieldCollection{T,D}}) where {T,D} =
    MultiHaloArrayStyle{D}()

function Broadcast.BroadcastStyle(::MultiHaloArrayStyle{Ndim},
        a::Base.Broadcast.DefaultArrayStyle{M}) where { Ndim ,M}
    Base.Broadcast.DefaultArrayStyle(Val(max(M, Ndim)))
end
function Broadcast.BroadcastStyle(::MultiHaloArrayStyle{Ndim},
        a::Base.Broadcast.AbstractArrayStyle{M}) where {Ndim,M}
        typeof(a)(Val(max(M,Ndim)))
end

function Broadcast.BroadcastStyle(::HaloArrayStyle{M},::MultiHaloArrayStyle{Ndim}
        ) where {Ndim,M}
        HaloArrayStyle(Val(max(M, Ndim)))
end


function Broadcast.BroadcastStyle(::MultiHaloArrayStyle{Ndim},
        ::MultiHaloArrayStyle{Mdim}) where {Mdim, Ndim}
    MultiHaloArrayStyle(Val(max(Mdim, Ndim)))
end

# make collections broadcastable so they aren't collected
Broadcast.broadcastable(x::FieldCollection) = x

# Find the first FieldCollection in the broadcast tree
find_mha(bc::Broadcasted) = find_mha(bc.args)
find_mha(args::Tuple) = find_mha(find_mha(args[1]), Base.tail(args))
find_mha(x) = x
find_mha(::Any, rest) = find_mha(rest)
find_mha(mha::FieldCollection, rest) = mha


# drop axes because it is easier to recompute
@inline function unpack_mha(bc::Broadcast.Broadcasted{Style}, i) where {Style}
    Broadcast.Broadcasted{Style}(bc.f, unpack_args_mha(i, bc.args))
end
@inline function unpack_mha(bc::Broadcast.Broadcasted{<:MultiHaloArrayStyle}, i)
    Broadcast.Broadcasted(bc.f, unpack_args_mha(i, bc.args))
end
unpack_mha(x, ::Any) = x
unpack_mha(x::FieldCollection, i) = _fields(x)[i]

function unpack_mha(x::AbstractArray{T, N}, i) where {T, N}
   x
end
function unpack_mha(x::HaloArray, i) 
    interior_view(x)
end
function unpack_mha(x::LocalHaloArray, i)
    interior_view(x)
end

@inline function unpack_args_mha(i, args::Tuple)
    (unpack_mha(args[1], i), unpack_args_mha(i, Base.tail(args))...)
end
unpack_args_mha(i, args::Tuple{Any}) = (unpack_mha(args[1], i),)
unpack_args_mha(::Any, args::Tuple{}) = ()


@inline function Base.copyto!(dest::MultiHaloArrayLike, bc::Broadcast.Broadcasted{<:MultiHaloArrayStyle{Ndim}}) where {Ndim}
    bc = Broadcast.flatten(bc)
    out = _fields(dest)
    for i in eachindex(out)
        d = out[i]
        copyto!(d, unpack_mha(bc, i))
    end
    return dest
end


@inline function Base.copy(bc::Broadcasted{<:MultiHaloArrayStyle{Ndim}}) where {Ndim}
    bc_flat = Broadcast.flatten(bc)
    
    dest = similar(bc)
    out = _fields(dest)

    for i in eachindex(out)
        d = out[i]
        copyto!(d, unpack_mha(bc_flat, i))
    end
   
    return dest
end

function Broadcast.materialize!(dest::MultiHaloArrayLike, bc::Broadcasted)
    bc_flat = Broadcast.flatten(bc)
    out = _fields(dest)
    for i in eachindex(out)
        d = out[i]
        Broadcast.materialize!(d, unpack_mha(bc_flat, i))
    end
    return dest
end


# Similar: allocate new MultiHaloArray during broadcast
function Base.similar(bc::Broadcasted{<:MultiHaloArrayStyle}, ::Type{T}) where {T}
    mha = find_mha(bc)
    return similar(mha, T)
end

function Base.similar(bc::Broadcasted{<:MultiHaloArrayStyle})
    mha = find_mha(bc)
    return similar(mha)
end
