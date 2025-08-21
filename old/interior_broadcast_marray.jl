using Base.Broadcast: Broadcasted, broadcastable, BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle

# Broadcast style marker for MultiHaloArray
struct MultiHaloArrayStyle{N} <: AbstractArrayStyle{N} end

# Custom broadcastable wrapper
struct MultiHaloArrayBroadcastable{T, N, A <: NamedTuple}
    data::MultiHaloArray{T, N, A}
end

# Tell Julia how to broadcast MultiHaloArray
Broadcast.broadcastable(mha::MultiHaloArray{T, N, A}) where {T, N, A} =
    MultiHaloArrayBroadcastable{T, N, A}(mha)

# Broadcast style logic
Broadcast.BroadcastStyle(::Type{<:MultiHaloArrayBroadcastable{T, N, A}}) where {T, N, A} =
    MultiHaloArrayStyle{N}()

Broadcast.BroadcastStyle(::MultiHaloArrayStyle{N}, ::AbstractArrayStyle) where {N} = MultiHaloArrayStyle{N}()
Broadcast.BroadcastStyle(::MultiHaloArrayStyle{N}, ::DefaultArrayStyle) where {N} = MultiHaloArrayStyle{N}()
Broadcast.BroadcastStyle(::MultiHaloArrayStyle{N}, ::HaloArrayStyle{N}) where {N} = MultiHaloArrayStyle{N}()
Broadcast.BroadcastStyle(::HaloArrayStyle{N}, ::MultiHaloArrayStyle{N}) where {N} = MultiHaloArrayStyle{N}()

#Broadcast.BroadcastStyle(::HaloArrayStyle{N}, ::MultiHaloArrayStyle{M}) where {N, M} = MultiHaloArrayStyle{max(N, M)}()
#Broadcast.BroadcastStyle(::MultiHaloArrayStyle{N}, ::HaloArrayStyle{M}) where {N, M} = MultiHaloArrayStyle{max(N, M)}()



#Base.size(bc::MultiHaloArrayBroadcastable) = size(bc.data)[2:end]
#Base.axes(bc::MultiHaloArrayBroadcastable) = axes(bc.data)[2:end]
Base.size(bc::MultiHaloArrayBroadcastable) = size(interior_view(first(values(bc.data.arrays))))
Base.axes(bc::MultiHaloArrayBroadcastable) = axes(interior_view(first(values(bc.data.arrays))))

#Base.axes(bc::MultiHaloArrayBroadcastable) = axes(interior_view(bc.data))


# Broadcast tree lowering
#function _unwrap_mha(bc::Broadcasted{Style}) where {Style}
#    args = map(_unwrap_mha, bc.args)
#    axs = axes(bc)
#
#    if all(x -> x isa Tuple, args)
#        n = length(args[1])
#        return map(i -> Broadcasted{Nothing}(bc.f, map(a -> a[i], args), axs), 1:n)
#    else
#        nfields = maximum(x -> x isa Tuple ? length(x) : 1, args)
#        expanded = map(arg -> arg isa Tuple ? arg : ntuple(_ -> arg, nfields), args)
#        return map(i -> Broadcasted{Nothing}(bc.f, map(a -> a[i], expanded), axs), 1:nfields)
#    end
#end

#function _unwrap_mha(bc::Broadcasted)
#    args = map(_unwrap_mha, bc.args)  # recursively unwrap args
#    if all(x -> x isa Tuple, args)
#        n = length(args[1])
#        return map(i -> Base.broadcasted(bc.f, map(a -> a[i], args)...), 1:n)
#    else
#        nfields = maximum(x -> x isa Tuple ? length(x) : 1, args)
#        expanded = map(arg -> arg isa Tuple ? arg : ntuple(_ -> arg, nfields), args)
#        return map(i -> Base.broadcasted(bc.f, map(a -> a[i], expanded)...), 1:nfields)
#    end
#end

# Base cases
function _unwrap_mha(bc::Broadcasted{Style}) where {Style}
    args = map(_unwrap_mha, bc.args)
    axs = axes(bc)
    if Style === Nothing
        Broadcasted{Nothing}(bc.f, args, axs)
    else
        Broadcasted(bc.f, args, axs)
    end
end


_unwrap_mha(mha::MultiHaloArrayBroadcastable) = mha
_unwrap_mha(ha::HaloArrayBroadcastable) = _unwrap_ha(ha)
_unwrap_mha(x) = x




#function _unwrap_mha(bc::Broadcasted{Style}) where {Style}
#    args = map(_unwrap_mha, bc.args)
#    axs = axes(bc)
#
#    if all(x -> x isa Tuple, args)
#        n = length(args[1])
#        return map(i -> Base.broadcasted(bc.f, map(a -> a[i], args)...), 1:n)
#    else
#        nfields = maximum(x -> x isa Tuple ? length(x) : 1, args)
#        expanded = map(arg -> arg isa Tuple ? arg : ntuple(_ -> arg, nfields), args)
#        return map(i -> Base.broadcasted(bc.f, map(a -> a[i], expanded)...), 1:nfields)
#    end
#end
#
#_unwrap_mha(mha::MultiHaloArrayBroadcastable) = map(interior_view, values(mha.data.arrays))
#_unwrap_mha(ha::HaloArrayBroadcastable) = interior_view(ha.data)
#_unwrap_mha(x) = x
function apply_tuple!(func,dest::A,arg::B) where {A<:Tuple,B<:Tuple}
        for (d, bc) in zip(dest,arg)
            func(interior_view(d), bc)
        end
end 

function apply_tuple!(func,dest::A,arg::B) where {A<:Tuple,B}
        
        for d in dest
            func(interior_view(d), arg)
        end
end


# Materialize into interior
function Broadcast.materialize!(dest::MultiHaloArray, bc_in::Broadcasted)
    bc_args = _unwrap_mha(bc_in)
    for (d, bc) in zip(values(dest.arrays),Iterators.cycle(bc_args))
            Broadcast.materialize!(d, bc)
    end
    #apply_tuple!(Broadcast.materialize!,values(dest.arrays),bc_args)
    return dest
end

# Copyto! into interior
function Base.copyto!(dest::MultiHaloArray, bc_in::Broadcasted{Nothing})
    bc_args = _unwrap_mha(bc_in)
    for (d, bc) in zip(values(dest.arrays),Iterators.cycle(bc_args))
        copyto!(d, bc)
    end
    #apply_tuple!(copyto!,values(dest.arrays),bc_args)
    return dest
end

@inline function Base.copyto!(dest::MultiHaloArray,
        bc::Broadcast.Broadcasted{MultiHaloArrayStyle{N}}) where {N
}

end

# Copyto! from broadcastable wrapper
Base.copyto!(dest::MultiHaloArrayBroadcastable, bc::Broadcasted{Nothing}) = copyto!(dest.data, bc)

# Copy: returns new MultiHaloArray with broadcast results in interiors
function Base.copy(bc::Broadcasted{<:MultiHaloArrayStyle})
    bc_args = _unwrap_mha(bc)
    @show bc_args

    dest = Base.similar(bc)

    for (d, bc) in zip(values(dest.arrays),Iterators.cycle(bc_args))
        copyto!(d, bc)
    end
    #apply_tuple!(copyto!,values(dest.arrays),bc_args)
    return dest
end

# Similar: allocate new MultiHaloArray during broadcast
function Base.similar(bc::Broadcasted{<:MultiHaloArrayStyle}, ::Type{T}) where {T}
    mha = find_mha(bc)::MultiHaloArrayBroadcastable
    return similar(mha.data, T)
end

function Base.similar(bc::Broadcasted{<:MultiHaloArrayStyle})
    mha = find_mha(bc)::MultiHaloArrayBroadcastable
    return similar(mha.data)
end

# Find MultiHaloArrayBroadcastable in tree
find_mha(bc::Broadcasted) = find_mha(bc.args)
find_mha(args::Tuple) = find_mha(find_mha(args[1]), Base.tail(args))
find_mha(x) = x
find_mha(::Any, rest) = find_mha(rest)
find_mha(mha::MultiHaloArrayBroadcastable, rest) = mha