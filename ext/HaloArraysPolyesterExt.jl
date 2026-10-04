module HaloArraysPolyesterExt

# Polyester `@batch` implementation of the ThreadBackend interface. Loaded only
# when the user has `using Polyester`. See src/thread_backend.jl for the
# interface and the OhMyThreads/Serial backends.

import HaloArrays
using HaloArrays: PolyesterBackend
using HaloArrays.StaticArrays: StaticArray
using Polyester: @batch

@inline function HaloArrays.tile_foreach(::PolyesterBackend, f::F, itr; scheduler=nothing) where {F}
    @batch for i in itr
        f(i)
    end
    return nothing
end

# ---- tile reduction -----------------------------------------------------------
# The tiles are split into at most one contiguous chunk per thread and every
# chunk is reduced inside @batch (all chunks at once: reducing the first one on
# the calling thread before the @batch, as an earlier version did to learn the
# result type, made the whole reduction serial with 2 threads and 2 tiles).
# The partial results are combined in chunk order, so a non-commutative `op`
# sees the tiles in order.
#
# How the partials are kept is a Holy trait, chosen once from the types by
# `_reduction_path` (the compiled code holds no trace of the choice):
#   ClausePath   — Polyester's reduction clause; each thread's partial lives in
#                  Polyester's per-thread storage, nothing is allocated. Taken
#                  for + * min max & | on a plain-bits result, i.e. by every
#                  reduction of the package (sum/norm/dot, maximum/minimum,
#                  all/any), static vectors included.
#   PartialsPath — one typed slot per chunk in a `partials` vector, for any
#                  other operator or a non-bits result.
#   EagerPath    — the result type is not known in advance (or the tiles cannot
#                  be indexed): the first chunk is reduced on the calling thread
#                  to fix the slot type, the rest in parallel.
#
# `f::F, op::OP` force specialisation throughout: these methods only pass `f`
# on, never call it, and Julia does not specialise on such a function argument
# by default — the result type would then be inferred at run time.
struct ClausePath end
struct PartialsPath end
struct EagerPath end

_reduction_path(op, ::Type{R}) where {R} =
    !isconcretetype(R)                              ? EagerPath()  :
    (isbitstype(R) && _identity(op, R) !== nothing) ? ClausePath() :
                                                      PartialsPath()

# Indexable tiles (`1:tile_count` in this package): choose the path from the
# inferred result type of one chunk.
function HaloArrays.tile_mapreduce(::PolyesterBackend, f::F, op::OP, itr::AbstractVector;
        scheduler=nothing) where {F,OP}
    _single_thread(itr) && return mapreduce(f, op, itr)
    R = Base.promote_op(_reduce_chunk, F, OP, typeof(itr), Int, Int)
    return _reduce(_reduction_path(op, R), f, op, itr, R)
end
# Tiles that cannot be indexed (a generator, …): only the eager path applies.
HaloArrays.tile_mapreduce(::PolyesterBackend, f::F, op::OP, itr; scheduler=nothing) where {F,OP} =
    _single_thread(itr) ? mapreduce(f, op, itr) : _eager_reduce(f, op, itr)

_reduce(::ClausePath, f::F, op::OP, itr, ::Type{R}) where {F,OP,R} =
    _clause_reduce(op, f, itr, _identity(op, R))
function _reduce(::PartialsPath, f::F, op::OP, itr, ::Type{R}) where {F,OP,R}
    len, nchunks = _chunking(itr)
    partials = Vector{R}(undef, nchunks)
    @batch for c in 1:nchunks
        @inbounds partials[c] = _reduce_chunk(f, op, itr, c, len)
    end
    return reduce(op, partials)
end
_reduce(::EagerPath, f::F, op::OP, itr, ::Type) where {F,OP} = _eager_reduce(f, op, itr)

# One thread, or one tile: nothing to split.
@inline _single_thread(itr) = min(length(itr), Threads.nthreads()) <= 1

# Tiles per chunk and number of chunks, every chunk non-empty.
@inline function _chunking(itr)
    n   = length(itr)
    len = cld(n, min(n, Threads.nthreads()))
    return len, cld(n, len)
end

# Reduce chunk `c` (tiles `(c-1)*len+1 : min(c*len, n)`) of an indexable `itr`;
# a view of a range is a range, so this allocates nothing for `1:tile_count`.
@inline function _reduce_chunk(f::F, op::OP, itr, c::Int, len::Int) where {F,OP}
    lo = first(eachindex(itr)) + (c - 1) * len
    hi = min(lo + len - 1, last(eachindex(itr)))
    return mapreduce(f, op, view(itr, lo:hi))
end

# ClausePath: the clause takes its operator literally, hence one method each.
for op in (:+, :*, :min, :max, :&, :|)
    @eval function _clause_reduce(::typeof($op), f::F, itr, init::R) where {F,R}
        s = init
        @batch reduction=(($op, s),) for i in itr
            s = $op(s, convert(R, f(i)))
        end
        return s
    end
end

# The operator's identity in the result type, or `nothing` when there is none
# (then PartialsPath). It seeds the final combine; Polyester seeds each
# thread's partial with its own `initializer`, the same values (`zero` for `+`).
# One consequence: a sum whose every term is -0.0 returns +0.0 here, where Base
# returns -0.0 — the value is still zero (== holds).
_identity(op, ::Type) = nothing
_identity(::typeof(+), ::Type{R}) where {R<:Union{Number,StaticArray}} = zero(R)
_identity(::typeof(*), ::Type{R}) where {R<:Number} = one(R)
_identity(::typeof(max), ::Type{R}) where {R<:Real} = typemin(R)
_identity(::typeof(min), ::Type{R}) where {R<:Real} = typemax(R)
_identity(::typeof(&), ::Type{Bool}) = true
_identity(::typeof(|), ::Type{Bool}) = false

# EagerPath: the first chunk, reduced on the calling thread, fixes the slot type.
function _eager_reduce(f::F, op::OP, itr) where {F,OP}
    chunks = collect(Iterators.partition(itr, cld(length(itr), min(length(itr), Threads.nthreads()))))
    first_result = mapreduce(f, op, chunks[1])
    length(chunks) == 1 && return first_result
    partials = Vector{typeof(first_result)}(undef, length(chunks))
    @inbounds partials[1] = first_result
    @batch for c in 2:length(chunks)
        @inbounds partials[c] = mapreduce(f, op, chunks[c])
    end
    return reduce(op, partials)
end

end # module
