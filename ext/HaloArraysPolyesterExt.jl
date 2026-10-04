module HaloArraysPolyesterExt

# Polyester `@batch` implementation of the ThreadBackend interface. Loaded only
# when the user has `using Polyester`. See src/thread_backend.jl for the
# interface and the OhMyThreads/Serial backends.

import HaloArrays
using HaloArrays: PolyesterBackend
using Polyester: @batch

@inline function HaloArrays.tile_foreach(::PolyesterBackend, f::F, itr; scheduler=nothing) where {F}
    @batch for i in itr
        f(i)
    end
    return nothing
end

# Chunk-and-combine: split the tile range into at most one contiguous chunk per
# thread, reduce every chunk inside @batch into its own slot of `partials`
# (race-free — each iteration writes a distinct index), then combine the slots
# in order, so a non-commutative `op` sees the tiles in order.
#
# The slot type comes from inference (`promote_op`), so all chunks, the first
# included, run in parallel. (Reducing the first chunk eagerly to learn the type
# serialised it before the @batch: with 2 threads and 2 tiles the whole
# reduction ran on the calling thread.) Chunk ranges are computed from the chunk
# index, so an indexable range of tiles allocates nothing but `partials`.
# `f::F, op::OP` force specialisation: `f` is only passed on here, never called,
# and Julia does not specialise on such a function argument by default — the
# result type would then be inferred at run time and `partials` left untyped.
function HaloArrays.tile_mapreduce(::PolyesterBackend, f::F, op::OP, itr; scheduler=nothing) where {F,OP}
    n = length(itr)
    n_threads = min(n, Threads.nthreads())
    n_threads <= 1 && return mapreduce(f, op, itr)
    itr isa AbstractVector || return _eager_tile_mapreduce(f, op, itr, n_threads)
    len     = cld(n, n_threads)          # tiles per chunk
    nchunks = cld(n, len)                # every chunk non-empty
    R = Base.promote_op(_reduce_chunk, typeof(f), typeof(op), typeof(itr), Int, Int)
    isconcretetype(R) || return _eager_tile_mapreduce(f, op, itr, n_threads)
    partials = Vector{R}(undef, nchunks)
    @batch for c in 1:nchunks
        @inbounds partials[c] = _reduce_chunk(f, op, itr, c, len)
    end
    return reduce(op, partials)
end

# Reduce chunk `c` (tiles `(c-1)*len+1 : min(c*len, n)`) of an indexable `itr`;
# a view of a range is a range, so this allocates nothing for `1:tile_count`.
@inline function _reduce_chunk(f::F, op::OP, itr, c::Int, len::Int) where {F,OP}
    lo = first(eachindex(itr)) + (c - 1) * len
    hi = min(lo + len - 1, last(eachindex(itr)))
    return mapreduce(f, op, view(itr, lo:hi))
end

# Fallback for a non-indexable `itr` or a non-concrete inferred result: the
# previous eager scheme (first chunk on the calling thread fixes the slot type).
function _eager_tile_mapreduce(f::F, op::OP, itr, n_threads) where {F,OP}
    chunks = collect(Iterators.partition(itr, cld(length(itr), n_threads)))
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
