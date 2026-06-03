module HaloArraysPolyesterExt

# Polyester `@batch` implementation of the ThreadBackend interface. Loaded only
# when the user has `using Polyester`. See src/thread_backend.jl for the
# interface and the OhMyThreads/Serial backends.

import HaloArrays
using HaloArrays: PolyesterBackend
using Polyester: @batch

@inline function HaloArrays.tile_foreach(::PolyesterBackend, f, itr; scheduler=nothing)
    @batch for i in itr
        f(i)
    end
    return nothing
end

# Option B (chunk-and-combine): split the (small) tile range into one chunk per
# thread, reduce each chunk serially into its own slot (race-free — each @batch
# iteration writes a distinct index), then combine the slots serially. This uses
# `@batch` only in its plain foreach form, avoiding Polyester's reduction-clause
# quirks and keeping the result type explicit/stable.
function HaloArrays.tile_mapreduce(::PolyesterBackend, f, op, itr; scheduler=nothing)
    n       = length(itr)
    nchunks = min(n, Threads.nthreads())
    chunks  = collect(Iterators.partition(itr, cld(n, nchunks)))

    # Reduce the first chunk eagerly: its value gives the partials' (concrete)
    # element type and is kept, so no work is discarded. With a single chunk
    # (e.g. one tile, or one thread) this is the whole answer — no @batch.
    first_result = mapreduce(f, op, chunks[1])
    nchunks == 1 && return first_result

    partials = Vector{typeof(first_result)}(undef, nchunks)
    @inbounds partials[1] = first_result
    @batch for c in 2:nchunks
        @inbounds partials[c] = mapreduce(f, op, chunks[c])
    end
    return reduce(op, partials)
end

end # module
