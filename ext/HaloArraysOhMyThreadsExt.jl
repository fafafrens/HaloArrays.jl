module HaloArraysOhMyThreadsExt

# OhMyThreads implementation of the ThreadBackend interface. Loaded only when
# the user has `using OhMyThreads`. See src/thread_backend.jl for the interface
# and the Base-threads (default) and Serial backends.

import HaloArrays
using HaloArrays: OhMyThreadsBackend
using OhMyThreads: tforeach, tmapreduce

@inline HaloArrays.tile_foreach(::OhMyThreadsBackend, f, itr; scheduler=:dynamic) =
    tforeach(f, itr; scheduler)
@inline HaloArrays.tile_mapreduce(::OhMyThreadsBackend, f, op, itr; scheduler=:dynamic) =
    tmapreduce(f, op, itr; scheduler)

end # module
