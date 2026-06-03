# ------------------------------------------------------------------------------
# Thread-execution backends
#
# A `ThreadBackend` selects *how* per-tile work is dispatched across threads.
# This is orthogonal to `halo_backend` (which describes the storage and
# communication layout): a single `ThreadedHaloArray` carries both.
#
# The entire interface is two operations over the tile index range:
#   tile_foreach(backend, f, itr)            — parallel `foreach`
#   tile_mapreduce(backend, f, op, itr)      — parallel `mapreduce`
#
# Users can add a backend simply by defining these two methods for a new
# `<:ThreadBackend`; nothing else in the package needs to change.
# ------------------------------------------------------------------------------

abstract type ThreadBackend end

"""
    OhMyThreadsBackend()

Dispatch per-tile work as OhMyThreads tasks. The default backend.
"""
struct OhMyThreadsBackend <: ThreadBackend end

"""
    SerialBackend()

Run per-tile work serially, on the calling thread. Useful for debugging data
races and for deterministic, single-threaded runs.
"""
struct SerialBackend <: ThreadBackend end

"""
    PolyesterBackend()

Dispatch per-tile work with Polyester's `@batch` (low task-spawn overhead).
Requires `using Polyester`; the methods live in the `HaloArraysPolyesterExt`
package extension.
"""
struct PolyesterBackend <: ThreadBackend end

"""
    tile_foreach(backend, f, itr; scheduler=:dynamic)

Apply `f` to every element of `itr` in parallel, according to `backend`. The
`scheduler` hint is honoured by [`OhMyThreadsBackend`](@ref) and ignored by the
others.
"""
function tile_foreach end

"""
    tile_mapreduce(backend, f, op, itr; scheduler=:dynamic)

Parallel `mapreduce(f, op, itr)` according to `backend`. `itr` must be non-empty
(in this package there is always at least one tile).
"""
function tile_mapreduce end

# --- OhMyThreads (default; OhMyThreads is a hard dependency) ------------------
@inline tile_foreach(::OhMyThreadsBackend, f, itr; scheduler=:dynamic) =
    tforeach(f, itr; scheduler)
@inline tile_mapreduce(::OhMyThreadsBackend, f, op, itr; scheduler=:dynamic) =
    tmapreduce(f, op, itr; scheduler)

# --- Serial -------------------------------------------------------------------
@inline tile_foreach(::SerialBackend, f, itr; scheduler=nothing) = (foreach(f, itr); nothing)
@inline tile_mapreduce(::SerialBackend, f, op, itr; scheduler=nothing) = mapreduce(f, op, itr)

# --- availability check (used at construction for a friendly early error) -----
_require_thread_backend(::ThreadBackend) = nothing
function _require_thread_backend(::PolyesterBackend)
    if isnothing(Base.get_extension(@__MODULE__, :HaloArraysPolyesterExt))
        throw(ArgumentError(
            "PolyesterBackend requires loading Polyester first (`using Polyester`); " *
            "its tile_foreach/tile_mapreduce methods live in the " *
            "HaloArraysPolyesterExt package extension."))
    end
    return nothing
end
