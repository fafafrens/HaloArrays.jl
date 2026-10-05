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

"""
    ThreadBackend

Supertype for the thread-execution backends of a [`ThreadedHaloArray`](@ref) —
*how* per-tile work is dispatched across threads (orthogonal to `halo_backend`,
which is *where* data lives). Built-ins: [`ThreadsBackend`](@ref) (Base tasks, the
default), [`SerialBackend`](@ref), [`PolyesterBackend`](@ref) (needs `using
Polyester`) and [`OhMyThreadsBackend`](@ref) (needs `using OhMyThreads`). Add your
own by defining [`tile_foreach`](@ref) and [`tile_mapreduce`](@ref) for a new
subtype.
"""
abstract type ThreadBackend end

"""
    ThreadsBackend()

Dispatch per-tile work as Base tasks (`Threads.@spawn`); no extra package. The
default backend. The tiles are split into one contiguous chunk per thread; all
chunks but the first are spawned and the calling thread works on the first one
meanwhile, so every chunk runs at once. Tasks compose with other Julia tasks,
so it is safe inside other parallel code (unlike [`PolyesterBackend`](@ref)).
"""
struct ThreadsBackend <: ThreadBackend end

"""
    OhMyThreadsBackend()

Dispatch per-tile work as OhMyThreads tasks, honouring the `scheduler` keyword
of [`tile_foreach`](@ref) (e.g. dynamic load balancing for uneven work).
Requires `using OhMyThreads`; the methods live in the `HaloArraysOhMyThreadsExt`
package extension.
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
    tile_foreach(backend, f, itr; scheduler)
    tile_foreach(f, backend, itr; scheduler)
    tile_foreach(f, u::AbstractSingleHaloArray)

Apply `f` to every element of `itr` in parallel, according to `backend`. The
`scheduler` hint (`:dynamic` by default) is honoured by
[`OhMyThreadsBackend`](@ref) and ignored by the others.

The function-first forms exist for `do`-block syntax, which always passes the
closure as the first argument. The array form runs the per-tile kernel
`f(tile_id)` over `1:tile_count(u)` using `u`'s own tile driver — inline on a
single-block array (Local/MPI: one tile), across `thread_backend(u)` on a
[`ThreadedHaloArray`](@ref):

```julia
tile_foreach(u) do tile
    s = tile_parent(u, tile)
    # per-tile work; touch only this tile — tiles may run concurrently
end
```

For explicit scheduler control fall back to the backend form,
`tile_foreach(thread_backend(u), f, 1:tile_count(u); scheduler=…)`.
"""
function tile_foreach end

"""
    tile_mapreduce(backend, f, op, itr; scheduler)
    tile_mapreduce(f, op, backend, itr; scheduler)
    tile_mapreduce(f, op, u::AbstractSingleHaloArray)

Parallel `mapreduce(f, op, itr)` according to `backend`. `itr` must be non-empty
(in this package there is always at least one tile). The function-first forms
exist for `do`-block syntax: `tile_mapreduce(+, backend, itr) do tile … end`.
The array form reduces `f(tile_id)` over `1:tile_count(u)` with `u`'s own tile
driver: `tile_mapreduce(+, u) do tile … end`.
"""
function tile_mapreduce end

# `do`-block forms. Backends implement only the backend-first methods; these
# forward to them. `f::Function` keeps them unambiguous with the untyped `f`
# slot of the backend-first methods (a backend is never a `Function`).
@inline tile_foreach(f::Function, backend::ThreadBackend, itr; kwargs...) =
    tile_foreach(backend, f, itr; kwargs...)
@inline tile_mapreduce(f::Function, op, backend::ThreadBackend, itr; kwargs...) =
    tile_mapreduce(backend, f, op, itr; kwargs...)

# Array-level forms: public face of the `_foreach_tile`/`_mapreduce_tile` tile
# drivers (abstract_haloarray.jl), so user kernels get the same dispatch as the
# package's own per-tile operations without naming the backend.
@inline tile_foreach(f::Function, u::AbstractSingleHaloArray) = _foreach_tile(f, u)
@inline tile_mapreduce(f::Function, op, u::AbstractSingleHaloArray) = _mapreduce_tile(f, op, u)

# --- chunking shared by the task-based backends (Base here, Polyester ext) -----
# At most one contiguous chunk of tiles per thread, every chunk non-empty.
@inline _single_thread(itr) = min(length(itr), Threads.nthreads()) <= 1
@inline function _chunking(itr)
    n   = length(itr)
    len = cld(n, min(n, Threads.nthreads()))
    return len, cld(n, len)
end
# Chunk `c` of an indexable `itr`; a view of a range is a range, so this
# allocates nothing for the `1:tile_count` the package passes.
@inline function _chunk(itr, c::Int, len::Int)
    lo = first(eachindex(itr)) + (c - 1) * len
    hi = min(lo + len - 1, last(eachindex(itr)))
    return view(itr, lo:hi)
end
@inline _reduce_chunk(f::F, op::OP, itr, c::Int, len::Int) where {F,OP} =
    mapreduce(f, op, _chunk(itr, c, len))

# --- Base threads (default) -------------------------------------------------------
# Chunks 2..n are spawned (held in a tuple: no vector to allocate), the calling
# thread works on chunk 1 meanwhile, then waits; reductions combine the chunk
# results in order, so a non-commutative `op` sees the tiles in order. The
# remaining allocations are the tasks themselves (≈6 per spawned task).
# `f::F, op::OP` force specialisation: `f` is only passed on here, never called.
function tile_foreach(::ThreadsBackend, f::F, itr::AbstractVector; scheduler=nothing) where {F}
    _single_thread(itr) && (foreach(f, itr); return nothing)
    len, nchunks = _chunking(itr)
    tasks = ntuple(nchunks - 1) do c
        Threads.@spawn foreach(f, _chunk(itr, c + 1, len))
    end
    foreach(f, _chunk(itr, 1, len))
    foreach(wait, tasks)
    return nothing
end
function tile_mapreduce(::ThreadsBackend, f::F, op::OP, itr::AbstractVector; scheduler=nothing) where {F,OP}
    _single_thread(itr) && return mapreduce(f, op, itr)
    len, nchunks = _chunking(itr)
    R = Base.promote_op(_reduce_chunk, F, OP, typeof(itr), Int, Int)
    tasks = ntuple(nchunks - 1) do c
        Threads.@spawn _reduce_chunk(f, op, itr, c + 1, len)
    end
    acc = _reduce_chunk(f, op, itr, 1, len)
    for t in tasks
        acc = op(acc, _fetch_as(t, R))
    end
    return acc
end
# `fetch` returns `Any`; assert the inferred chunk type when it is concrete, so
# sum/norm/dot stay type-stable (a union-typed result is combined as is).
@inline _fetch_as(t, ::Type{R}) where {R} = isconcretetype(R) ? fetch(t)::R : fetch(t)
# Tiles that cannot be indexed: collect them first.
tile_foreach(b::ThreadsBackend, f::F, itr; scheduler=nothing) where {F} =
    tile_foreach(b, f, collect(itr))
tile_mapreduce(b::ThreadsBackend, f::F, op::OP, itr; scheduler=nothing) where {F,OP} =
    tile_mapreduce(b, f, op, collect(itr))

# --- Serial -------------------------------------------------------------------
@inline tile_foreach(::SerialBackend, f, itr; scheduler=nothing) = (foreach(f, itr); nothing)
@inline tile_mapreduce(::SerialBackend, f, op, itr; scheduler=nothing) = mapreduce(f, op, itr)

# --- availability check (used at construction for a friendly early error) -----
_require_thread_backend(::ThreadBackend) = nothing
function _require_thread_backend(::OhMyThreadsBackend)
    if isnothing(Base.get_extension(@__MODULE__, :HaloArraysOhMyThreadsExt))
        throw(ArgumentError(
            "OhMyThreadsBackend requires loading OhMyThreads first (`using OhMyThreads`); " *
            "its tile_foreach/tile_mapreduce methods live in the " *
            "HaloArraysOhMyThreadsExt package extension. The default ThreadsBackend() " *
            "needs no extra package."))
    end
    return nothing
end
function _require_thread_backend(::PolyesterBackend)
    if isnothing(Base.get_extension(@__MODULE__, :HaloArraysPolyesterExt))
        throw(ArgumentError(
            "PolyesterBackend requires loading Polyester first (`using Polyester`); " *
            "its tile_foreach/tile_mapreduce methods live in the " *
            "HaloArraysPolyesterExt package extension."))
    end
    return nothing
end
