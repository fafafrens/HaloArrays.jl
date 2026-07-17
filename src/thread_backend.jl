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
which is *where* data lives). Built-ins: [`OhMyThreadsBackend`](@ref) (default),
[`SerialBackend`](@ref), [`PolyesterBackend`](@ref). Add your own by defining
[`tile_foreach`](@ref) and [`tile_mapreduce`](@ref) for a new subtype.
"""
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
    tile_foreach(f, backend, itr; scheduler=:dynamic)
    tile_foreach(f, u::AbstractSingleHaloArray)

Apply `f` to every element of `itr` in parallel, according to `backend`. The
`scheduler` hint is honoured by [`OhMyThreadsBackend`](@ref) and ignored by the
others.

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
    tile_mapreduce(backend, f, op, itr; scheduler=:dynamic)
    tile_mapreduce(f, op, backend, itr; scheduler=:dynamic)
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
