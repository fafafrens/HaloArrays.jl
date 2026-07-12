# mapreduce / any / all for HaloArray (MPI) live in mpi_support.jl

# Reduce over one or more already-resolved interior views. With a single view we
# forward straight to Base's (efficient) reduction; with two or more we reduce
# over a lazy `zip` rather than `reducer(f, op, A, B)` — Base's multi-iterator
# `mapreduce` materializes `map(f, A, B)` into a full interior-sized array
# (O(N) allocation per call). `reducer` is `mapreduce`/`mapfoldl`/`mapfoldr`.
@inline function _reduce_views(reducer::R, f::F, op::OP, views::Tuple; kws...) where {R,F,OP}
    return length(views) == 1 ? reducer(f, op, views[1]; kws...) :
                                reducer(t -> f(t...), op, zip(views...); kws...)
end

# ---- contiguous-aware interior sum-reductions (the Krylov hot path) ----------
# `interior_view` is a *strided* SubArray (halo padding separates columns), and
# Base's `mapreduce`/`dot` don't SIMD strided arrays — they fall to per-element
# cartesian iteration, ~5–9× slower than contiguous. These kernels reduce over the
# raw `parent` directly: loop the trailing cartesian indices, `@simd` the
# contiguous leading dimension. Used by `sum`/`norm`/`dot` on every backend (the
# per-rank/per-tile local reduction); MPI/threaded wrap them with Allreduce / tile
# combine. Only the `+`-accumulating ops live here — `@simd` reassociates, which
# matches `sum`/`norm`/`dot` semantics but not order-sensitive folds or max/min.
# Fast CPU path — a dense `Array` parent: @simd over the contiguous leading dim.
@inline function _interior_acc(f::F, p::Array, rng::Tuple) where {F}
    inner = rng[1]
    outer = CartesianIndices(Base.tail(rng))
    s = zero(typeof(f(zero(eltype(p)))))
    @inbounds for J in outer
        @simd for i in inner
            s += f(p[i, J])
        end
    end
    return s
end
@inline function _interior_dot(px::Array, py::Array, rng::Tuple)
    inner = rng[1]
    outer = CartesianIndices(Base.tail(rng))
    s = zero(promote_type(eltype(px), eltype(py)))
    @inbounds for J in outer
        @simd for i in inner
            s += conj(px[i, J]) * py[i, J]
        end
    end
    return s
end
# Generic fallback — GPU (CuArray/MtlArray/…) or any non-dense parent: reduce over
# the interior *view* so the array type's own (GPU) kernels run. The scalar-indexed
# @simd loop above would throw under `allowscalar(false)` or crawl on a GPUArray;
# this preserves the original, device-agnostic behaviour for those parents.
@inline _interior_acc(f::F, p::AbstractArray, rng::Tuple) where {F} =
    mapreduce(f, +, @view p[rng...])
@inline _interior_dot(px::AbstractArray, py::AbstractArray, rng::Tuple) =
    LinearAlgebra.dot(@view(px[rng...]), @view(py[rng...]))

# ---- local parts, written once over the tile drivers -------------------------
# The "local part" of every reduction — this array's own cells, reduced per tile
# and combined with `op` across tiles (`_mapreduce_tile`: single-block = one
# inline tile; threaded = tile_mapreduce over the thread backend). The MPI
# HaloArray methods (mpi_support.jl) wrap these SAME local parts in an
# Allreduce, so each reduction's local math exists in exactly one place.
_local_mapreduce(reducer::R, f::F, op::OP, arrays::Tuple; kws...) where {R,F,OP} =
    _mapreduce_tile(t -> _reduce_views(reducer, f, op,
        map(h -> interior_view(h, t), arrays); kws...), op, first(arrays))
_local_sum(f::F, u) where {F} =
    _mapreduce_tile(t -> _interior_acc(f, tile_parent(u, t), interior_range(u, t)), +, u)
_local_dot(x, y) =
    _mapreduce_tile(t -> _interior_dot(tile_parent(x, t), tile_parent(y, t), interior_range(x, t)), +, x)
_local_any(f::F, u) where {F} = _mapreduce_tile(t -> any(f, interior_view(u, t)), |, u)
_local_all(f::F, u) where {F} = _mapreduce_tile(t -> all(f, interior_view(u, t)), &, u)
_local_equal(x, y) = _mapreduce_tile(t -> interior_view(x, t) == interior_view(y, t), &, x)

# Reduce each tile (serially, with the user's kwargs), then combine the
# per-tile results with `op` across tiles. One definition per reducer covers
# LocalHaloArray and ThreadedHaloArray; the MPI HaloArray methods
# (mpi_support.jl) are more specific and Allreduce the same local part.
# `mapreduce` additionally supports `dims=`, routed to the backend-uniform
# dims-reduction below (the MPI method routes to its plan-based equivalent).
# Shared dims= kwarg validation: the canonical dims, or nothing for the scalar path.
@inline function _dims_kwarg(kws, nargs::Int)
    dims = get(kws, :dims, Colon())
    dims === Colon() && return nothing
    nargs == 1 || throw(ArgumentError(
        "`dims=` reduction over multiple halo arrays is not supported; map into one array first."))
    length(kws) == 1 || throw(ArgumentError(
        "only the `dims` keyword is supported for a halo-array dims-reduction (got $(keys(kws)))"))
    return dims
end

function Base.mapreduce(
        f::F, op::OP, halo::AbstractSingleHaloArray, etc::Vararg{AbstractSingleHaloArray}; kws...,
    ) where {F<:Function, OP}
    dims = _dims_kwarg(kws, 1 + length(etc))
    dims === nothing || return mapreduce_haloarray_dims(f, op, halo, dims)
    return _local_mapreduce(mapreduce, f, op, (halo, etc...); kws...)
end

for func in (:mapfoldl, :mapfoldr)
    @eval function Base.$func(
            f::F, op::OP, halo::AbstractSingleHaloArray, etc::Vararg{AbstractSingleHaloArray}; kws...,
        ) where {F<:Function, OP}
        # A per-tile fold with dims= would combine same-shaped per-tile arrays
        # across ALL tiles — wrong for tiles that lie along kept dimensions.
        # (A single-block LocalHaloArray keeps Base's semantics on its one view.)
        (:dims in keys(kws) && halo isa ThreadedHaloArray) && throw(ArgumentError(
            "`$($(string(func)))` with `dims=` is not supported on a tiled ThreadedHaloArray; " *
            "use `mapreduce`/`sum`/… with `dims=` (commutative ops only)."))
        return _local_mapreduce($func, f, op, (halo, etc...); kws...)
    end
end

for func in (:mapreduce, :mapfoldl, :mapfoldr)
    @eval function Base.$func(
            f::F, op::OP, z::Iterators.Zip{<:Tuple{AbstractSingleHaloArray,Vararg{AbstractSingleHaloArray}}}; kws...,
        ) where {F<:Function, OP}
        g(args...) = f(args)
        return $func(g, op, z.is...; kws...)
    end
end

Base.any(f::F, u::AbstractSingleHaloArray) where {F<:Function} = _local_any(f, u)
Base.all(f::F, u::AbstractSingleHaloArray) where {F<:Function} = _local_all(f, u)

# `==` compares the interiors (ghosts excluded, like every reduction). Without
# this method Base's generic AbstractArray `==` iterates the arrays — which is
# interior-LOCAL, so under MPI each rank would compare only its own subdomain
# and ranks could silently disagree. The HaloArray method (mpi_support.jl)
# Allreduces the same local part so every rank returns the same answer.
function Base.:(==)(x::AbstractSingleHaloArray, y::AbstractSingleHaloArray)
    size(x) == size(y) || return false
    return _local_equal(x, y)
end

function Base.:(==)(x::AbstractHaloCollection, y::AbstractHaloCollection)
    length(eachfield(x)) == length(eachfield(y)) || return false
    return all(fxy -> fxy[1] == fxy[2], zip(eachfield(x), eachfield(y)))
end

# mapreduce/mapfoldl/mapfoldr over a multi-field container reduce each field
# across the inputs, then reduce the per-field results. One definition covers any
# AbstractHaloCollection (MultiHaloArray + ArrayOfHaloArray) via `eachfield`.
# Single collection (the common case: sum/maximum/minimum/norm/…): fold over
# the fields directly. No intermediate results container — 0-alloc for BOTH the
# tuple-backed MultiHaloArray and the array-backed ArrayOfHaloArray (unlike a
# `map` over `eachfield`, which materializes a Vector for the array case). The
# field-combine op is `op` itself; `kws` (e.g. `init`) apply at that combine.
# `mapreduce` additionally supports `dims=`: each field is reduced along `dims`
# and the same collection kind is rebuilt around the reduced fields.
function Base.mapreduce(f::F, op::OP, c::AbstractHaloCollection; kws...) where {F<:Function, OP}
    dims = _dims_kwarg(kws, 1)
    dims === nothing || return mapreduce_haloarray_dims(f, op, c, dims)
    return mapreduce(field -> mapreduce(f, op, field), op, eachfield(c); kws...)
end

for func in (:mapfoldl, :mapfoldr)
    @eval Base.$func(f::F, op::OP, halo::AbstractHaloCollection; kws...) where {F<:Function, OP} =
        $func(field -> $func(f, op, field), op, eachfield(halo); kws...)
end

for func in (:mapreduce, :mapfoldl, :mapfoldr)

    # Two or more collections: combine the i-th field across inputs, then reduce.
    # `map` over `eachfield` keeps the tuple case 0-alloc; the array case
    # materializes O(#fields) (a `zip` of tuples would be type-unstable).
    @eval function Base.$func(
            f::F, op::OP, halo::AbstractHaloCollection, etc::Vararg{AbstractHaloCollection}; kws...,
        ) where {F<:Function, OP}
        per_field_results = map((fields...) -> $func(f, op, fields...),
                                map(eachfield, (halo, etc...))...)
        return reduce(op, per_field_results; kws...)
    end

    @eval function Base.$func(
            f::F, op::OP, z::Iterators.Zip{<:Tuple{AbstractHaloCollection,Vararg{AbstractHaloCollection}}}; kws...,
        ) where {F<:Function, OP}
        g(args...) = f(args)
        $func(g, op, z.is...; kws...)
    end
end


# Field-wise short-circuit all/any for any collection kind (named or indexed):
# reduce over fields, each field reducing over its own interior.
Base.all(f::F, c::FieldCollection) where {F<:Function} =
    all(field -> all(f, field), _fields(c))
Base.any(f::F, c::FieldCollection) where {F<:Function} =
    any(field -> any(f, field), _fields(c))

for func in (:mapreduce, :mapfoldl, :mapfoldr)
    @eval function Base.$func(
            f::F, op::OP, halo::MaybeHaloArray, etc::Vararg{MaybeHaloArray}; kws...,
        ) where {F<:Function, OP}
        all(is_active, (halo, etc...)) ||
            throw(ErrorException("MaybeHaloArray: attempt to reduce inactive value"))
        return $func(f, op, getdata(halo), getdata.(etc)...; kws...)
    end

    @eval function Base.$func(
            f::F, op::OP, z::Iterators.Zip{<:Tuple{MaybeHaloArray,Vararg{MaybeHaloArray}}}; kws...,
        ) where {F<:Function, OP}
        g(args...) = f(args)
        $func(g, op, z.is...; kws...)
    end
end

function Base.all(f::F, halo::MaybeHaloArray) where {F<:Function}
    is_active(halo) || throw(ErrorException("MaybeHaloArray: attempt to reduce inactive value"))
    return all(f, getdata(halo))
end

function Base.any(f::F, halo::MaybeHaloArray) where {F<:Function}
    is_active(halo) || throw(ErrorException("MaybeHaloArray: attempt to reduce inactive value"))
    return any(f, getdata(halo))
end

Base.sum(halo::AbstractHaloArray) = mapreduce(identity, +, halo)
Base.sum(f::F, halo::AbstractHaloArray) where {F<:Function} = mapreduce(f, +, halo)
Base.maximum(halo::AbstractHaloArray) = mapreduce(identity, max, halo)
Base.minimum(halo::AbstractHaloArray) = mapreduce(identity, min, halo)

# Fast `sum` (no-arg): contiguous-aware interior reduction (see `_interior_acc`).
# `sum(f, ...)` and max/min keep the generic mapreduce path above.
Base.sum(u::AbstractSingleHaloArray) = _local_sum(identity, u)

# dot, norm, and the in-place BLAS-1 ops (rmul!/lmul!/axpy!/axpby!) — all built
# on the mapreduce/broadcast above — live in vector_space.jl.

# ---- dims= reductions, backend-preserving ---------------------------------
#
# Every backend returns its own kind of reduced array with the reduced
# dimensions DROPPED, carrying the kept dimensions' halo width and boundary
# conditions: LocalHaloArray → LocalHaloArray, ThreadedHaloArray →
# ThreadedHaloArray (kept-dims tile layout, same thread backend), HaloArray →
# MaybeHaloArray (its result lives only on the coordinate-0 slice of the
# topology — the one backend where "may be absent" is real). Serial results
# are plain arrays, always active; `is_active`/`interior_view`/`free!` behave
# uniformly on all of them, so backend-generic code needs no branches. `f`
# must preserve the element type (the distributed backend requires it; serial
# backends match). The `dims` normalization and op unwrapping live in
# mpi_support.jl; the HaloArray (MPI) oneshot method lives there too, built on
# DimReductionPlan.

# Geometry of the reduced array: kept dims' sizes and boundary conditions.
@inline _kept_dims(::Val{N}, dims_n) where {N} =
    Tuple(d for d in 1:N if !(d in dims_n))

"""
    DimReductionPlan(u, dims) -> plan

Precompute everything a dimensional reduction of `u` along `dims` needs, so
each subsequent [`reduce!`](@ref) call reuses it — the backend-generic way to
run the same reduction repeatedly (e.g. saving a profile every few steps).
Works on every backend; the plan is geometry-only, so `f` and `op` are
supplied per [`reduce!`](@ref) call and one plan serves `sum`, `maximum`, …
over any array sharing `u`'s geometry. The plan's output has its build-time
element type baked in — `eltype(u)` unless overridden with the
`output_eltype` keyword — so a reduction whose result type differs (e.g. `+`
on `Bool`) needs the keyword, or the one-shot forms
(`sum(u; dims=…)`/[`mapreduce_haloarray_dims`](@ref)), which are transient
plans built with the promoted element type and so follow Base's promotion.

- `LocalHaloArray` / `ThreadedHaloArray`: holds the preallocated reduced array
  (for threaded, tiled by the kept-dims layout); [`free!`](@ref) is a no-op.
- `HaloArray` (MPI): additionally holds the slice sub-communicators (built
  once via `MPI.Cart_sub`), so each `reduce!` costs a single `MPI.Reduce`.
  Construction is collective, and [`free!`](@ref) releases the communicators
  (unreleased ones are reclaimed at `MPI.Finalize`).

# Example
```julia
plan = DimReductionPlan(u, 2)          # once, outside the loop — any backend
for step in 1:nsteps
    step!(u)
    profile = reduce!(plan, identity, +, u)   # overwrites the plan's output
    is_active(profile) && save(profile)
end
free!(plan)
```
"""
abstract type DimReductionPlan end

# Serial (Local/Threaded) plan: no communicators — the canonical dims, the
# source geometry for validation, the preallocated output, and (threaded only)
# the geometry-derived grouping of source tiles per output tile, computed once
# here so reduce! does no coordinate arithmetic.
struct SerialDimReductionPlan{K,G,O,S} <: DimReductionPlan
    dims_to_remove::NTuple{K,Int}
    source_geometry::G
    output::O
    source_tiles::S     # nothing (Local) or Vector{Vector{Int}} indexed by output tile
end

function DimReductionPlan(u::LocalHaloArray{T,N,A,Halo}, dims;
        output_eltype=T) where {T,N,A,Halo}
    dims_n = _normalize_reduce_dims(Val(N), dims)
    keep   = _kept_dims(Val(N), dims_n)
    M      = length(keep)
    # Storage allocated like the source's (`similar` on the parent), so a
    # GPU-backed array gets a device-resident reduced output.
    data = fill!(similar(parent(u), output_eltype,
        ntuple(i -> interior_size(u)[keep[i]] + 2Halo, Val(M))), zero(output_eltype))
    out  = LocalHaloArray(data, Halo, ntuple(i -> u.boundary_condition[keep[i]], Val(M)))
    SerialDimReductionPlan(dims_n, interior_size(u), out, nothing)
end

function DimReductionPlan(u::ThreadedHaloArray{T,N,A,Halo}, dims;
        output_eltype=T) where {T,N,A,Halo}
    dims_n = _normalize_reduce_dims(Val(N), dims)
    keep   = _kept_dims(Val(N), dims_n)
    M      = length(keep)
    ts     = tile_size(u)
    layout = u.topology.dims
    out = _on_device_of(tile_parent(u, 1),
        ThreadedHaloArray(output_eltype, ntuple(i -> ts[keep[i]], Val(M)), Halo;
            dims=ntuple(i -> layout[keep[i]], Val(M)),
            boundary_condition=ntuple(i -> u.boundary_condition[keep[i]], Val(M)),
            thread_backend=thread_backend(u)))
    # Group the source tiles by kept coordinate = output tile (ascending id, so
    # the op-combine order is deterministic).
    out_tiles = LinearIndices(out.topology.dims)
    source_tiles = [Int[] for _ in 1:tile_count(out)]
    for t in 1:tile_count(u)
        c = tile_coordinates(u, t)
        push!(source_tiles[out_tiles[ntuple(i -> c[keep[i]], Val(M))...]], t)
    end
    SerialDimReductionPlan(dims_n, (ts, layout), out, source_tiles)
end

# A plan's output element type is fixed at build time (before f/op are known),
# so a promoting reduction (e.g. + on Bool, where Bool+Bool::Int) cannot land
# in it — fail with the story instead of an InexactError/MPI type mismatch.
@inline function _check_plan_eltype(reduced_eltype::Type, out_eltype::Type)
    reduced_eltype === out_eltype || throw(ArgumentError(
        "this reduction promotes $out_eltype to $reduced_eltype, but a " *
        "DimReductionPlan's output is typed at construction; use the one-shot " *
        "form (`sum(u; dims=…)`/`mapreduce_haloarray_dims`), which promotes " *
        "like Base, or convert the array first."))
    return nothing
end

function reduce!(plan::SerialDimReductionPlan, f::F, op::OP, u::LocalHaloArray) where {F,OP}
    interior_size(u) == plan.source_geometry ||
        throw(DimensionMismatch("array geometry $(interior_size(u)) does not match the plan's $(plan.source_geometry)"))
    red = mapreduce(f, _normalize_reduction_op(op), interior_view(u); dims=plan.dims_to_remove)
    _check_plan_eltype(eltype(red), eltype(plan.output))
    # Broadcast assignment (GPU-safe; only singleton dims differ).
    iv = interior_view(plan.output)
    iv .= reshape(red, size(iv))
    return plan.output
end

function reduce!(plan::SerialDimReductionPlan, f::F, op::OP, u::ThreadedHaloArray) where {F,OP}
    (tile_size(u), u.topology.dims) == plan.source_geometry ||
        throw(DimensionMismatch("array tiling $((tile_size(u), u.topology.dims)) does not match the plan's $(plan.source_geometry)"))
    _assemble_reduced_tiles!(plan.output, f, _normalize_reduction_op(op), u,
        plan.dims_to_remove, plan.source_tiles)
    return plan.output
end

# The eltype a reduction produces, predicted from types alone (what Base's
# broadcast/reduction machinery uses): Bool sums count in Int, an Int field
# divided by a scalar sums in Float64, … Falls back to T when inference can't
# give a concrete answer — then reduce!'s runtime guard still has the last word.
function _reduced_eltype(f::F, op::OP, ::Type{T}) where {F,OP,T}
    S = Base.promote_op(f, T)
    R = Base.promote_op(op, S, S)
    return isconcretetype(R) ? R : T
end

"""
    mapreduce_haloarray_dims(f, op, u, dims)

Map `f` over the interior cells of `u` and reduce with `op` along the spatial
dimensions in `dims`, returning a reduced array of the **same backend** with
those dimensions **dropped** (kept dimensions retain their halo width and
boundary conditions). Works on every backend and on field collections —
equivalent to the `dims=` keyword forms (`sum(u; dims=…)`,
`maximum(u; dims=…)`, …):

- `LocalHaloArray` → a reduced `LocalHaloArray`.
- `ThreadedHaloArray` → a reduced `ThreadedHaloArray` whose tile layout is the
  original layout with the reduced dimensions dropped, on the same thread
  backend — ready for further threaded work.
- `HaloArray` (MPI) → a `MaybeHaloArray`: collective across the topology; the
  result lives on the coordinate-0 slice of the reduced dimensions (inactive
  elsewhere) and **owns the sub-communicator** its topology lives on —
  [`free!`](@ref) it when done to keep communicator use bounded when reducing
  in a loop (otherwise it is reclaimed at `MPI.Finalize`).
- `MultiHaloArray` / `ArrayOfHaloArray` → every field reduced, same collection
  kind rebuilt around the reduced fields (`MaybeHaloArray`-wrapped only when
  the fields are MPI-backed).

Implemented as a transient [`DimReductionPlan`](@ref) — built with the
promoted output element type (so `Bool` sums count in `Int`, like Base), used
for one [`reduce!`](@ref), and released. For a reduction that runs repeatedly
over the same array shape, build the plan once instead. `is_active`,
`interior_view`, and [`free!`](@ref) behave uniformly on every return kind
(serial results are always active; `free!` is a no-op on them), so
backend-generic code needs no branches.
"""
function mapreduce_haloarray_dims(f::F, op::OP, u::AbstractSingleHaloArray, dims) where {F,OP}
    op_n = _normalize_reduction_op(op)
    plan = DimReductionPlan(u, dims; output_eltype=_reduced_eltype(f, op_n, eltype(u)))
    out  = reduce!(plan, f, op_n, u)
    _release_transient!(plan)
    return out
end

_release_transient!(::SerialDimReductionPlan) = nothing

# Rebuild `x`'s storage on the device `proto` lives on (no-op for host arrays):
# a reduced output must live where its source lives. The typename-wrapper is
# the base array constructor Adapt targets (JLArray{Float64,2} → JLArray).
_on_device_of(::Array, x) = x
_on_device_of(proto::AbstractArray, x) = Adapt.adapt(Base.typename(typeof(proto)).wrapper, x)

# Shared threaded assembly: each OUTPUT tile is the `op`-combination of its
# precomputed source tiles (grouped by kept coordinate at plan construction),
# reduced along `dims_n`. (Combining every tile with `op`, as the generic tile
# driver would, silently mixes tiles that lie along kept dimensions.) Runs
# through `out`'s tile driver — parallel on the array's thread backend, and
# race-free since each task writes only its own output tile.
function _assemble_reduced_tiles!(out::ThreadedHaloArray, f::F, op::OP,
        u::ThreadedHaloArray, dims_n, source_tiles::Vector{Vector{Int}}) where {F,OP}
    _foreach_tile(out) do ot
        slab(t) = dropdims(mapreduce(f, op, interior_view(u, t); dims=dims_n); dims=dims_n)
        tiles = source_tiles[ot]
        dest  = interior_view(out, ot)
        first = slab(tiles[1])
        _check_plan_eltype(eltype(first), eltype(out))
        # The first slab ASSIGNS (a generic op has no known neutral element to
        # pre-fill with — combining into the zeroed tile would be wrong for
        # *, max, …); the rest combine. Broadcasts throughout: GPU-safe.
        dest .= first
        for t in Iterators.drop(tiles, 1)
            dest .= op.(dest, slab(t))
        end
    end
    return out
end

# free! on a serial plan: nothing to release (no communicators); the plan
# stays usable — releasing is meaningful only on the MPI backend.
free!(plan::SerialDimReductionPlan) = plan

# Collections: reduce each field along `dims` and rebuild the same collection
# kind around the reduced fields (unwrapped — a distributed field's topology
# still carries its active flag). Whether the rebuilt collection needs the
# MaybeHaloArray wrapper is the backend's decision (MPI only); its active
# state is read from the fields' topologies by the 1-arg constructor
# (is_active(collection) = all(is_active, fields)), not tracked by hand.
function mapreduce_haloarray_dims(f::F, op::OP, c::AbstractHaloCollection, dims) where {F,OP}
    reduced = _map_fields(field -> getdata(mapreduce_haloarray_dims(f, op, field, dims)), c)
    return _wrap_reduced(halo_backend(c), reduced)
end

# Trait-dispatched wrap: only the distributed backend's results may be absent
# on a rank, so only it wraps in MaybeHaloArray (activity read from the
# fields' topologies by the 1-arg constructor).
_wrap_reduced(::MPIHaloBackend, reduced)      = MaybeHaloArray(reduced)
_wrap_reduced(::AbstractHaloBackend, reduced) = reduced
