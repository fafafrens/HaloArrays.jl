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
# Accumulator zero of the type Base's `sum` reduces in — the `add_sum`-promoted
# type, NOT the element type. This widens `Bool`/`Int8`/… to `Int` so a sum of
# narrow integers doesn't overflow (matching `sum(::Array)`), while `Float64`
# stays `Float64` (the hot path is byte-for-byte unchanged). `norm` sums `abs2`
# through the same path, so it widens too.
@inline _acc_zero(f::F, ::Type{T}) where {F,T} =
    (S = typeof(f(zero(T))); zero(Base.promote_op(Base.add_sum, S, S)))

# Element-wise ‖·‖² and ⟨·,·⟩ so `norm`/`dot` work for vector-valued cells (e.g.
# `SVector` fields) exactly like Base, which recurses `abs2`/`*` into the element.
# For a scalar element these inline to `abs2` / `conj(x)*y`, so the numeric hot
# path (and its `@simd`) is byte-for-byte unchanged; a static vector falls to
# `sum(abs2, ·)` / `dot(·,·)`, returning the same scalar Base does.
@inline _elt_abs2(x::Number) = abs2(x)
@inline _elt_abs2(x) = sum(abs2, x)
@inline _elt_dot(x::Number, y::Number) = conj(x) * y
@inline _elt_dot(x, y) = LinearAlgebra.dot(x, y)
# Scalar accumulator zero of ‖·‖² / ⟨·,·⟩ for a (possibly vector-valued) element
# type — `Float64` for an `SVector{N,Float64}`, unchanged for numeric elements.
@inline _sqnorm_zero(::Type{T}) where {T} = zero(Base.promote_op(_elt_abs2, T))
@inline _dot_zero(::Type{X}, ::Type{Y}) where {X,Y} = zero(Base.promote_op(_elt_dot, X, Y))

# Fast CPU path — a dense `Array` parent: @simd over the contiguous leading dim.
@inline function _interior_acc(f::F, p::Array, rng::Tuple) where {F}
    inner = rng[1]
    outer = CartesianIndices(Base.tail(rng))
    s = _acc_zero(f, eltype(p))
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
    s = _dot_zero(eltype(px), eltype(py))   # scalar; matches Base's `dot` accumulator
    @inbounds for J in outer
        @simd for i in inner
            s += _elt_dot(px[i, J], py[i, J])
        end
    end
    return s
end
# Generic fallback — GPU (CuArray/MtlArray/…) or any non-dense parent: reduce over
# the interior *view* so the array type's own (GPU) kernels run. The scalar-indexed
# @simd loop above would throw under `allowscalar(false)` or crawl on a GPUArray;
# this preserves the original, device-agnostic behaviour for those parents.
@inline _interior_acc(f::F, p::AbstractArray, rng::Tuple) where {F} =
    sum(f, @view p[rng...])   # `sum` widens narrow ints like Base (not `mapreduce(f,+)`)
@inline _interior_dot(px::AbstractArray, py::AbstractArray, rng::Tuple) =
    LinearAlgebra.dot(@view(px[rng...]), @view(py[rng...]))

# ---- local parts, written once over the tile drivers -------------------------
# The "local part" of every reduction — this array's own cells, reduced per tile
# and combined with `op` across tiles (`_mapreduce_tile`: single-block = one
# inline tile; threaded = tile_mapreduce over the thread backend). The MPI
# HaloArray methods (mpi_support.jl) wrap these SAME local parts in an
# Allreduce, so each reduction's local math exists in exactly one place.
# `kws` here is only ever `init` (dims is routed away earlier). For the
# COMMUTATIVE `mapreduce` path the caller passes NO kws and seeds with `init`
# once via `_apply_init` — forwarding it into every tile counted it per tile.
# The order-sensitive folds still forward it (it must seed one end of the fold,
# which is only well-defined on a single tile; see the fold methods).
_local_mapreduce(reducer::R, f::F, op::OP, arrays::Tuple; kws...) where {R,F,OP} =
    _mapreduce_tile(t -> _reduce_views(reducer, f, op,
        map(h -> interior_view(h, t), arrays); kws...), op, first(arrays))

# Seed a commutative reduction with `init` exactly ONCE, after the tiles (and,
# on MPI, the ranks) have been combined. `haskey` is compile-time constant (the
# kw names are in the Pairs type), so this is type-stable and free without `init`.
@inline _apply_init(op::OP, r, kws) where {OP} =
    haskey(kws, :init) ? op(kws[:init], r) : r
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
    r = _local_mapreduce(mapreduce, f, op, (halo, etc...))   # per-tile, no init
    return _apply_init(op, r, kws)                            # seed once (commutative)
end

for func in (:mapfoldl, :mapfoldr)
    @eval function Base.$func(
            f::F, op::OP, halo::AbstractSingleHaloArray, etc::Vararg{AbstractSingleHaloArray}; kws...,
        ) where {F<:Function, OP}
        # `dims=` is rejected for order-sensitive folds on every halo array: a
        # per-tile / per-rank reduction reorders the fold (and on a tiled array
        # would combine same-shaped per-tile results across ALL tiles, mixing
        # tiles that lie along kept dimensions). Use `mapreduce`/`sum`/… with
        # `dims=` (commutative ops only). Guard here rather than letting the
        # call fall through to Base's dims-less `$func`, which errors obscurely.
        :dims in keys(kws) && throw(ArgumentError(
            "`$($(string(func)))` with `dims=` is not supported on a halo array; " *
            "use `mapreduce`/`sum`/… with `dims=` (commutative ops only)."))
        # Folds forward `init` into the tile fold (it seeds one end): exact Base
        # on a single tile; on a tiled array the cross-tile order is unspecified
        # regardless — use `mapreduce` for a commutative reduction.
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
# Const-foldable tuple filter (recursive, no Vector/generator): with a literal
# `dims` the whole reduction infers a concrete NTuple; a `Tuple(gen if …)` would
# type as `Tuple{Vararg}` and block that. `pred::P` forces specialization.
@inline _tfilter(::P, ::Tuple{}) where {P} = ()
@inline _tfilter(pred::P, t::Tuple) where {P} =
    pred(first(t)) ? (first(t), _tfilter(pred, Base.tail(t))...) : _tfilter(pred, Base.tail(t))

# Sorted, duplicate-free Int tuple of dims — tuple-based and const-foldable
# (the Vector `sort!∘unique!∘collect` stays only as the fallback for a tuple
# that is not already strictly increasing — a rare, non-literal case).
@inline _canonical_dims(d::Integer) = (Int(d),)
@inline _canonical_dims(dims::Tuple) =
    _is_increasing(map(Int, dims)) ? map(Int, dims) : Tuple(sort!(unique!(collect(Int, dims))))
@inline _canonical_dims(dims) = Tuple(sort!(unique!(collect(Int, dims))))
@inline _is_increasing(::Tuple{}) = true
@inline _is_increasing(::Tuple{Any}) = true
@inline _is_increasing(t::Tuple) = (first(t) < t[2]) && _is_increasing(Base.tail(t))

# The kept dims are exactly `N - K` of them (`K = |dims_n|`, known from the
# tuple type), so build them with `ntuple` — a type-known length — instead of a
# value-dependent filter (whose length the compiler can't prove, forcing a
# `Union` and an `Any` result). `dims_n` is sorted ascending (`_canonical_dims`),
# so the i-th kept dim is `i` shifted up past each removed dim at or below it.
# This makes the reduction infer a concrete result even for a runtime `dims::Int`.
@inline _kept_dims(::Val{N}, dims_n::NTuple{K,Int}) where {N,K} =
    ntuple(Val(N - K)) do i
        d = i
        for r in dims_n
            d += (r <= d)
        end
        d
    end

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
    # `map` over `keep` (an NTuple{N-K}) keeps the length from its type — no
    # `Val(length(keep))`, which would be a value-level (unstable) construction.
    # Storage allocated like the source's (`similar` on the parent), so a
    # GPU-backed array gets a device-resident reduced output.
    data = fill!(similar(parent(u), output_eltype,
        map(k -> interior_size(u)[k] + 2Halo, keep)), zero(output_eltype))
    out  = LocalHaloArray(data, Halo, map(k -> u.boundary_condition[k], keep))
    SerialDimReductionPlan(dims_n, interior_size(u), out, nothing)
end

function DimReductionPlan(u::ThreadedHaloArray{T,N,A,Halo}, dims;
        output_eltype=T) where {T,N,A,Halo}
    dims_n = _normalize_reduce_dims(Val(N), dims)
    keep   = _kept_dims(Val(N), dims_n)     # NTuple{N-K}; `map` over it preserves length
    ts     = tile_size(u)
    layout = u.topology.dims
    out = _on_device_of(tile_parent(u, 1),
        ThreadedHaloArray(output_eltype, map(k -> ts[k], keep), Halo;
            dims=map(k -> layout[k], keep),
            boundary_condition=map(k -> u.boundary_condition[k], keep),
            thread_backend=thread_backend(u)))
    # Group the source tiles by kept coordinate = output tile (ascending id, so
    # the op-combine order is deterministic).
    out_tiles = LinearIndices(out.topology.dims)
    source_tiles = [Int[] for _ in 1:tile_count(out)]
    for t in 1:tile_count(u)
        c = tile_coordinates(u, t)
        push!(source_tiles[out_tiles[map(k -> c[k], keep)...]], t)
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
- `MultiHaloArray` / `ArrayOfHaloArray` → `dims` is in **collection**
  coordinates: field axes come first (`1:F`), then the shared spatial axes
  (`F+1:D`). Field axes reduce **locally** (an elementwise fold across fields —
  no communication, always a bare result), collapsing every field axis into one
  `HaloArray` (`MultiHaloArray` drops the field names) or a partial set into a
  smaller collection. Spatial axes reduce per field as above. The result is
  `MaybeHaloArray`-wrapped (outermost) only when a spatial axis was reduced on
  MPI; a pure field reduction is never wrapped.

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

# ---- collection dims reductions: classify field vs spatial axes ----------
#
# A collection's axes are (field axes 1..F, then spatial axes F+1..D). `dims`
# is given in these collection coordinates. The two axis kinds reduce with
# different machinery:
#   • FIELD axes reduce LOCALLY — every rank holds all its fields (they share a
#     topology), so it is an elementwise fold across fields: no communication,
#     no plan, and the result is a bare array/collection (never MaybeHaloArray).
#     Collapsing every field axis yields one HaloArray (MultiHaloArray drops the
#     field names); a partial reduction yields a smaller collection.
#   • SPATIAL axes reduce through the per-field array machinery (a transient
#     DimReductionPlan each), MaybeHaloArray-wrapped on MPI.
# A mixed reduction folds fields first, then spatial-reduces the result; the
# MaybeHaloArray, when present, is always outermost (a collection cannot hold
# Maybe fields). `op` must be commutative — the field and spatial passes reduce
# in separate phases — which the `dims=` reductions already require.

function _classify_collection_dims(c::AbstractHaloCollection, dims)
    D = ndims(c); S = _spatial_ndims(c); Fax = D - S
    dn = _canonical_dims(dims)
    all(d -> 1 <= d <= D, dn) ||
        throw(ArgumentError("dims $dims out of range for a $D-dimensional collection"))
    isempty(dn) && throw(ArgumentError("dims must select at least one dimension"))
    fdims = _tfilter(d -> d <= Fax, dn)
    sdims = map(d -> d - Fax, _tfilter(d -> d > Fax, dn))   # shifted to field-local coords
    (length(fdims) == Fax && length(sdims) == S) && throw(ArgumentError(
        "Reducing every axis of a collection to a scalar is not supported; use `sum(c)` etc."))
    return fdims, sdims
end

# The collection one-shot is a transient CollectionDimReductionPlan (built with
# the promoted output eltype, one reduce!, released) — mirroring the array
# one-shot, so `sum(c; dims=…)` and a hoisted plan share one code path.
function mapreduce_haloarray_dims(f::F, op::OP, c::AbstractHaloCollection, dims) where {F,OP}
    op_n = _normalize_reduction_op(op)
    plan = DimReductionPlan(c, dims; output_eltype=_reduced_eltype(f, op_n, eltype(c)))
    out  = reduce!(plan, f, op_n, c)
    _release_transient!(plan)
    return out
end

# Trait-dispatched wrap: only the distributed backend's results may be absent
# on a rank, so only it wraps in MaybeHaloArray.
_wrap_reduced(::MPIHaloBackend, reduced)      = MaybeHaloArray(reduced)
_wrap_reduced(::AbstractHaloBackend, reduced) = reduced

# ---- CollectionDimReductionPlan ------------------------------------------
# Classifies `dims` into field/spatial axes once, and — for the spatial axes —
# holds one reused per-field array DimReductionPlan (which owns the expensive
# MPI sub-communicators). `reduce!` folds the field axes locally each call and
# drives the reused plans for the spatial axes, so a hoisted collection plan
# rebuilds no communicators. The field fold and the small result wrapper are
# rebuilt per call (both cheap and local); the communicator reuse is the win.
struct CollectionDimReductionPlan{FD,SD,SP} <: DimReductionPlan
    fdims::FD          # field axes (collection coords), may be ()
    sdims::SD          # spatial axes (field-local coords), may be ()
    spatial_plans::SP  # () for a pure-field reduction; else one array plan per (intermediate) field
end

@inline _source_fields(x::AbstractSingleHaloArray) = (x,)
@inline _source_fields(c::AbstractHaloCollection)  = Tuple(_fields(c))

function DimReductionPlan(c::AbstractHaloCollection, dims; output_eltype=nothing)
    fdims, sdims = _classify_collection_dims(c, dims)
    isempty(sdims) && return CollectionDimReductionPlan(fdims, sdims, ())  # pure field: no plans
    # The fields the spatial plans reduce: the source fields (pure spatial) or a
    # geometry template of the field-folded intermediate (values irrelevant — a
    # fresh fold at each reduce! shares the same topology, `similar` reusing it).
    fields = isempty(fdims) ? _source_fields(c) :
             _source_fields(_reduce_field_axes(identity, +, c, fdims))
    splans = map(fields) do fld
        DimReductionPlan(fld, sdims; output_eltype = output_eltype === nothing ? eltype(fld) : output_eltype)
    end
    return CollectionDimReductionPlan(fdims, sdims, splans)
end

"""
    reduce!(plan::CollectionDimReductionPlan, f, op, c) -> reduced array/collection

Reduce collection `c` along the plan's dimensions. Field axes fold locally each
call; spatial axes reuse the plan's per-field [`DimReductionPlan`](@ref)s (no
communicator rebuild). Returns a bare array/collection, or a `MaybeHaloArray`
around one when spatial axes were reduced on MPI — the same shape the one-shot
[`mapreduce_haloarray_dims`](@ref) produces. `op` must be commutative.
"""
function reduce!(plan::CollectionDimReductionPlan, f::F, op::OP, c::AbstractHaloCollection) where {F,OP}
    op_n = _normalize_reduction_op(op)
    isempty(plan.spatial_plans) && return _reduce_field_axes(f, op_n, c, plan.fdims)  # pure field (bare)
    fs, src = isempty(plan.fdims) ? (f, c) : (identity, _reduce_field_axes(f, op_n, c, plan.fdims))
    srcfields = _source_fields(src)
    length(srcfields) == length(plan.spatial_plans) ||
        throw(DimensionMismatch("collection field layout does not match the plan"))
    reduced = map((sp, fld) -> getdata(reduce!(sp, fs, op_n, fld)), plan.spatial_plans, srcfields)
    return _rewrap_reduced(src, reduced)
end

# Single folded field spatially reduced → wrap that one field; a (smaller)
# collection → rebuild the same kind around the reduced fields. Maybe outermost.
_rewrap_reduced(::AbstractSingleHaloArray, reduced) =
    _wrap_reduced(halo_backend(reduced[1]), reduced[1])
_rewrap_reduced(src::AbstractHaloCollection, reduced) =
    _wrap_reduced(halo_backend(src), _rebuild_like(src, reduced))

_rebuild_like(c::MultiHaloArray, fields) =
    MultiHaloArray(NamedTuple{keys(getfield(c, :arrays))}(Tuple(fields)))
_rebuild_like(c::ArrayOfHaloArray, fields) =
    ArrayOfHaloArray(reshape(collect(fields), field_shape(c)))

free!(plan::CollectionDimReductionPlan) = (foreach(free!, plan.spatial_plans); plan)
_release_transient!(plan::CollectionDimReductionPlan) =
    (foreach(_release_transient!, plan.spatial_plans); nothing)

# ---- local field-axis fold ------------------------------------------------
# Fold an ordered set of same-geometry fields into one fresh field, elementwise:
# `result[cell] = mapreduce(f, op, (field₁[cell], …, fieldₖ[cell]))`. One
# `map!` per tile over the interior views (backend-uniform — `interior_view(x, t)`
# ignores the tile id on single-block backends, selects the tile on a
# ThreadedHaloArray — and GPU-safe, `mapreduce` folding the per-cell tuple).
function _fold_fields(f::F, op::OP, S_elt::Type, fields) where {F,OP}
    fs  = Tuple(fields)                       # ≥ 1 field
    acc = similar(first(fs), S_elt)
    combine(xs::Vararg) = mapreduce(f, op, xs)
    # Per-tile and independent (each tile writes only its own cells), so it runs
    # through the same tile driver as the rest of the reductions — parallel on a
    # ThreadedHaloArray's thread backend, inline on single-block backends.
    _foreach_tile(acc) do t
        map!(combine, interior_view(acc, t), map(fld -> interior_view(fld, t), fs)...)
    end
    return acc
end

# MultiHaloArray has one field axis, so a field reduction always collapses every
# named field into a single (name-free) HaloArray.
_reduce_field_axes(f::F, op::OP, c::MultiHaloArray, ::Any) where {F,OP} =
    _fold_fields(f, op, _reduced_eltype(f, op, eltype(c)), eachfield(c))

# ArrayOfHaloArray field axes may be multi-dimensional: fold along `fdims`,
# each kept-index group becoming one field. All field axes consumed → a single
# HaloArray; a partial reduction → a smaller ArrayOfHaloArray.
function _reduce_field_axes(f::F, op::OP, c::ArrayOfHaloArray, fdims) where {F,OP}
    container = getfield(c, :arrays)
    S_elt = _reduced_eltype(f, op, eltype(c))
    Fax   = ndims(container)
    kept  = Tuple(d for d in 1:Fax if !(d in fdims))
    redshape  = ntuple(i -> size(container, fdims[i]), length(fdims))
    keptshape = ntuple(i -> size(container, kept[i]), length(kept))
    fibre(ok) = _fold_fields(f, op, S_elt,
        (container[_weave_index(fdims, kept, Fax, rk, ok)...] for rk in CartesianIndices(redshape)))
    isempty(kept) && return fibre(CartesianIndex())      # every field axis → one field
    outfields = [fibre(ok) for ok in CartesianIndices(keptshape)]
    return ArrayOfHaloArray(reshape(outfields, keptshape))
end

# Reconstruct a full container index from the reduced-axis index `rk` (over
# `fdims`) and the kept-axis index `ok` (over `kept`).
@inline function _weave_index(fdims, kept, Fax::Int, rk, ok)
    ntuple(Fax) do d
        j = findfirst(==(d), fdims)
        j === nothing ? Tuple(ok)[findfirst(==(d), kept)::Int] : Tuple(rk)[j]
    end
end
