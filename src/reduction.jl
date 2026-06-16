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

for func in (:mapreduce, :mapfoldl, :mapfoldr)
    @eval function Base.$func(
            f::F, op::OP, halo::LocalHaloArray, etc::Vararg{LocalHaloArray}; kws...,
        ) where {F<:Function, OP}
        return _reduce_views($func, f, op, map(interior_view, (halo, etc...)); kws...)
    end

    @eval function Base.$func(
            f::F, op::OP, z::Iterators.Zip{<:Tuple{LocalHaloArray,Vararg{LocalHaloArray}}}; kws...,
        ) where {F<:Function, OP}
        g(args...) = f(args)
        return $func(g, op, z.is...; kws...)
    end

    @eval function Base.$func(
            f::F, op::OP, halo::ThreadedHaloArray, etc::Vararg{ThreadedHaloArray}; kws...,
        ) where {F<:Function, OP}
        # Reduce each tile (serially, with the user's kwargs), then combine the
        # per-tile results with `op` across tiles via the array's thread backend.
        per_tile(tile_id) = _reduce_views($func, f, op, map(h -> interior_view(h, tile_id), (halo, etc...)); kws...)
        return tile_mapreduce(thread_backend(halo), per_tile, op, 1:tile_count(halo); scheduler=:static)
    end

    @eval function Base.$func(
            f::F, op::OP, z::Iterators.Zip{<:Tuple{ThreadedHaloArray,Vararg{ThreadedHaloArray}}}; kws...,
        ) where {F<:Function, OP}
        g(args...) = f(args)
        return $func(g, op, z.is...; kws...)
    end
end

function Base.any(f::F, u::LocalHaloArray) where {F<:Function}
    return any(f, interior_view(u))
end

function Base.all(f::F, u::LocalHaloArray) where {F<:Function}
    return all(f, interior_view(u))
end

function Base.any(f::F, u::ThreadedHaloArray) where {F<:Function}
    return tile_mapreduce(thread_backend(u), tile_id -> any(f, interior_view(u, tile_id)), |,
        1:tile_count(u); scheduler=:static)
end

function Base.all(f::F, u::ThreadedHaloArray) where {F<:Function}
    return tile_mapreduce(thread_backend(u), tile_id -> all(f, interior_view(u, tile_id)), &,
        1:tile_count(u); scheduler=:static)
end

# mapreduce/mapfoldl/mapfoldr over a multi-field container reduce each field
# across the inputs, then reduce the per-field results. One definition covers any
# AbstractHaloCollection (MultiHaloArray + ArrayOfHaloArray) via `eachfield`.
for func in (:mapreduce, :mapfoldl, :mapfoldr)
    @eval function Base.$func(
            f::F, op::OP, halo::AbstractHaloCollection, etc::Vararg{AbstractHaloCollection}; kws...,
        ) where {F<:Function, OP}
        # Reduce each field across the inputs, then combine the per-field results.
        # `map` over `eachfield` directly (multi-iterator): a MultiHaloArray's
        # fields are a Tuple → the results stay a Tuple (no allocation); an
        # ArrayOfHaloArray's are an Array → a small O(#fields) results array
        # (irreducible without regressing the tuple case — `zip` of tuples is
        # type-unstable). `kws` (e.g. `init`) apply once, at the combine.
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
        all(isactive, (halo, etc...)) ||
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
    isactive(halo) || throw(ErrorException("MaybeHaloArray: attempt to reduce inactive value"))
    return all(f, getdata(halo))
end

function Base.any(f::F, halo::MaybeHaloArray) where {F<:Function}
    isactive(halo) || throw(ErrorException("MaybeHaloArray: attempt to reduce inactive value"))
    return any(f, getdata(halo))
end

Base.sum(halo::AbstractHaloArray) = mapreduce(identity, +, halo)
Base.sum(f::F, halo::AbstractHaloArray) where {F<:Function} = mapreduce(f, +, halo)
Base.maximum(halo::AbstractHaloArray) = mapreduce(identity, max, halo)
Base.minimum(halo::AbstractHaloArray) = mapreduce(identity, min, halo)

# Fast `sum` (no-arg): contiguous-aware interior reduction (see `_interior_acc`).
# `sum(f, ...)` and max/min keep the generic mapreduce path above.
Base.sum(u::LocalHaloArray) = _interior_acc(identity, parent(u), interior_range(u))
Base.sum(u::ThreadedHaloArray) = tile_mapreduce(thread_backend(u),
    t -> _interior_acc(identity, tile_parent(u, t), interior_range(u, t)), +,
    1:tile_count(u); scheduler=:static)

# dot, norm, and the in-place BLAS-1 ops (rmul!/lmul!/axpy!/axpby!) — all built
# on the mapreduce/broadcast above — live in vector_space.jl.


# mapreduce_haloarray_dims and mapreduce_mhaloarray_dims live in mpi_support.jl
