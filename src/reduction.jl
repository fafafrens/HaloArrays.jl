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
for func in (:mapreduce, :mapfoldl, :mapfoldr)
    @eval function Base.$func(
            f::F, op::OP, halo::AbstractSingleHaloArray, etc::Vararg{AbstractSingleHaloArray}; kws...,
        ) where {F<:Function, OP}
        return _local_mapreduce($func, f, op, (halo, etc...); kws...)
    end

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
for func in (:mapreduce, :mapfoldl, :mapfoldr)
    # Single collection (the common case: sum/maximum/minimum/norm/…): fold over
    # the fields directly. No intermediate results container — 0-alloc for BOTH the
    # tuple-backed MultiHaloArray and the array-backed ArrayOfHaloArray (unlike a
    # `map` over `eachfield`, which materializes a Vector for the array case). The
    # field-combine op is `op` itself; `kws` (e.g. `init`) apply at that combine.
    @eval Base.$func(f::F, op::OP, halo::AbstractHaloCollection; kws...) where {F<:Function, OP} =
        $func(field -> $func(f, op, field), op, eachfield(halo); kws...)

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


# mapreduce_haloarray_dims and mapreduce_mhaloarray_dims live in mpi_support.jl
