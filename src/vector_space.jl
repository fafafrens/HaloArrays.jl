# ============================================================
# Vector-space / Hilbert-space interface for halo arrays.
#
# Everything an iterative solver (Krylov.jl, IterativeSolvers, a hand-rolled
# GMRES, an implicit SciML integrator …) needs to treat a halo array as an
# element of a vector space, gathered in one place:
#
#   - scalar multiply / divide               (`*`, `/`)
#   - `norm` and inner product (`dot`)       — GLOBAL reductions (MPI Allreduce,
#       threaded tile reduction, per-field for collections), built on `mapreduce`
#   - in-place BLAS-1 updates                 (`rmul!`, `lmul!`, `axpy!`, `axpby!`)
#       — elementwise, via the interior-only broadcast, so solvers never fall back
#       to the generic scalar-`eachindex` loops (slower, worse inference, and
#       local-only/wrong under MPI)
#
# With these, a halo array can be the state vector of a matrix-free Krylov solve
# on any backend — distributed (MPI) included, since `dot`/`norm` are global and
# the elementwise ops are correct local-per-rank.
# ============================================================

# ---- scalar multiply / divide -----------------------------------------------
Base.:/(halo::AbstractSingleHaloArray, x::Number) = halo ./ x
Base.:*(halo::AbstractSingleHaloArray, x::Number) = halo .* x
Base.:*(x::Number, halo::AbstractSingleHaloArray) = x .* halo

# ---- norm (global reduction) ------------------------------------------------
# The 2-norm (the Krylov hot path) is its own branch-free method; the general-`p`
# method carries the `p == Inf` / else dispatch and reuses the 2-norm for p == 2.
LinearAlgebra.norm(halo::AbstractSingleHaloArray) =
    sqrt(_global(_local_sum(_elt_abs2, halo), +, communicator(halo)))
# Expressed in GLOBAL primitives (mapreduce), not the _local_* helpers, so the
# p-norm is globally correct on every backend through mapreduce's combine —
# only the hot 2-norm is written on the local part directly.
# `norm` (not `abs`) per element: for a scalar `norm(x::Number) == abs(x)`, so the
# numeric result is unchanged, while a vector-valued cell contributes its own
# Euclidean length — matching Base's `norm(::Array{<:AbstractVector}, p)`.
function LinearAlgebra.norm(halo::AbstractSingleHaloArray, p::Real)
    p == 2    && return norm(halo)
    p == 1    && return mapreduce(norm, +, halo)
    p == Inf  && return mapreduce(norm, max, halo)
    p == -Inf && return mapreduce(norm, min, halo)
    # Count of nonzero cells, in the SCALAR type of a norm (Float64 for an SVector
    # eltype, like Base's `norm(::Array{<:SVector}, 0)`; Float32 stays Float32).
    p == 0    && return oftype(sqrt(_sqnorm_zero(eltype(halo))), mapreduce(x -> !iszero(x), +, halo))
    return mapreduce(x -> norm(x)^p, +, halo)^(1 / p)
end
# The 2-norm above IS the fast path: `_local_sum` runs the contiguous-aware
# `_interior_acc` per tile (single-block inline, threaded across the backend),
# then one combine across ranks (`_global`, reduction.jl).

# Collections: the 2-norm sums the fields' local parts and combines ONCE
# (one Allreduce per norm on an MPI state, not one per field — this is in every
# Krylov iteration on a collection state); the p-norms combine the per-field
# global norms, each already a global reduction. Forwarding per field avoids
# Base's generic `norm`, which iterates the collection through its scalar
# `AbstractArray` interface and allocates O(N) every call. ‖x‖₂ = √Σ_f ‖x_f‖²,
# ‖x‖_∞ = maxₚ ‖x_f‖_∞, ‖x‖_p = (Σ_f ‖x_f‖_p^p)^{1/p}.
LinearAlgebra.norm(x::AbstractHaloCollection) =
    sqrt(_global(sum(f -> _local_sum(_elt_abs2, f), eachfield(x)), +, communicator(x)))
function LinearAlgebra.norm(x::AbstractHaloCollection, p::Real)
    p == 2    && return norm(x)
    p == 1    && return sum(f -> norm(f, 1), eachfield(x))
    p == Inf  && return maximum(f -> norm(f, Inf), eachfield(x))
    p == -Inf && return minimum(f -> norm(f, -Inf), eachfield(x))
    p == 0    && return sum(f -> norm(f, 0), eachfield(x))
    return sum(f -> norm(f, p)^p, eachfield(x))^(1 / p)
end

# ---- inner product (global reduction) ---------------------------------------
# ⟨x,y⟩ = Σ conj(xᵢ)·yᵢ over interior cells. Forward to the *interior* dot per
# backend rather than the two-argument `mapreduce(dot, +, x, y)`: Base's
# multi-iterator `mapreduce` materializes `map(dot, …)` into a full interior-sized
# array, allocating O(N) every call — and `dot` is in every Krylov inner loop.
# The interior dot is allocation-free (BLAS-backed for contiguous storage); the
# combine across ranks is one Allreduce (`_global`), for a collection too: its
# local part sums the fields (`_local_dot`, reduction.jl, which also rejects a
# field-count mismatch), so a Krylov inner product on a multi-field state is
# one collective, not one per field.
LinearAlgebra.dot(x::AbstractSingleHaloArray, y::AbstractSingleHaloArray) =
    _global(_local_dot(x, y), +, communicator(x))
LinearAlgebra.dot(x::AbstractHaloCollection, y::AbstractHaloCollection) =
    _global(_local_dot(x, y), +, communicator(x))
# MaybeHaloArray: forward to the inner array (its own dot/norm already do the right
# global reduction); an inactive value contributes nothing. Without these, `dot`
# hits the two-arg-mapreduce fallback below and `norm` hits Base's generic path —
# both O(N) per call (the same hot-path leak fixed for collections above).
# An inactive value contributes the SCALAR zero of the reduction (`dot`/`norm`
# return a scalar even for vector-valued cells — `zero(eltype)` would be an
# `SVector`), so `_dot_zero`/`_sqnorm_zero` give the right type on every eltype.
LinearAlgebra.dot(x::MaybeHaloArray, y::MaybeHaloArray) =
    is_active(x) ? LinearAlgebra.dot(getdata(x), getdata(y)) : _dot_zero(eltype(x), eltype(y))
LinearAlgebra.norm(m::MaybeHaloArray) =
    is_active(m) ? LinearAlgebra.norm(getdata(m)) : sqrt(_sqnorm_zero(eltype(m)))
LinearAlgebra.norm(m::MaybeHaloArray, p::Real) =
    is_active(m) ? LinearAlgebra.norm(getdata(m), p) : sqrt(_sqnorm_zero(eltype(m)))
# Every kind has its own `dot` above; a mixed pair (a single array with a
# collection) is an error. Without this method it reached Base's generic
# `dot(::AbstractArray, ::AbstractArray)`: scalar-indexed and rank-local under
# MPI, and silently a number whenever the lengths happen to agree.
LinearAlgebra.dot(x::AbstractHaloArray, y::AbstractHaloArray) = throw(ArgumentError(
    "dot between a $(nameof(typeof(x))) and a $(nameof(typeof(y))) is not defined"))

# ---- in-place BLAS-1 updates (elementwise) -----------------------------------
# Default path: the interior-only broadcast — correct on every backend (GPU
# included) and for collections/Maybe. But the interior view is strided, so on a
# dense CPU `Array` this broadcast vectorizes poorly (~1.7× slower than contiguous).
# The single-array backends route through contiguous `@simd` kernels (below) that
# dispatch on the parent type: dense `Array` → vectorized loop; anything else
# (GPU) → the same broadcast, so device parents keep their on-device kernels. Same
# split as the reductions (`_interior_acc`/`_interior_dot`). All elementwise, no
# reduction ⇒ correct local-per-rank/tile (MPI-safe, no communication).
LinearAlgebra.rmul!(x::AbstractHaloArray, s::Number) = (x .= x .* s)
LinearAlgebra.lmul!(s::Number, x::AbstractHaloArray) = (x .= s .* x)
LinearAlgebra.axpy!(s::Number, x::AbstractHaloArray, y::AbstractHaloArray) = (y .= y .+ s .* x)
LinearAlgebra.axpby!(s::Number, x::AbstractHaloArray, t::Number, y::AbstractHaloArray) =
    (y .= s .* x .+ t .* y)

# ---- Diagonal operators whose diagonal is a halo array ------------------------
# Diagonal scaling (Jacobi / error-weight preconditioning) reaches halo arrays
# through the SciML stack: OrdinaryDiffEq wraps every iterative linear solver
# with `Pl = InvPreconditioner(Diagonal(weight))`, `Pr = Diagonal(weight)`,
# where `weight = similar(u)` — i.e. a halo array. LinearAlgebra's generic
# Diagonal kernels apply these by scalar-indexing every *global* index, which
# is local-only on a distributed HaloArray (it works on 1 rank by accident and
# throws on 2+). The operation is elementwise, so route it through the
# interior-only broadcast: no communication, correct on every backend.
const _HaloDiagonal = LinearAlgebra.Diagonal{<:Any,<:AbstractHaloArray}

LinearAlgebra.mul!(out::AbstractHaloArray, D::_HaloDiagonal, b::AbstractHaloArray) =
    (out .= D.diag .* b; out)

function LinearAlgebra.mul!(out::AbstractHaloArray, D::_HaloDiagonal, b::AbstractHaloArray,
        α::Number, β::Number)
    d = D.diag
    # β == 0 must not *read* `out` (BLAS contract): `out` is often fresh
    # `similar` memory, and 0.0 * NaN would poison the result.
    if iszero(β)
        out .= α .* (d .* b)
    else
        out .= α .* (d .* b) .+ β .* out
    end
    return out
end

LinearAlgebra.ldiv!(out::AbstractHaloArray, D::_HaloDiagonal, b::AbstractHaloArray) =
    (out .= b ./ D.diag; out)
LinearAlgebra.ldiv!(D::_HaloDiagonal, b::AbstractHaloArray) = (b .= b ./ D.diag; b)

# `UniformScaling` (`I`, `c*I`), the other preconditioner spelling solvers pass
# around. LinearAlgebra's methods take an `AbstractVecOrMat`, which a 3-D halo
# array is not; these are elementwise, so the same interior-only broadcast.
LinearAlgebra.ldiv!(out::AbstractHaloArray, J::UniformScaling, b::AbstractHaloArray) =
    (out .= b ./ J.λ; out)
LinearAlgebra.ldiv!(J::UniformScaling, b::AbstractHaloArray) = (b .= b ./ J.λ; b)
LinearAlgebra.rdiv!(b::AbstractHaloArray, J::UniformScaling) = (b .= b ./ J.λ; b)

# The one kernel behind every in-place elementwise update of one tile's
# interior: `_interior_map!(f, dest, rng[, src])` writes `dest[I] = f(dest[I][, src[I]])`,
# `_interior_map2!(f, px, py, rng)` writes `(px[I], py[I]) = f(px[I], py[I])` in
# one fused pass (both outputs from both old inputs, no temporary). On a dense
# `Array` parent: a contiguous `@simd` loop over the leading dimension (the
# closure inlines; same speed as the hand-written loops it replaces). On any
# other parent (GPU): a broadcast over the interior views, so device parents
# keep their on-device kernels — the two-output form needs one temporary there.
@inline function _interior_map!(f::F, dest::Array, rng::Tuple) where {F}
    inner = rng[1]; outer = CartesianIndices(Base.tail(rng))
    @inbounds for J in outer
        @simd for i in inner; dest[i, J] = f(dest[i, J]); end
    end
    return nothing
end
@inline function _interior_map!(f::F, dest::Array, rng::Tuple, src::Array) where {F}
    inner = rng[1]; outer = CartesianIndices(Base.tail(rng))
    @inbounds for J in outer
        @simd for i in inner; dest[i, J] = f(dest[i, J], src[i, J]); end
    end
    return nothing
end
@inline function _interior_map2!(f::F, px::Array, py::Array, rng::Tuple) where {F}
    @inbounds for I in CartesianIndices(rng)
        px[I], py[I] = f(px[I], py[I])
    end
    return nothing
end
@inline _interior_map!(f::F, dest::AbstractArray, rng::Tuple) where {F} =
    (@views dest[rng...] .= f.(dest[rng...]); nothing)
@inline _interior_map!(f::F, dest::AbstractArray, rng::Tuple, src::AbstractArray) where {F} =
    (@views dest[rng...] .= f.(dest[rng...], src[rng...]); nothing)
@inline function _interior_map2!(f::F, px::AbstractArray, py::AbstractArray, rng::Tuple) where {F}
    @views xi = px[rng...]; @views yi = py[rng...]
    t = f.(xi, yi)                        # the pairs, one temporary
    xi .= first.(t); yi .= last.(t)
    return nothing
end

# Per-tile driving goes through `_foreach_tile` (abstract_haloarray.jl, next to
# the one-tile decomposition trait it is built on). The two-array kernels index
# BOTH parents with x's range under @inbounds, so they must pass the shared
# geometry guard first (an unchecked mismatch is an out-of-bounds read/write).
LinearAlgebra.rmul!(x::AbstractSingleHaloArray, s::Number) =
    (_foreach_tile(t -> _interior_map!(v -> v * s, tile_parent(x, t), interior_range(x, t)), x); x)
LinearAlgebra.lmul!(s::Number, x::AbstractSingleHaloArray) =           # s on the left: not `rmul!` for a non-commutative cell type
    (_foreach_tile(t -> _interior_map!(v -> s * v, tile_parent(x, t), interior_range(x, t)), x); x)
function LinearAlgebra.axpy!(s::Number, x::AbstractSingleHaloArray, y::AbstractSingleHaloArray)
    _check_same_geometry(x, y, "axpy!")
    _foreach_tile(t -> _interior_map!((yv, xv) -> yv + s * xv, tile_parent(y, t), interior_range(x, t), tile_parent(x, t)), x)
    return y
end
function LinearAlgebra.axpby!(s::Number, x::AbstractSingleHaloArray, t::Number, y::AbstractSingleHaloArray)
    _check_same_geometry(x, y, "axpby!")
    _foreach_tile(tt -> _interior_map!((yv, xv) -> s * xv + t * yv, tile_parent(y, tt), interior_range(x, tt), tile_parent(x, tt)), x)
    return y
end

# ---- swap + Givens/Householder on two vectors (elementwise, MPI-safe) --------
# These complete the BLAS-1 surface (SSWAP, SROT). Each mixes two whole vectors
# elementwise — no reduction — so it is correct local-per-rank/tile (MPI-safe);
# each is one pair-valued cell function through `_interior_map2!` above.
# Collections delegate per field, so each field picks its own driver.

"""
    swap!(x, y) -> (x, y)

Swap the interior contents of two halo arrays in place (BLAS-1 `swap`), with no
temporary allocation. Ghost cells are left as-is (re-filled by the next
[`synchronize_halo!`](@ref)). `x` and `y` must share geometry; works on any
backend and is MPI-safe (each rank/tile swaps its own cells). On a
[`ThreadedHaloArray`](@ref) the tiles are processed in parallel.
"""
function swap!(x::AbstractSingleHaloArray, y::AbstractSingleHaloArray)
    _check_same_geometry(x, y, "swap!")
    _foreach_tile(t -> _interior_map2!((a, b) -> (b, a), tile_parent(x, t), tile_parent(y, t), interior_range(x, t)), x)
    return x, y
end

"""
    dots!(h, V, w, j) -> h

`h[i] = dot(V[i], w)` for `i in 1:j`: the inner products of one vector with
`j` others, with a **single global reduction** where the backend has one. On a
distributed [`HaloArray`](@ref) that is one `Allreduce` of `j` numbers instead
of `j` separate ones; on the local and threaded backends there is nothing to
reduce globally and the `j` local reductions run as for `dot`. This is
the primitive classical Gram–Schmidt needs (see `HaloArrays.Krylov`).

`h` is a `Vector` (or a contiguous view of one — the MPI path reduces it in
place) of length ≥ `j`; entries past `j` are untouched. `V` is indexable
(`V[i]`); `w` and every `V[i]` must share geometry. A collection reduces once as
well, over all its fields. Anything else (e.g. [`MaybeHaloArray`](@ref), plain
vectors) falls back to `j` calls of `dot`.
"""
function dots!(h, V, w, j)
    for i in 1:j
        h[i] = LinearAlgebra.dot(V[i], w)
    end
    return h
end
# The j local products (a collection's over all its fields), then ONE combine
# of the j numbers: in place on an MPI communicator, nothing on the serial
# backends (`_global`, reduction.jl).
function dots!(h, V, w::Union{AbstractSingleHaloArray,AbstractHaloCollection}, j)
    for i in 1:j
        h[i] = _local_dot(V[i], w)
    end
    _global(view(h, 1:j), +, communicator(w))
    return h
end

# Givens rotation:  x .= c*x + s*y,  y .= -conj(s)*x + c*y
function LinearAlgebra.rotate!(x::AbstractSingleHaloArray, y::AbstractSingleHaloArray, c, s)
    _check_same_geometry(x, y, "rotate!")
    _foreach_tile(t -> _interior_map2!((a, b) -> (c * a + s * b, c * b - conj(s) * a),
                                       tile_parent(x, t), tile_parent(y, t), interior_range(x, t)), x)
    return x, y
end

# Householder reflection:  x .= c*x + s*y,  y .= conj(s)*x − c*y
function LinearAlgebra.reflect!(x::AbstractSingleHaloArray, y::AbstractSingleHaloArray, c, s)
    _check_same_geometry(x, y, "reflect!")
    _foreach_tile(t -> _interior_map2!((a, b) -> (c * a + s * b, conj(s) * a - c * b),
                                       tile_parent(x, t), tile_parent(y, t), interior_range(x, t)), x)
    return x, y
end

# Field collections: apply the per-field method (each field picks serial/threaded).
# Tuple broadcast rather than a two-tuple `foreach`, which silently stops at the
# shorter tuple (see the collection `dot`); `Ref` keeps the scalars out of it.
swap!(x::AbstractHaloCollection, y::AbstractHaloCollection) =
    (swap!.(eachfield(x), eachfield(y)); (x, y))
LinearAlgebra.rotate!(x::AbstractHaloCollection, y::AbstractHaloCollection, c, s) =
    (rotate!.(eachfield(x), eachfield(y), Ref(c), Ref(s)); (x, y))
LinearAlgebra.reflect!(x::AbstractHaloCollection, y::AbstractHaloCollection, c, s) =
    (reflect!.(eachfield(x), eachfield(y), Ref(c), Ref(s)); (x, y))

# MaybeHaloArray: forward to the inner array when active, no-op when inactive
# (completes the BLAS-1 surface — `dot`/`norm`/`axpy!`/… already cover Maybe).
function swap!(x::MaybeHaloArray, y::MaybeHaloArray)
    is_active(x) && swap!(getdata(x), getdata(y))
    return x, y
end
function LinearAlgebra.rotate!(x::MaybeHaloArray, y::MaybeHaloArray, c, s)
    is_active(x) && LinearAlgebra.rotate!(getdata(x), getdata(y), c, s)
    return x, y
end
function LinearAlgebra.reflect!(x::MaybeHaloArray, y::MaybeHaloArray, c, s)
    is_active(x) && LinearAlgebra.reflect!(getdata(x), getdata(y), c, s)
    return x, y
end
