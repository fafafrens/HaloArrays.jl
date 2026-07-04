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
LinearAlgebra.norm(halo::AbstractSingleHaloArray) = sqrt(mapreduce(abs2, +, halo))
function LinearAlgebra.norm(halo::AbstractSingleHaloArray, p::Real)
    p == 2   && return norm(halo)
    p == Inf && return mapreduce(abs, max, halo)
    return mapreduce(x -> abs(x)^p, +, halo)^(1 / p)
end
# Fast 2-norm: contiguous-aware Σ|·|² over the parent (see `_interior_acc`) instead
# of `mapreduce(abs2, +, strided interior_view)`. The general-p method above still
# routes p == 2 here via `norm(halo)` (dynamic dispatch picks the concrete type).
LinearAlgebra.norm(u::LocalHaloArray) = sqrt(_interior_acc(abs2, parent(u), interior_range(u)))
LinearAlgebra.norm(u::ThreadedHaloArray) = sqrt(tile_mapreduce(thread_backend(u),
    t -> _interior_acc(abs2, tile_parent(u, t), interior_range(u, t)), +,
    1:tile_count(u); scheduler=:static))

# Collections: combine the per-field norms (each field already does its own global
# reduction — MPI Allreduce / tile combine). Forwarding per field avoids Base's
# generic `norm`, which iterates the collection through its scalar `AbstractArray`
# interface and allocates O(N) every call (a hot-path leak in any Krylov solve on a
# collection state). Mirrors the per-field `dot` below. ‖x‖₂ = √Σ_f ‖x_f‖²,
# ‖x‖_∞ = maxₚ ‖x_f‖_∞, ‖x‖_p = (Σ_f ‖x_f‖_p^p)^{1/p}.
LinearAlgebra.norm(x::AbstractHaloCollection) = sqrt(sum(f -> norm(f)^2, eachfield(x)))
function LinearAlgebra.norm(x::AbstractHaloCollection, p::Real)
    p == 2   && return norm(x)
    p == Inf && return maximum(f -> norm(f, Inf), eachfield(x))
    return sum(f -> norm(f, p)^p, eachfield(x))^(1 / p)
end

# ---- inner product (global reduction) ---------------------------------------
# ⟨x,y⟩ = Σ conj(xᵢ)·yᵢ over interior cells. Forward to the *interior* dot per
# backend rather than the two-argument `mapreduce(dot, +, x, y)`: Base's
# multi-iterator `mapreduce` materializes `map(dot, …)` into a full interior-sized
# array, allocating O(N) every call — and `dot` is in every Krylov inner loop.
# The interior dot is allocation-free (BLAS-backed for contiguous storage) and
# globally correct (MPI Allreduce; per-tile / per-field combine).
LinearAlgebra.dot(x::LocalHaloArray, y::LocalHaloArray) =
    _interior_dot(parent(x), parent(y), interior_range(x))
LinearAlgebra.dot(x::ThreadedHaloArray, y::ThreadedHaloArray) =
    tile_mapreduce(thread_backend(x),
        t -> _interior_dot(tile_parent(x, t), tile_parent(y, t), interior_range(x, t)), +,
        1:tile_count(x); scheduler=:static)
LinearAlgebra.dot(x::AbstractHaloCollection, y::AbstractHaloCollection) =
    sum(fxy -> dot(fxy[1], fxy[2]), zip(eachfield(x), eachfield(y)))
# MaybeHaloArray: forward to the inner array (its own dot/norm already do the right
# global reduction); an inactive value contributes nothing. Without these, `dot`
# hits the two-arg-mapreduce fallback below and `norm` hits Base's generic path —
# both O(N) per call (the same hot-path leak fixed for collections above).
LinearAlgebra.dot(x::MaybeHaloArray, y::MaybeHaloArray) =
    isactive(x) ? LinearAlgebra.dot(getdata(x), getdata(y)) : zero(eltype(x))
LinearAlgebra.norm(m::MaybeHaloArray) =
    isactive(m) ? LinearAlgebra.norm(getdata(m)) : zero(real(eltype(m)))
LinearAlgebra.norm(m::MaybeHaloArray, p::Real) =
    isactive(m) ? LinearAlgebra.norm(getdata(m), p) : zero(real(eltype(m)))
# Fallback for any other halo-array type.
LinearAlgebra.dot(x::AbstractHaloArray, y::AbstractHaloArray) = mapreduce(LinearAlgebra.dot, +, x, y)

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

# contiguous @simd over the leading (contiguous) dim of a dense Array parent …
@inline function _interior_scal!(p::Array, rng::Tuple, s)
    inner = rng[1]; outer = CartesianIndices(Base.tail(rng))
    @inbounds for J in outer
        @simd for i in inner; p[i, J] *= s; end
    end
end
@inline function _interior_axpy!(py::Array, px::Array, rng::Tuple, s)
    inner = rng[1]; outer = CartesianIndices(Base.tail(rng))
    @inbounds for J in outer
        @simd for i in inner; py[i, J] += s * px[i, J]; end
    end
end
@inline function _interior_axpby!(py::Array, px::Array, rng::Tuple, s, t)
    inner = rng[1]; outer = CartesianIndices(Base.tail(rng))
    @inbounds for J in outer
        @simd for i in inner; py[i, J] = s * px[i, J] + t * py[i, J]; end
    end
end
# … and the GPU / non-dense fallback: broadcast over the interior view.
@inline _interior_scal!(p::AbstractArray, rng::Tuple, s) = (@views p[rng...] .*= s; nothing)
@inline _interior_axpy!(py::AbstractArray, px::AbstractArray, rng::Tuple, s) =
    (@views py[rng...] .+= s .* px[rng...]; nothing)
@inline _interior_axpby!(py::AbstractArray, px::AbstractArray, rng::Tuple, s, t) =
    (@views py[rng...] .= s .* px[rng...] .+ t .* py[rng...]; nothing)

# Single-block backends (LocalHaloArray, MPI HaloArray): one parent block.
for HT in (:LocalHaloArray, :HaloArray)
    @eval begin
        LinearAlgebra.rmul!(x::$HT, s::Number) = (_interior_scal!(parent(x), interior_range(x), s); x)
        LinearAlgebra.lmul!(s::Number, x::$HT) = (_interior_scal!(parent(x), interior_range(x), s); x)
        LinearAlgebra.axpy!(s::Number, x::$HT, y::$HT) =
            (_interior_axpy!(parent(y), parent(x), interior_range(x), s); y)
        LinearAlgebra.axpby!(s::Number, x::$HT, t::Number, y::$HT) =
            (_interior_axpby!(parent(y), parent(x), interior_range(x), s, t); y)
    end
end

# ThreadedHaloArray: drive the per-tile kernels across the array's thread backend.
LinearAlgebra.rmul!(x::ThreadedHaloArray, s::Number) =
    (tile_foreach(thread_backend(x), t -> _interior_scal!(tile_parent(x, t), interior_range(x, t), s),
        1:tile_count(x); scheduler=:static); x)
LinearAlgebra.lmul!(s::Number, x::ThreadedHaloArray) =
    (tile_foreach(thread_backend(x), t -> _interior_scal!(tile_parent(x, t), interior_range(x, t), s),
        1:tile_count(x); scheduler=:static); x)
LinearAlgebra.axpy!(s::Number, x::ThreadedHaloArray, y::ThreadedHaloArray) =
    (tile_foreach(thread_backend(x), t -> _interior_axpy!(tile_parent(y, t), tile_parent(x, t), interior_range(x, t), s),
        1:tile_count(x); scheduler=:static); y)
LinearAlgebra.axpby!(s::Number, x::ThreadedHaloArray, t::Number, y::ThreadedHaloArray) =
    (tile_foreach(thread_backend(x), tt -> _interior_axpby!(tile_parent(y, tt), tile_parent(x, tt), interior_range(x, tt), s, t),
        1:tile_count(x); scheduler=:static); y)

# ---- swap + Givens/Householder on two vectors (elementwise, MPI-safe) --------
# These complete the BLAS-1 surface (SSWAP, SROT). Each mixes two whole vectors
# elementwise — no reduction — so it is correct local-per-rank/tile (MPI-safe).
# Both outputs depend on both old inputs, so each cell is updated through scalar
# locals in a single fused pass: no temporary vector, no extra traversal.
#
# A single-tile kernel does the cell math on one raw padded array over
# `interior_range` (so no scalar-getindex on the halo array itself). The
# per-backend methods just choose how to drive the tiles: a non-threaded array
# (Local/MPI) is one block, so its method runs the kernel straight on `parent`;
# a ThreadedHaloArray drives its tiles with `tile_foreach` so they split across
# threads — matching the threaded broadcast that already backs axpy!/lmul!/…
# Collections delegate per field, so each field picks its own driver.

# Dense Array parents (CPU): scalar fused pass — no temporary, both outputs from
# the two scalar locals. Non-Array (GPU) parents take the broadcast fallbacks
# below: each is an in-place two-output op, so it needs one temp (copy of old x);
# scalar-indexing a GPUArray would error / crawl. Same Array-gating as the other
# BLAS-1 ops; keeps the CPU path 0-alloc, the GPU path correct.
@inline function _swap_tile!(px::Array, py::Array, rng)
    @inbounds for I in CartesianIndices(rng)
        px[I], py[I] = py[I], px[I]
    end
end
# Givens rotation:  x .= c*x + s*y,  y .= -conj(s)*x + c*y
@inline function _rotate_tile!(px::Array, py::Array, rng, c, s)
    @inbounds for I in CartesianIndices(rng)
        a = px[I]; b = py[I]
        px[I] = c * a + s * b
        py[I] = c * b - conj(s) * a
    end
end
# Householder reflection:  x .= c*x + s*y,  y .= conj(s)*x − c*y
@inline function _reflect_tile!(px::Array, py::Array, rng, c, s)
    @inbounds for I in CartesianIndices(rng)
        a = px[I]; b = py[I]
        px[I] = c * a + s * b
        py[I] = conj(s) * a - c * b
    end
end

# --- GPU / non-dense fallbacks: broadcast over the interior views (one temp) ---
@inline function _swap_tile!(px::AbstractArray, py::AbstractArray, rng)
    @views xi = px[rng...]; @views yi = py[rng...]
    tmp = copy(xi); xi .= yi; yi .= tmp
    return nothing
end
@inline function _rotate_tile!(px::AbstractArray, py::AbstractArray, rng, c, s)
    @views xi = px[rng...]; @views yi = py[rng...]
    a = copy(xi)                          # old x (both outputs need it)
    @. xi = c * a + s * yi
    @. yi = c * yi - conj(s) * a
    return nothing
end
@inline function _reflect_tile!(px::AbstractArray, py::AbstractArray, rng, c, s)
    @views xi = px[rng...]; @views yi = py[rng...]
    a = copy(xi)
    @. xi = c * a + s * yi
    @. yi = conj(s) * a - c * yi
    return nothing
end

"""
    swap!(x, y) -> (x, y)

Swap the interior contents of two halo arrays in place (BLAS-1 `swap`), with no
temporary allocation. Ghost cells are left as-is (re-filled by the next
[`synchronize_halo!`](@ref)). `x` and `y` must share geometry; works on any
backend and is MPI-safe (each rank/tile swaps its own cells). On a
[`ThreadedHaloArray`](@ref) the tiles are processed in parallel.
"""
function swap!(x::AbstractSingleHaloArray, y::AbstractSingleHaloArray)
    _swap_tile!(parent(x), parent(y), interior_range(x))   # one block (Local/MPI)
    return x, y
end

function LinearAlgebra.rotate!(x::AbstractSingleHaloArray, y::AbstractSingleHaloArray, c, s)
    _rotate_tile!(parent(x), parent(y), interior_range(x), c, s)
    return x, y
end

function LinearAlgebra.reflect!(x::AbstractSingleHaloArray, y::AbstractSingleHaloArray, c, s)
    _reflect_tile!(parent(x), parent(y), interior_range(x), c, s)
    return x, y
end

# ThreadedHaloArray: split the tiles across threads (as the threaded broadcast does).
function swap!(x::ThreadedHaloArray, y::ThreadedHaloArray)
    rng = interior_range(x)
    tile_foreach(thread_backend(x), t -> _swap_tile!(tile_parent(x, t), tile_parent(y, t), rng),
        eachindex(parent(x)); scheduler=:static)
    return x, y
end

function LinearAlgebra.rotate!(x::ThreadedHaloArray, y::ThreadedHaloArray, c, s)
    rng = interior_range(x)
    tile_foreach(thread_backend(x), t -> _rotate_tile!(tile_parent(x, t), tile_parent(y, t), rng, c, s),
        eachindex(parent(x)); scheduler=:static)
    return x, y
end

function LinearAlgebra.reflect!(x::ThreadedHaloArray, y::ThreadedHaloArray, c, s)
    rng = interior_range(x)
    tile_foreach(thread_backend(x), t -> _reflect_tile!(tile_parent(x, t), tile_parent(y, t), rng, c, s),
        eachindex(parent(x)); scheduler=:static)
    return x, y
end

# Field collections: apply the per-field method (each field picks serial/threaded).
swap!(x::AbstractHaloCollection, y::AbstractHaloCollection) =
    (foreach(swap!, eachfield(x), eachfield(y)); (x, y))
LinearAlgebra.rotate!(x::AbstractHaloCollection, y::AbstractHaloCollection, c, s) =
    (foreach((fx, fy) -> rotate!(fx, fy, c, s), eachfield(x), eachfield(y)); (x, y))
LinearAlgebra.reflect!(x::AbstractHaloCollection, y::AbstractHaloCollection, c, s) =
    (foreach((fx, fy) -> reflect!(fx, fy, c, s), eachfield(x), eachfield(y)); (x, y))

# MaybeHaloArray: forward to the inner array when active, no-op when inactive
# (completes the BLAS-1 surface — `dot`/`norm`/`axpy!`/… already cover Maybe).
function swap!(x::MaybeHaloArray, y::MaybeHaloArray)
    isactive(x) && swap!(getdata(x), getdata(y))
    return x, y
end
function LinearAlgebra.rotate!(x::MaybeHaloArray, y::MaybeHaloArray, c, s)
    isactive(x) && LinearAlgebra.rotate!(getdata(x), getdata(y), c, s)
    return x, y
end
function LinearAlgebra.reflect!(x::MaybeHaloArray, y::MaybeHaloArray, c, s)
    isactive(x) && LinearAlgebra.reflect!(getdata(x), getdata(y), c, s)
    return x, y
end
