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
    dot(interior_view(x), interior_view(y))
LinearAlgebra.dot(x::ThreadedHaloArray, y::ThreadedHaloArray) =
    tile_mapreduce(thread_backend(x),
        t -> dot(interior_view(x, t), interior_view(y, t)), +, 1:tile_count(x); scheduler=:static)
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

# ---- in-place BLAS-1 updates (elementwise, via interior-only broadcast) ------
LinearAlgebra.rmul!(x::AbstractHaloArray, s::Number) = (x .= x .* s)
LinearAlgebra.lmul!(s::Number, x::AbstractHaloArray) = (x .= s .* x)
LinearAlgebra.axpy!(s::Number, x::AbstractHaloArray, y::AbstractHaloArray) = (y .= y .+ s .* x)
LinearAlgebra.axpby!(s::Number, x::AbstractHaloArray, t::Number, y::AbstractHaloArray) =
    (y .= s .* x .+ t .* y)

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

@inline function _swap_tile!(px, py, rng)
    @inbounds for I in CartesianIndices(rng)
        px[I], py[I] = py[I], px[I]
    end
end
# Givens rotation:  x .= c*x + s*y,  y .= -conj(s)*x + c*y
@inline function _rotate_tile!(px, py, rng, c, s)
    @inbounds for I in CartesianIndices(rng)
        a = px[I]; b = py[I]
        px[I] = c * a + s * b
        py[I] = c * b - conj(s) * a
    end
end
# Householder reflection:  x .= c*x + s*y,  y .= conj(s)*x − c*y
@inline function _reflect_tile!(px, py, rng, c, s)
    @inbounds for I in CartesianIndices(rng)
        a = px[I]; b = py[I]
        px[I] = c * a + s * b
        py[I] = conj(s) * a - c * b
    end
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
