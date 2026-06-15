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
function LinearAlgebra.norm(halo::AbstractSingleHaloArray, p::Real=2)
    if p == 2
        return sqrt(mapreduce(abs2, +, halo))
    elseif p == Inf
        return mapreduce(abs, max, halo)
    else
        return mapreduce(x -> abs(x)^p, +, halo)^(1/p)
    end
end

# ---- inner product (global reduction) ---------------------------------------
# ⟨x,y⟩ = Σ conj(xᵢ)·yᵢ over interior cells. The two-argument `mapreduce`
# inherits the correct global semantics on every backend (MPI Allreduce, threaded
# tile reduction, per-field for collections); it overrides the generic
# AbstractArray `dot`, which would only reduce locally (silently wrong across ranks).
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
