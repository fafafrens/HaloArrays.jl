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
# The kernels loop over storage tiles exactly like the unified RHS — `tile_count`
# is 1 for Local/MPI (the whole padded block / this rank's block) and many for a
# ThreadedHaloArray — and index the raw `tile_parent` array over `interior_range`
# (so no scalar-getindex on the halo array itself). Collections delegate per
# field, reusing the single-array kernel.

"""
    swap!(x, y) -> (x, y)

Swap the interior contents of two halo arrays in place (BLAS-1 `swap`), with no
temporary allocation. Ghost cells are left as-is (re-filled by the next
[`synchronize_halo!`](@ref)). `x` and `y` must share geometry; works on any
backend and is MPI-safe (each rank/tile swaps its own cells).
"""
function swap!(x::AbstractSingleHaloArray, y::AbstractSingleHaloArray)
    @inbounds for tile in 1:tile_count(x)
        px = tile_parent(x, tile); py = tile_parent(y, tile)
        for I in CartesianIndices(interior_range(x))
            px[I], py[I] = py[I], px[I]
        end
    end
    return x, y
end

# Givens rotation:  x .= c*x + s*y,  y .= -conj(s)*x + c*y   (LinearAlgebra.rotate!)
function LinearAlgebra.rotate!(x::AbstractSingleHaloArray, y::AbstractSingleHaloArray, c, s)
    @inbounds for tile in 1:tile_count(x)
        px = tile_parent(x, tile); py = tile_parent(y, tile)
        for I in CartesianIndices(interior_range(x))
            a = px[I]; b = py[I]
            px[I] = c * a + s * b
            py[I] = c * b - conj(s) * a
        end
    end
    return x, y
end

# Householder reflection:  x .= c*x + s*y,  y .= conj(s)*x − c*y  (LinearAlgebra.reflect!)
function LinearAlgebra.reflect!(x::AbstractSingleHaloArray, y::AbstractSingleHaloArray, c, s)
    @inbounds for tile in 1:tile_count(x)
        px = tile_parent(x, tile); py = tile_parent(y, tile)
        for I in CartesianIndices(interior_range(x))
            a = px[I]; b = py[I]
            px[I] = c * a + s * b
            py[I] = conj(s) * a - c * b
        end
    end
    return x, y
end

# Field collections: apply the single-array kernel field by field (also alloc-free).
swap!(x::AbstractHaloCollection, y::AbstractHaloCollection) =
    (foreach(swap!, eachfield(x), eachfield(y)); (x, y))
LinearAlgebra.rotate!(x::AbstractHaloCollection, y::AbstractHaloCollection, c, s) =
    (foreach((fx, fy) -> rotate!(fx, fy, c, s), eachfield(x), eachfield(y)); (x, y))
LinearAlgebra.reflect!(x::AbstractHaloCollection, y::AbstractHaloCollection, c, s) =
    (foreach((fx, fy) -> reflect!(fx, fy, c, s), eachfield(x), eachfield(y)); (x, y))
