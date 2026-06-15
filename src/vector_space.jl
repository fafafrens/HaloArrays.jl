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
# These complete the BLAS-1 surface (SSWAP, SROT). They mix two whole vectors
# elementwise — no reduction — so the interior-only broadcast is correct on every
# backend (each rank/tile rotates its own cells). Both outputs depend on both old
# inputs, so the new `x` is staged in a temporary before `x` is overwritten.

"""
    swap!(x, y) -> (x, y)

Swap the interior contents of two halo arrays in place (BLAS-1 `swap`). Ghost
cells are left as-is (re-filled by the next [`synchronize_halo!`](@ref)). `x` and
`y` must share geometry; works on any backend and is MPI-safe (each rank swaps
its own cells).
"""
function swap!(x::AbstractHaloArray, y::AbstractHaloArray)
    tmp = copy(x)
    x .= y
    y .= tmp
    return x, y
end

# Givens rotation:  [x; y] .= [c s; -conj(s) c] * [x; y]  (LinearAlgebra.rotate!)
function LinearAlgebra.rotate!(x::AbstractHaloArray, y::AbstractHaloArray, c, s)
    xnew = c .* x .+ s .* y
    y .= c .* y .- conj(s) .* x        # still the old `x` here
    x .= xnew
    return x, y
end

# Householder reflection:  x .= c*x + s*y,  y .= conj(s)*x − c*y  (LinearAlgebra.reflect!)
function LinearAlgebra.reflect!(x::AbstractHaloArray, y::AbstractHaloArray, c, s)
    xnew = c .* x .+ s .* y
    y .= conj(s) .* x .- c .* y        # still the old `x` here
    x .= xnew
    return x, y
end
