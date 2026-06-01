# ============================================================
# HaloArrays.jl — matrix-free Poisson solve via a SciMLOperator
#
# Run with:
#   julia --project=examples examples/poisson_operator.jl
#
# This example shows how to turn a HaloArray stencil into a
# composable linear operator WITHOUT defining any new types in the
# package — we just wrap a function with SciMLOperators.FunctionOperator
# and feed it to a Krylov method (a short self-contained CG here).
#
# We solve the 2-D Poisson equation
#
#     -∇²u = f   on (0,1)²,   u = 0 on the boundary
#
# with a manufactured solution u_exact = sin(πx) sin(πy), so
# f = -∇²u_exact = 2π² sin(πx) sin(πy) and we can measure the error.
#
# Homogeneous Dirichlet (Antireflecting) BC makes -∇² symmetric
# positive-definite, so CG converges. (Periodic would be singular.)
#
# This runs on a LocalHaloArray. The CG driver itself is backend-agnostic:
# it only needs mul!, dot, norm and broadcast, and HaloArrays.jl defines
# dot/norm as GLOBAL reductions (MPI Allreduce / threaded tile reduction),
# so the inner products stay correct across ranks and tiles. To go
# distributed, swap LocalHaloArray for a HaloArray on a CartesianTopology —
# the operator below works unchanged, since it uses parent + interior_range
# on the owned block. A ThreadedHaloArray would instead need the tile-loop
# form of the stencil (parent(u) is the tile collection, not one flat grid;
# see tutorial_threaded.jl); the CG driver stays the same.
# ============================================================

using HaloArrays
using SciMLOperators
using LinearAlgebra: dot, norm, mul!
using Printf

# ------------------------------------------------------------
# 1. The operator: -∇² applied to a HaloArray
# ------------------------------------------------------------
# FunctionOperator's in-place convention is op(y, x, u, p, t): y is the
# output, x the input vector, u the (unused) state, p the parameters, t the
# time. We synchronize_halo!(x) first so the stencil can read ghost cells;
# for the Antireflecting BC that enforces u = 0 at the wall.

function neg_laplacian!(y, x, _u, p, _t)
    inv_h2 = p.inv_h2
    synchronize_halo!(x)                      # fill ghosts (applies the BC)
    xd = parent(x)
    yd = parent(y)
    ex = CartesianIndex(1, 0)
    ey = CartesianIndex(0, 1)
    @inbounds for I in CartesianIndices(interior_range(x))
        lap = (xd[I+ex] - 2xd[I] + xd[I-ex]) +
              (xd[I+ey] - 2xd[I] + xd[I-ey])
        yd[I] = -lap * inv_h2
    end
    return y
end

# ------------------------------------------------------------
# 2. A short self-contained Conjugate Gradient
# ------------------------------------------------------------
# Works on any vector type that supports mul!, dot, norm, broadcast,
# similar, copy — which now includes every HaloArray backend.

function cg!(x, A, b; tol=1e-10, maxiter=2000)
    Ap = similar(b)
    mul!(Ap, A, x)                 # Ap = A*x  (operator application)
    r  = copy(b)
    r .-= Ap                       # residual r = b - A*x
    p  = copy(r)
    rsold = dot(r, r)
    bnorm = norm(b)
    res   = sqrt(rsold)
    iters = 0
    for k in 1:maxiter
        iters = k
        mul!(Ap, A, p)
        α = rsold / dot(p, Ap)
        x .+= α .* p
        r .-= α .* Ap
        rsnew = dot(r, r)
        res = sqrt(rsnew)
        res ≤ tol * bnorm && break
        p .= r .+ (rsnew / rsold) .* p
        rsold = rsnew
    end
    return x, iters, res
end

# ------------------------------------------------------------
# 3. Set up and solve the manufactured Poisson problem
# ------------------------------------------------------------

function run_poisson(; n=64)
    h      = 1.0 / n                       # cell width; centers at (i-1/2)h
    inv_h2 = 1.0 / h^2
    center(i) = (i - 0.5) * h

    # homogeneous Dirichlet via Antireflecting (ghost = -interior ⇒ u=0 at wall)
    dirichlet = ((Antireflecting(), Antireflecting()), (Antireflecting(), Antireflecting()))

    u_exact = LocalHaloArray(Float64, (n, n), 1; boundary_condition=dirichlet)
    rhs     = LocalHaloArray(Float64, (n, n), 1; boundary_condition=dirichlet)
    u       = LocalHaloArray(Float64, (n, n), 1; boundary_condition=dirichlet)

    # Manufactured solution u = x(1-x)·y(1-y): zero on the boundary, smooth,
    # but rich in Fourier modes — so CG actually iterates (a single sine mode
    # would be an eigenvector and converge in one step). Then -∇²u = 2(x(1-x)+y(1-y)).
    fill_from_global_indices!(u_exact) do I
        cx, cy = center(I[1]), center(I[2])
        cx * (1 - cx) * cy * (1 - cy)
    end
    fill_from_global_indices!(rhs) do I
        cx, cy = center(I[1]), center(I[2])
        2 * (cx * (1 - cx) + cy * (1 - cy))
    end
    fill!(u, 0.0)                          # initial guess

    # Wrap the stencil as a composable SciMLOperator (no new types).
    L = FunctionOperator(neg_laplacian!, u, rhs;
        islinear=true, isconstant=true, p=(inv_h2=inv_h2,))

    _, iters, res = cg!(u, L, rhs; tol=1e-10)

    err = maximum(abs, interior_view(u) .- interior_view(u_exact))
    @printf("  n=%-4d  CG iters=%-4d  residual=%.2e  max|u-u_exact|=%.3e  (≈O(h²)=%.3e)\n",
        n, iters, res, err, h^2)
    return u
end

println("=" ^ 64)
println("Matrix-free Dirichlet Poisson via FunctionOperator + CG")
println("=" ^ 64)

# A small convergence study: the error should drop ~4× when n doubles (O(h²)).
for n in (32, 64, 128)
    run_poisson(; n=n)
end

println()
println("Poisson operator example complete.")
