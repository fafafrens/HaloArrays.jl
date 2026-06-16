# ============================================================
# HaloArrays.jl — matrix-free Poisson solve via a SciMLOperator
#
# Run with:
#   julia --project=examples examples/poisson/operator.jl
#
# This example turns a HaloArray stencil into a composable linear operator
# WITHOUT defining any new types in the package — we wrap a function with
# SciMLOperators.FunctionOperator and feed it to coordinate-free Krylov
# solvers (CG, BiCGStab, GMRES) that run directly on the halo array.
#
# We solve the 2-D Poisson equation
#
#     -∇²u = f   on (0,1)²,   u = 0 on the boundary
#
# with the manufactured solution u_exact = x(1-x)·y(1-y) (zero on the
# boundary, rich in Fourier modes so the solvers genuinely iterate). Then
# f = -∇²u_exact = 2(x(1-x) + y(1-y)), and we can measure the error.
#
# Homogeneous Dirichlet (Antireflecting) BC makes -∇² symmetric
# positive-definite, so all three methods apply (CG is the natural choice;
# BiCGStab/GMRES are shown to demonstrate the operator works with general
# solvers too). Periodic would be singular.
#
# The solvers (examples/poisson/krylov_solvers.jl) are coordinate-free: they touch
# the unknown only through mul!, dot, norm and broadcast, and HaloArrays.jl
# defines dot/norm as GLOBAL reductions. So swapping LocalHaloArray for an
# MPI HaloArray gives a correct distributed solve with no change to the
# solvers — the operator below works unchanged too, since it uses parent +
# interior_range on the owned block. (A ThreadedHaloArray would need the
# tile-loop form of the stencil; see tutorials/threaded.jl.)
# ============================================================

using HaloArrays
using SciMLOperators
using LinearAlgebra: mul!
using LinearSolve
using Printf

include("krylov_solvers.jl")        # cg!, bicgstab!, gmres! (hand-rolled, for §3–4)

# ------------------------------------------------------------
# 1. The operator: -∇² applied to a HaloArray
# ------------------------------------------------------------
# FunctionOperator's in-place convention is op(y, x, u, p, t): y output,
# x input, u (unused) state, p parameters, t time. synchronize_halo!(x)
# fills ghosts according to the BC (Antireflecting ⇒ u = 0 at the wall).

function neg_laplacian!(y, x, _u, p, _t)
    inv_h2 = p.inv_h2
    synchronize_halo!(x)
    xd = parent(x); yd = parent(y)
    ex = CartesianIndex(1, 0); ey = CartesianIndex(0, 1)
    @inbounds for I in CartesianIndices(interior_range(x))
        lap = (xd[I+ex] - 2xd[I] + xd[I-ex]) + (xd[I+ey] - 2xd[I] + xd[I-ey])
        yd[I] = -lap * inv_h2
    end
    return y
end

# ------------------------------------------------------------
# 2. Build the manufactured Poisson problem on an n×n grid
# ------------------------------------------------------------
function setup(n)
    h = 1.0 / n; inv_h2 = 1.0 / h^2; center(i) = (i - 0.5) * h
    dirichlet = ((Antireflecting(), Antireflecting()), (Antireflecting(), Antireflecting()))
    uex = LocalHaloArray(Float64, (n, n), 1; boundary_condition=dirichlet)
    rhs = LocalHaloArray(Float64, (n, n), 1; boundary_condition=dirichlet)
    fill_from_global_indices!(uex) do I
        cx, cy = center(I[1]), center(I[2]); cx * (1 - cx) * cy * (1 - cy)
    end
    fill_from_global_indices!(rhs) do I
        cx, cy = center(I[1]), center(I[2]); 2 * (cx * (1 - cx) + cy * (1 - cy))
    end
    L = FunctionOperator(neg_laplacian!, similar(rhs), similar(rhs);
        islinear=true, isconstant=true, issymmetric=true, isposdef=true,
        p=(inv_h2=inv_h2,))
    return L, rhs, uex, h
end

# ------------------------------------------------------------
# 3. CG convergence study — error should drop ~4× per doubling (O(h²))
# ------------------------------------------------------------
println("=" ^ 66)
println("Matrix-free Dirichlet Poisson — CG convergence (O(h²))")
println("=" ^ 66)
for n in (32, 64, 128)
    L, rhs, uex, h = setup(n)
    u = similar(rhs); fill!(u, 0.0)
    _, iters, res = cg!(u, L, rhs; tol=1e-10)
    err = maximum(abs, interior_view(u) .- interior_view(uex))
    @printf("  n=%-4d  CG iters=%-4d  residual=%.2e  max|u-u_exact|=%.3e  (h²=%.3e)\n",
        n, iters, res, err, h^2)
end

# ------------------------------------------------------------
# 4. Same problem, three different Krylov methods — all converge to
#    the same solution (the operator is solver-agnostic)
# ------------------------------------------------------------
println()
println("=" ^ 66)
println("Same n=64 problem solved three ways")
println("=" ^ 66)
let n = 64
    L, rhs, uex, _ = setup(n)
    solvers = (("CG",       (u, A, b) -> cg!(u, A, b; tol=1e-10)),
               ("BiCGStab", (u, A, b) -> bicgstab!(u, A, b; tol=1e-10)),
               ("GMRES(50)",(u, A, b) -> gmres!(u, A, b; tol=1e-10, restart=50)))
    for (name, solve) in solvers
        u = similar(rhs); fill!(u, 0.0)
        _, iters, res = solve(u, L, rhs)
        err = maximum(abs, interior_view(u) .- interior_view(uex))
        @printf("  %-10s iters=%-4d  residual=%.2e  max|u-u_exact|=%.3e\n",
            name, iters, res, err)
    end
end

# ------------------------------------------------------------
# 5. The same operator through LinearSolve.jl — using the coordinate-free
#    solvers HaloArrays ships (HaloCG / HaloBiCGStab / HaloGMRES).
#
# These work because the unknown is a *2-D* halo array: LinearSolve's KrylovJL_*
# (and SimpleGMRES) model the unknown as a flat `AbstractVector`, which a
# geometry-carrying N-D halo array is not. HaloCG/HaloBiCGStab/HaloGMRES are
# coordinate-free (mul!/dot/norm/broadcast only), so they take the halo array
# directly — and stay correct under MPI. Pass `u0` explicitly: LinearSolve's
# default initial guess flattens `b`, which a halo array can't be.
# ------------------------------------------------------------
println()
println("=" ^ 66)
println("Same n=64 problem through LinearSolve (coordinate-free, N-D)")
println("=" ^ 66)
let n = 64
    L, rhs, uex, _ = setup(n)
    for (name, alg) in (("HaloCG", HaloCG()),
                        ("HaloBiCGStab", HaloBiCGStab()),
                        ("HaloGMRES(50)", HaloGMRES(restart = 50)))
        sol = solve(LinearProblem(L, rhs; u0 = zero(rhs)), alg; reltol = 1e-10)
        err = maximum(abs, interior_view(sol.u) .- interior_view(uex))
        @printf("  %-14s retcode=%-8s  max|u-u_exact|=%.3e\n", name, sol.retcode, err)
    end
end

println()
println("Poisson operator example complete.")
