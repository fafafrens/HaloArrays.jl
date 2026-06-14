# Stiff reaction–diffusion in 1-D, solved with an *implicit* SciML scheme using
# autodiff Jacobians — with the HaloArray itself as the ODE state.
#
#     u_t = D·u_xx + R·u·(1 − u)        (Fisher–KPP; the diffusion makes it stiff)
#
# The point here is the integration pattern, not the physics:
#
#   • The HaloArray is the SciML state *directly* — no marshalling to a Vector.
#   • The RHS is the ordinary `synchronize_halo!` + stencil, in place.
#   • We use an implicit integrator (FBDF) with autodiff Jacobians, solved
#     *matrix-free* (`concrete_jac = false`) — no dense Jacobian is ever formed.
#
# The one subtlety is the *linear solver* for the Newton/W system. A custom array
# state works with any solver that allocates its work vectors via `similar(b)`
# (which preserves the halo geometry) and applies the operator via `mul!`:
#
#   • `SimpleGMRES()` — built into LinearSolve, `similar`-based. The simplest
#     choice (used in the first run below).
#   • `IterativeSolversJL_CG()` — `similar`-based and cheaper, but CG needs a
#     symmetric (SPD) operator, so only for pure diffusion (no reaction term).
#   • Krylov.jl via its `KrylovConstructor` (also `similar`-based) — wired in with
#     a `LinearSolveFunction` in `krylov_gmres_bridge` below, so Krylov.jl's
#     solvers (and its preconditioners) run with the HaloArray as the vector.
#
# What does *not* work is LinearSolve's `KrylovJL_*` wrappers: they allocate work
# vectors via `S(undef, n)`, and a geometry-carrying HaloArray has no `(undef, n)`
# constructor. You can still use Krylov.jl — through `KrylovConstructor`, as the
# bridge shows — just not the `KrylovJL_*` wrapper.

using HaloArrays
using OrdinaryDiffEq
using LinearSolve
using Krylov
using Printf

const D = 1.0           # diffusion coefficient
const R = 8.0           # reaction rate
const E1 = CartesianIndex(1)

function rhs!(du, u, p, t)
    dx2inv = p
    synchronize_halo!(u)                 # ghosts (periodic); runs on Duals too
    s = parent(u)
    d = parent(du)
    @inbounds for I in CartesianIndices(interior_range(u))
        lap = (s[I - E1] - 2 * s[I] + s[I + E1]) * dx2inv
        d[I] = D * lap + R * s[I] * (1 - s[I])
    end
    return nothing
end

# Route LinearSolve through Krylov.jl's native KrylovConstructor workspace, which
# allocates with `similar(b)` and applies the (matrix-free) operator via `mul!`.
# This is what lets Krylov.jl run with the HaloArray as the solver vector.
function krylov_gmres_bridge(A, b, u, p, isfresh, Pl, Pr, cacheval; kwargs...)
    workspace = krylov_workspace(Val(:gmres), KrylovConstructor(b))
    gmres!(workspace, A, b)
    copyto!(u, Krylov.solution(workspace))
    return u
end

function run_stiff_reaction_diffusion(; nx = 128, tend = 0.3,
        linsolve = SimpleGMRES(), label = "SimpleGMRES")
    dx = 1.0 / nx
    u0 = LocalHaloArray(Float64, (nx,), 1; boundary_condition = :periodic)
    iv = interior_view(u0)
    for i in 1:nx
        x = (i - 0.5) * dx
        iv[i] = 0.1 + 0.8 * exp(-50 * (x - 0.5)^2)     # a localized bump
    end
    prob = ODEProblem(rhs!, u0, (0.0, tend), inv(dx^2))

    # Implicit, autodiff Jacobians, matrix-free, HaloArray as the state.
    alg = FBDF(linsolve = linsolve, concrete_jac = false)
    sol = solve(prob, alg; reltol = 1e-7, abstol = 1e-7, save_everystep = false)

    # High-accuracy explicit reference for a correctness check.
    ref = solve(prob, Tsit5(); reltol = 1e-10, abstol = 1e-10, save_everystep = false)
    err = maximum(abs, collect(interior_view(sol.u[end])) .- collect(interior_view(ref.u[end])))

    @printf("implicit FBDF + %s (autodiff, matrix-free)\n", label)
    @printf("  retcode           : %s\n", sol.retcode)
    @printf("  accepted steps    : %d  (vs explicit Tsit5: %d)\n",
            sol.stats.naccept, ref.stats.naccept)
    @printf("  max |impl − expl| : %.2e\n", err)
    @printf("  mass %.5f -> %.5f\n", sum(interior_view(u0)), sum(interior_view(sol.u[end])))
    err < 1e-5 || error("implicit solution disagrees with the explicit reference")
    return sol
end

function main()
    # Two linear-solver routes, both with the HaloArray as the ODE state:
    run_stiff_reaction_diffusion(; linsolve = SimpleGMRES(), label = "SimpleGMRES")
    println()
    run_stiff_reaction_diffusion(; linsolve = LinearSolveFunction(krylov_gmres_bridge),
        label = "Krylov.jl (KrylovConstructor bridge)")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
