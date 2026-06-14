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
# The one catch — and the reason this passes an explicit `linsolve` — is how the
# Krylov solver allocates its work vectors:
#
#   • `KrylovJL_GMRES()` (the usual default) builds them with `S(undef, n)`,
#     assuming a plain `Vector` constructor. A HaloArray carries geometry (halo
#     width, boundary conditions, shape) and has no such constructor, so that
#     path errors out.
#   • `SimpleGMRES()` builds them with `similar(b)` — which returns a proper
#     HaloArray — and applies the operator via `mul!`. It therefore only ever
#     touches the state through the vector-space interface the HaloArray already
#     provides (`similar`, `dot`, `norm`, broadcast). That one works.
#
# So the rule for picking a linear solver for a custom array state is: it must
# allocate via `similar(b)` (not `S(undef, n)`) and apply via `mul!`. For a
# *symmetric* operator (pure diffusion, no reaction) `IterativeSolversJL_CG()`
# works too and is cheaper — but CG needs an SPD `W`, so use GMRES in general.

using HaloArrays
using OrdinaryDiffEq
using LinearSolve
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

function run_stiff_reaction_diffusion(; nx = 128, tend = 0.3)
    dx = 1.0 / nx
    u0 = LocalHaloArray(Float64, (nx,), 1; boundary_condition = :periodic)
    iv = interior_view(u0)
    for i in 1:nx
        x = (i - 0.5) * dx
        iv[i] = 0.1 + 0.8 * exp(-50 * (x - 0.5)^2)     # a localized bump
    end
    prob = ODEProblem(rhs!, u0, (0.0, tend), inv(dx^2))

    # Implicit, autodiff Jacobians, matrix-free, HaloArray as the state.
    alg = FBDF(linsolve = SimpleGMRES(), concrete_jac = false)
    sol = solve(prob, alg; reltol = 1e-7, abstol = 1e-7, save_everystep = false)

    # High-accuracy explicit reference for a correctness check.
    ref = solve(prob, Tsit5(); reltol = 1e-10, abstol = 1e-10, save_everystep = false)
    err = maximum(abs, collect(interior_view(sol.u[end])) .- collect(interior_view(ref.u[end])))

    @printf("implicit FBDF + SimpleGMRES (autodiff, matrix-free)\n")
    @printf("  retcode           : %s\n", sol.retcode)
    @printf("  accepted steps    : %d  (vs explicit Tsit5: %d)\n",
            sol.stats.naccept, ref.stats.naccept)
    @printf("  max |impl − expl| : %.2e\n", err)
    @printf("  mass %.5f -> %.5f\n", sum(interior_view(u0)), sum(interior_view(sol.u[end])))
    err < 1e-5 || error("implicit solution disagrees with the explicit reference")
    return sol
end

function main()
    run_stiff_reaction_diffusion()
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
