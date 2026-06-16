# Stiff reaction–diffusion (Fisher–KPP) solved with an *implicit* SciML scheme,
# autodiff Jacobians, and a *matrix-free* Krylov solve — with the HaloArray
# itself as the ODE state.
#
#     u_t = D·u_xx + R·u·(1 − u)        (the diffusion makes it stiff)
#
# Key points:
#   • The HaloArray is the SciML state directly (no marshalling to a Vector).
#   • Implicit FBDF + `concrete_jac = false` ⇒ no dense Jacobian; the Newton/W
#     system is solved by GMRES using only matrix-vector products.
#   • The linear solver is `HaloKrylov(:gmres)` from HaloArrays' LinearSolve/Krylov
#     extension — a `KrylovJL` whose workspace is built with `similar` (via
#     `KrylovConstructor`), so the HaloArray can be the solver vector. (The stock
#     `KrylovJL_*` allocate via `S(undef, n)`, which a geometry-carrying HaloArray
#     has no constructor for; the extension fixes that.) The state here is 1-D —
#     `HaloKrylov` needs that, since Krylov.jl requires `b::AbstractVector`. For
#     a 2-D/3-D state use the coordinate-free `HaloCG`/`HaloGMRES`/`HaloBiCGStab`.
#
# Only the state *constructor* changes between backends — the RHS is
# backend-agnostic (it loops over storage tiles: one for Local/MPI, many for
# Threaded), and the solver and physics are identical. This file runs the Local
# and Threaded backends; the distributed (MPI) version is the companion
# `stiff_reaction_diffusion_implicit_mpi_1d.jl`.

using HaloArrays
using OrdinaryDiffEq
using LinearSolve
using Krylov
using Printf

const D = 1.0
const R = 8.0
const E1 = CartesianIndex(1)

ic(x) = 0.1 + 0.8 * exp(-50 * (x - 0.5)^2)          # a localized bump

# Backend-agnostic RHS. Loop over the storage tiles — `tile_count` is 1 for a
# LocalHaloArray or a distributed HaloArray (its sole tile is the whole padded
# block / this rank's block) and many for a ThreadedHaloArray — and apply the
# stencil to each tile's padded array. `synchronize_halo!` fills ghosts (also on
# the ForwardDiff Duals during the Jacobian-vector products).
function rhs!(du, u, p, t)
    dx2inv = p
    synchronize_halo!(u)
    for tile in 1:tile_count(u)
        s = tile_parent(u, tile); d = tile_parent(du, tile)
        @inbounds for I in CartesianIndices(interior_range(u))
            d[I] = D * (s[I - E1] - 2s[I] + s[I + E1]) * dx2inv + R * s[I] * (1 - s[I])
        end
    end
    return nothing
end

# Solve implicitly + matrix-free, and check against a high-accuracy explicit
# reference. Backend-agnostic: every operation here (broadcast, sum, maximum) is
# interior-only and works on any halo-array backend.
function solve_stiff(u0, f!; nx, tend = 0.3, label)
    prob = ODEProblem(f!, u0, (0.0, tend), inv((1.0 / nx)^2))
    # `HaloKrylov(:gmres)` (the LinearSolve/Krylov extension) runs GMRES with the
    # HaloArray itself as the solver vector — matrix-free, cached. `concrete_jac =
    # false` keeps the Newton/W system matrix-free (Jacobian-vector products).
    alg = FBDF(linsolve = HaloKrylov(:gmres), concrete_jac = false)
    sol = solve(prob, alg; reltol = 1e-7, abstol = 1e-7, save_everystep = false)
    ref = solve(prob, Tsit5(); reltol = 1e-10, abstol = 1e-10, save_everystep = false)
    err = maximum(abs, sol.u[end] - ref.u[end])
    @printf("%-24s retcode=%s  steps=%-4d (vs Tsit5 %d)  max|impl-expl|=%.2e\n",
            label, sol.retcode, sol.stats.naccept, ref.stats.naccept, err)
    err < 1e-5 || error("$label: implicit solution disagrees with the explicit reference")
    return sol
end

function main()
    nx = 128
    u_local = LocalHaloArray(Float64, (nx,), 1; boundary_condition = :periodic)
    fill_from_global_indices!(I -> ic((I[1] - 0.5) / nx), u_local)
    solve_stiff(u_local, rhs!; nx = nx, label = "LocalHaloArray")

    ntiles = max(2, Threads.nthreads())
    tile = 64
    gnx = ntiles * tile
    u_threaded = ThreadedHaloArray(Float64, (tile,), 1; dims = (ntiles,), boundary_condition = :periodic)
    fill_from_global_indices!(I -> ic((I[1] - 0.5) / gnx), u_threaded)
    solve_stiff(u_threaded, rhs!; nx = gnx, label = "ThreadedHaloArray $(ntiles)x$(tile)")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
