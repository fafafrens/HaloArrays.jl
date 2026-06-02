using Printf
using DiffEqBase
using OrdinaryDiffEq

include("common.jl")

# The same 2-D periodic heat problem on a LocalHaloArray and a ThreadedHaloArray,
# solved two ways:
#   (1) hand-written explicit Euler steps (solve_heat!)
#   (2) OrdinaryDiffEq with Tsit5            (heat_rhs!)
# Both arrays reuse the stencil kernels in common.jl, so the only difference is
# the storage type and how the inner loop is parallelised.

const ALPHA = 1.0
const CFL   = 0.2

# ---- (1) hand-stepped explicit Euler --------------------------------------
function stepped_local(; n=(64, 64), nt=100)
    u  = LocalHaloArray(Float64, n, 1; boundary_condition=:periodic)
    dx = ntuple(d -> 1.0 / n[d], Val(2))
    dt = stable_heat_dt(ALPHA, CFL, dx)
    fill_centered_gaussian!(u; baseline=1.0, amplitude=1.0)
    solve_heat!(u; alpha=ALPHA, dt, dx, nt)
    return u
end

function stepped_threaded(; tile_size=(32, 32), dims=(2, 2), nt=100)
    u  = ThreadedHaloArray(Float64, tile_size, 1; dims, boundary_condition=:periodic)
    dx = ntuple(d -> 1.0 / global_size(u)[d], Val(2))
    dt = stable_heat_dt(ALPHA, CFL, dx)
    fill_centered_gaussian!(u; baseline=1.0, amplitude=1.0)
    solve_heat!(u; alpha=ALPHA, dt, dx, nt)
    return u
end

# ---- (2) OrdinaryDiffEq with Tsit5 ----------------------------------------
function diffeq_solve(u0; nt=100, reltol=1.0e-6, abstol=1.0e-8)
    dx    = ntuple(d -> 1.0 / global_size(u0)[d], Val(ndims(u0)))
    tspan = (0.0, nt * stable_heat_dt(ALPHA, CFL, dx))
    sol   = solve(ODEProblem(heat_rhs!, u0, tspan, (ALPHA, dx)), Tsit5(); reltol, abstol)
    u = sol.u[end]
    synchronize_halo!(u)
    return u, sol
end

function diffeq_local()
    u0 = LocalHaloArray(Float64, (64, 64), 1; boundary_condition=:periodic)
    fill_centered_gaussian!(u0; baseline=1.0, amplitude=1.0)
    return diffeq_solve(u0)
end

function diffeq_threaded()
    u0 = ThreadedHaloArray(Float64, (32, 32), 1; dims=(2, 2), boundary_condition=:periodic)
    fill_centered_gaussian!(u0; baseline=1.0, amplitude=1.0)
    return diffeq_solve(u0)
end

function main()
    lu, tu = stepped_local(), stepped_threaded()
    @printf("explicit Euler  Local:    size=%s tiles=1  mean=%.12f\n",
        string(size(lu)), interior_mean(lu))
    @printf("explicit Euler  Threaded: size=%s tiles=%d mean=%.12f\n",
        string(size(tu)), tile_count(tu), interior_mean(tu))

    du, dsol   = diffeq_local()
    dtu, dtsol = diffeq_threaded()
    @printf("OrdinaryDiffEq  Local:    size=%s tiles=1  t=%.3e mean=%.12f\n",
        string(size(du)), dsol.t[end], interior_mean(du))
    @printf("OrdinaryDiffEq  Threaded: size=%s tiles=%d t=%.3e mean=%.12f\n",
        string(size(dtu)), tile_count(dtu), dtsol.t[end], interior_mean(dtu))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
