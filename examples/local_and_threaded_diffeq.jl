using Printf
using DifferentialEquations

include("heat_diffusion_common.jl")

const ALPHA = 1.0
const CFL = 0.2

function solve_heat_diffeq(u0; alpha=ALPHA, cfl=CFL, domain_length=(1.0, 1.0), nt=100)
    dx = ntuple(d -> domain_length[d] / global_size(u0)[d], Val(ndims(u0)))
    dt = stable_heat_dt(alpha, cfl, dx)
    tspan = (0.0, nt * dt)
    prob = ODEProblem(heat_rhs!, u0, tspan, (alpha, dx))
    sol = solve(prob, Tsit5(); dt, adaptive=false, save_everystep=false)
    u = sol.u[end]
    synchronize_halo!(u)
    return u, dt
end

function run_local_example()
    u0 = LocalHaloArray(Float64, (64, 64), 1; boundary_condition=:periodic)
    fill_centered_gaussian!(u0; baseline=1.0, amplitude=1.0)
    return solve_heat_diffeq(u0)
end

function run_threaded_example()
    u0 = ThreadedHaloArray(Float64, (32, 32), 1; dims=(2, 2), boundary_condition=:periodic)
    fill_centered_gaussian!(u0; baseline=1.0, amplitude=1.0)
    return solve_heat_diffeq(u0)
end

function main()
    local_u, local_dt = run_local_example()
    threaded_u, threaded_dt = run_threaded_example()

    @printf("DifferentialEquations LocalHaloArray:    size=%s, tiles=1,    dt=%.3e, final mean=%.12f\n",
        string(size(local_u)), local_dt, interior_mean(local_u))
    @printf("DifferentialEquations ThreadedHaloArray: size=%s, tiles=%d, dt=%.3e, final mean=%.12f\n",
        string(size(threaded_u)), tile_count(threaded_u), threaded_dt, interior_mean(threaded_u))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
