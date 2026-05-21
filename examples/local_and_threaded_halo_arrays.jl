using Printf

include("heat_diffusion_common.jl")

const ALPHA = 1.0
const CFL = 0.2

function run_local_example(; n=(64, 64), nt=100)
    u = LocalHaloArray(Float64, n, 1; boundary_condition=:periodic)
    dx = ntuple(d -> 1.0 / n[d], Val(2))
    dt = stable_heat_dt(ALPHA, CFL, dx)

    fill_centered_gaussian!(u; baseline=1.0, amplitude=1.0)
    solve_heat!(u; alpha=ALPHA, dt, dx, nt)
    return u, dt
end

function run_threaded_example(; tile_size=(32, 32), dims=(2, 2), nt=100)
    u = ThreadedHaloArray(Float64, tile_size, 1; dims, boundary_condition=:periodic)
    dx = ntuple(d -> 1.0 / global_size(u)[d], Val(2))
    dt = stable_heat_dt(ALPHA, CFL, dx)

    fill_centered_gaussian!(u; baseline=1.0, amplitude=1.0)
    solve_heat!(u; alpha=ALPHA, dt, dx, nt)
    return u, dt
end

function main()
    local_u, local_dt = run_local_example()
    threaded_u, threaded_dt = run_threaded_example()

    @printf("LocalHaloArray:    size=%s, tiles=1,    dt=%.3e, final mean=%.12f\n",
        string(size(local_u)), local_dt, interior_mean(local_u))
    @printf("ThreadedHaloArray: size=%s, tiles=%d, dt=%.3e, final mean=%.12f\n",
        string(size(threaded_u)), tile_count(threaded_u), threaded_dt, interior_mean(threaded_u))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
