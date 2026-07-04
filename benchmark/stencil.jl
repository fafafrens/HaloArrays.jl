# Stencil throughput: LocalHaloArray vs ThreadedHaloArray.
#
# Reuses the heat kernels from examples/heat/common.jl (same code paths a user
# runs), timing `synchronize_halo!` + one explicit Euler step per iteration and
# reporting sustained Mcell/s. Run with several thread counts to see scaling:
#
#   julia --project=benchmark       benchmark/stencil.jl
#   julia --project=benchmark -t 8  benchmark/stencil.jl
#
# HALO_BENCH_QUICK=1 shrinks the problem for a fast smoke run.

using Printf

include(joinpath(@__DIR__, "..", "examples", "heat", "common.jl"))

const QUICK = get(ENV, "HALO_BENCH_QUICK", "0") == "1"
const ALPHA = 1.0
const CFL   = 0.2

# Time nsteps of (halo refresh + stencil); return seconds.
function bench_steps!(u; dx, nsteps)
    dt   = stable_heat_dt(ALPHA, CFL, dx)
    next = similar(u)
    cur  = u
    synchronize_halo!(cur)                       # warmup / compile
    heat_step!(next, cur, ALPHA, dt, dx)
    t0 = time_ns()
    for _ in 1:nsteps
        synchronize_halo!(cur)
        heat_step!(next, cur, ALPHA, dt, dx)
        cur, next = next, cur
    end
    return (time_ns() - t0) / 1e9
end

# Aim for ~1 s of work per case so the numbers are stable but the run is short.
steps_for(ncells) = max(10, round(Int, (QUICK ? 2e7 : 4e8) / ncells))

function run_local(n)
    u  = LocalHaloArray(Float64, n, 1; boundary_condition=:periodic)
    dx = ntuple(d -> 1.0 / n[d], Val(length(n)))
    fill_centered_gaussian!(u)
    nsteps = steps_for(prod(n))
    t = bench_steps!(u; dx, nsteps)
    @printf("  Local    %-12s  %6d steps  %8.3f s  %9.1f Mcell/s\n",
        string(n), nsteps, t, prod(n) * nsteps / t / 1e6)
end

function run_threaded(n)
    nt   = Threads.nthreads()
    dims = (nt, 1)
    all(n .% dims .== 0) || return
    tile = (n[1] ÷ dims[1], n[2] ÷ dims[2])
    u  = ThreadedHaloArray(Float64, tile, 1; dims, boundary_condition=:periodic)
    dx = ntuple(d -> 1.0 / n[d], Val(length(n)))
    fill_centered_gaussian!(u)
    nsteps = steps_for(prod(n))
    t = bench_steps!(u; dx, nsteps)
    @printf("  Threaded %-12s  %6d steps  %8.3f s  %9.1f Mcell/s   (%d tiles, %d threads)\n",
        string(n), nsteps, t, prod(n) * nsteps / t / 1e6, tile_count(u), nt)
end

function main()
    sizes = QUICK ? [(128, 128)] : [(256, 256), (1024, 1024), (2048, 2048)]
    println("2-D heat stencil, halo width 1, Float64, periodic BC")
    println("threads = $(Threads.nthreads())")
    for n in sizes
        run_local(n)
        run_threaded(n)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
