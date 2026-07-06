include("common.jl")
include("heat_solver_common.jl")

using Base.Threads: nthreads

function option_problem_size(options, ndims, default)
    if haskey(options, "size")
        return option_tuple(options, "size", ndims, default)
    elseif haskey(options, "owned-size")
        @warn "--owned-size is accepted for compatibility; use --size for local/threaded heat solver benchmarks" maxlog=1
        return option_tuple(options, "owned-size", ndims, default)
    else
        return _default_tuple(default, ndims)
    end
end

function compare_backends!(local_u, local_tmp, threaded_u, threaded_tmp, problem_size;
        steps, alpha, dt, dx)
    fill_heat_initial!(local_u, problem_size)
    fill_heat_initial!(threaded_u, problem_size)
    solve_heat_steps!(local_u, local_tmp; steps=steps, alpha=alpha, dt=dt, dx=dx)
    solve_heat_steps!(threaded_u, threaded_tmp; steps=steps, alpha=alpha, dt=dt, dx=dx)

    local_solution = snapshot_interior(local_u)
    threaded_solution = snapshot_interior(threaded_u)
    threaded_error = maximum(abs.(threaded_solution .- local_solution))
    println("  Threaded vs Local max error: ", threaded_error)
    println()
    return nothing
end

function main()
    options = parse_args()
    nd = option_int(options, "ndims", 2)
    halo_width_value = option_int(options, "halo", 1)
    samples = option_int(options, "samples", 20)
    warmups = option_int(options, "warmups", 3)
    steps = option_int(options, "steps", 10)
    alpha = parse(Float64, option_string(options, "alpha", "0.01"))
    cfl = parse(Float64, option_string(options, "cfl", "0.4"))
    timer = Symbol(option_string(options, "timer", "manual"))
    timer in (:manual, :benchmarktools) || error("--timer must be manual or benchmarktools")
    problem_size = option_problem_size(options, nd, 64)
    tile_dims = option_tuple(options, "tile-dims", nd, 2)
    dx = ntuple(d -> 1.0 / problem_size[d], nd)
    dt = stable_heat_dt(alpha, cfl, dx)

    local_u = LocalHaloArray(Float64, problem_size, halo_width_value; boundary_condition=:periodic)
    local_tmp = similar(local_u)

    threaded_tile_size = tile_size_from_owned_size(problem_size, tile_dims)
    threaded_u = ThreadedHaloArray(Float64, threaded_tile_size, halo_width_value;
        dims=tile_dims, boundary_condition=:periodic)
    threaded_tmp = similar(threaded_u)

    println("Local/threaded heat solver benchmark")
    println("  ndims:                ", nd)
    println("  problem size:         ", problem_size)
    println("  threaded tile dims:   ", tile_dims)
    println("  threaded tile size:   ", threaded_tile_size)
    println("  Julia threads:        ", nthreads())
    println("  halo width:           ", halo_width_value)
    println("  steps per sample:     ", steps)
    println("  alpha:                ", alpha)
    println("  dt:                   ", dt)
    println("  samples:              ", samples)
    println("  warmups:              ", warmups)
    println("  timer:                ", timer)
    println()

    compare_backends!(local_u, local_tmp, threaded_u, threaded_tmp, problem_size;
        steps=steps, alpha=alpha, dt=dt, dx=dx)

    fill_heat_initial!(local_u, problem_size)
    fill_heat_initial!(threaded_u, problem_size)

    metadata = Dict{String,Any}(
        "ndims" => nd,
        "global_size" => joined_tuple(problem_size),
        "tile_dims" => joined_tuple(tile_dims),
        "threads" => nthreads(),
        "halo_width" => halo_width_value,
        "steps" => steps,
        "alpha" => alpha,
        "dt" => dt,
        "timer" => string(timer),
    )
    rows = Dict{String,Any}[]

    benchmark_case!(rows, "heat_solver_local_threaded", "local_haloarray", () -> begin
        solve_heat_steps!(local_u, local_tmp; steps=steps, alpha=alpha, dt=dt, dx=dx)
    end, samples, warmups, metadata, timer)

    benchmark_case!(rows, "heat_solver_local_threaded", "local_synchronize_halo", () -> begin
        synchronize_halo!(local_u)
    end, samples, warmups, metadata, timer)

    benchmark_case!(rows, "heat_solver_local_threaded", "local_heat_step_only", () -> begin
        heat_step!(local_tmp, local_u, alpha, dt, dx)
    end, samples, warmups, metadata, timer)

    benchmark_case!(rows, "heat_solver_local_threaded", "local_single_step", () -> begin
        heat_single_step!(local_tmp, local_u; alpha=alpha, dt=dt, dx=dx)
    end, samples, warmups, metadata, timer)

    benchmark_case!(rows, "heat_solver_local_threaded", "threaded_haloarray", () -> begin
        solve_heat_steps!(threaded_u, threaded_tmp; steps=steps, alpha=alpha, dt=dt, dx=dx)
    end, samples, warmups, metadata, timer)

    benchmark_case!(rows, "heat_solver_local_threaded", "threaded_synchronize_halo", () -> begin
        synchronize_halo!(threaded_u)
    end, samples, warmups, metadata, timer)

    benchmark_case!(rows, "heat_solver_local_threaded", "threaded_heat_step_only", () -> begin
        heat_step!(threaded_tmp, threaded_u, alpha, dt, dx)
    end, samples, warmups, metadata, timer)

    benchmark_case!(rows, "heat_solver_local_threaded", "threaded_single_step", () -> begin
        heat_single_step!(threaded_tmp, threaded_u; alpha=alpha, dt=dt, dx=dx)
    end, samples, warmups, metadata, timer)

    maybe_write_csv(options, rows)
end

main()
