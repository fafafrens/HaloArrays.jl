include("common.jl")
include("heat_solver_common.jl")

using Base.Threads: nthreads

function compare_backends!(mpi_u, mpi_tmp, local_u, local_tmp, threaded_u, threaded_tmp,
        problem_size; steps, alpha, dt, dx, comm, rank)
    fill_heat_initial!(mpi_u, problem_size)
    solve_heat_steps!(mpi_u, mpi_tmp; steps=steps, alpha=alpha, dt=dt, dx=dx)
    mpi_solution = gather_haloarray(mpi_u; root=0)

    if rank == 0
        fill_heat_initial!(local_u, problem_size)
        fill_heat_initial!(threaded_u, problem_size)
        solve_heat_steps!(local_u, local_tmp; steps=steps, alpha=alpha, dt=dt, dx=dx)
        solve_heat_steps!(threaded_u, threaded_tmp; steps=steps, alpha=alpha, dt=dt, dx=dx)

        local_solution = snapshot_interior(local_u)
        threaded_solution = snapshot_interior(threaded_u)
        mpi_error = maximum(abs.(mpi_solution .- local_solution))
        threaded_error = maximum(abs.(threaded_solution .- local_solution))
        println("  MPI vs Local max error:      ", mpi_error)
        println("  Threaded vs Local max error: ", threaded_error)
        println()
    end

    MPI.Barrier(comm)
    return nothing
end

function main()
    comm = ensure_mpi()
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)
    if nproc == 1
        error("benchmarks/heat_solver_mpi.jl is an MPI benchmark; run it with mpiexec -n N where N > 1. Use benchmarks/heat_solver_local_threaded.jl for single-process local/threaded comparisons.")
    end

    options = parse_args()
    nd = option_int(options, "ndims", 2)
    halo_width_value = option_int(options, "halo", 1)
    samples = option_int(options, "samples", 20)
    warmups = option_int(options, "warmups", 3)
    steps = option_int(options, "steps", 10)
    alpha = parse(Float64, option_string(options, "alpha", "0.01"))
    cfl = parse(Float64, option_string(options, "cfl", "0.4"))
    local_timer = Symbol(option_string(options, "timer", "manual"))
    local_timer in (:manual, :benchmarktools) || error("--timer must be manual or benchmarktools")
    owned_size_per_rank = option_owned_size(options, nd, 64)
    tile_dims = option_tuple(options, "tile-dims", nd, 2)

    topology = make_periodic_topology(comm, nd)
    problem_size = problem_size_from_topology(owned_size_per_rank, topology.dims)
    dx = ntuple(d -> 1.0 / problem_size[d], nd)
    dt = stable_heat_dt(alpha, cfl, dx)

    mpi_u = HaloArray(Float64, owned_size_per_rank, halo_width_value, topology; boundary_condition=:periodic)
    mpi_tmp = similar(mpi_u)

    local_u = local_tmp = threaded_u = threaded_tmp = nothing
    if rank == 0
        local_u = LocalHaloArray(Float64, problem_size, halo_width_value; boundary_condition=:periodic)
        local_tmp = similar(local_u)

        threaded_tile_size = tile_size_from_owned_size(problem_size, tile_dims)
        threaded_u = ThreadedHaloArray(Float64, threaded_tile_size, halo_width_value;
            dims=tile_dims, boundary_condition=:periodic)
        threaded_tmp = similar(threaded_u)
    end

    if rank == 0
        println("MPI heat solver backend benchmark")
        println("  ranks:                ", nproc)
        println("  topology:             ", topology.dims)
        println("  ndims:                ", nd)
        println("  MPI owned size/rank:  ", owned_size_per_rank)
        println("  global problem size:  ", problem_size)
        println("  threaded tile dims:   ", tile_dims)
        println("  Julia threads:        ", nthreads())
        println("  halo width:           ", halo_width_value)
        println("  steps per sample:     ", steps)
        println("  alpha:                ", alpha)
        println("  dt:                   ", dt)
        println("  samples:              ", samples)
        println("  warmups:              ", warmups)
        println("  local/thread timer:   ", local_timer)
        println()
    end

    compare_backends!(mpi_u, mpi_tmp, local_u, local_tmp, threaded_u, threaded_tmp, problem_size;
        steps=steps, alpha=alpha, dt=dt, dx=dx, comm=comm, rank=rank)

    fill_heat_initial!(mpi_u, problem_size)
    rank == 0 && fill_heat_initial!(local_u, problem_size)
    rank == 0 && fill_heat_initial!(threaded_u, problem_size)

    metadata = Dict{String,Any}(
        "ranks" => nproc,
        "topology" => joined_tuple(topology.dims),
        "ndims" => nd,
        "owned_size_per_rank" => joined_tuple(owned_size_per_rank),
        "global_size" => joined_tuple(problem_size),
        "tile_dims" => joined_tuple(tile_dims),
        "threads" => nthreads(),
        "halo_width" => halo_width_value,
        "steps" => steps,
        "alpha" => alpha,
        "dt" => dt,
        "local_timer" => string(local_timer),
    )
    rows = Dict{String,Any}[]

    benchmark_case!(rows, "heat_solver_mpi", "mpi_haloarray", () -> begin
        solve_heat_steps!(mpi_u, mpi_tmp; steps=steps, alpha=alpha, dt=dt, dx=dx)
    end, samples, warmups, metadata; comm=comm, rank=rank)

    benchmark_case!(rows, "heat_solver_mpi", "mpi_synchronize_halo", () -> begin
        synchronize_halo!(mpi_u)
    end, samples, warmups, metadata; comm=comm, rank=rank)

    benchmark_case!(rows, "heat_solver_mpi", "mpi_heat_step_only", () -> begin
        heat_step!(mpi_tmp, mpi_u, alpha, dt, dx)
    end, samples, warmups, metadata; comm=comm, rank=rank)

    benchmark_case!(rows, "heat_solver_mpi", "mpi_single_step", () -> begin
        heat_single_step!(mpi_tmp, mpi_u; alpha=alpha, dt=dt, dx=dx)
    end, samples, warmups, metadata; comm=comm, rank=rank)

    if rank == 0
        benchmark_case!(rows, "heat_solver_mpi", "local_haloarray", () -> begin
            solve_heat_steps!(local_u, local_tmp; steps=steps, alpha=alpha, dt=dt, dx=dx)
        end, samples, warmups, metadata, local_timer)

        benchmark_case!(rows, "heat_solver_mpi", "local_synchronize_halo", () -> begin
            synchronize_halo!(local_u)
        end, samples, warmups, metadata, local_timer)

        benchmark_case!(rows, "heat_solver_mpi", "local_heat_step_only", () -> begin
            heat_step!(local_tmp, local_u, alpha, dt, dx)
        end, samples, warmups, metadata, local_timer)

        benchmark_case!(rows, "heat_solver_mpi", "local_single_step", () -> begin
            heat_single_step!(local_tmp, local_u; alpha=alpha, dt=dt, dx=dx)
        end, samples, warmups, metadata, local_timer)

        benchmark_case!(rows, "heat_solver_mpi", "threaded_haloarray", () -> begin
            solve_heat_steps!(threaded_u, threaded_tmp; steps=steps, alpha=alpha, dt=dt, dx=dx)
        end, samples, warmups, metadata, local_timer)

        benchmark_case!(rows, "heat_solver_mpi", "threaded_synchronize_halo", () -> begin
            synchronize_halo!(threaded_u)
        end, samples, warmups, metadata, local_timer)

        benchmark_case!(rows, "heat_solver_mpi", "threaded_heat_step_only", () -> begin
            heat_step!(threaded_tmp, threaded_u, alpha, dt, dx)
        end, samples, warmups, metadata, local_timer)

        benchmark_case!(rows, "heat_solver_mpi", "threaded_single_step", () -> begin
            heat_single_step!(threaded_tmp, threaded_u; alpha=alpha, dt=dt, dx=dx)
        end, samples, warmups, metadata, local_timer)

        maybe_write_csv(options, rows)
    end

    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
