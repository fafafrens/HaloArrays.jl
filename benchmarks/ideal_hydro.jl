include("common.jl")
include(joinpath(@__DIR__, "..", "examples", "ideal_hydro_common.jl"))

using Printf
using Test

function _option_cases(options)
    raw = lowercase(option_string(options, "cases", "local,threaded,mpi"))
    return Set(Symbol.(split(raw, ",")))
end

function _ideal_hydro_local_state(nx, ny, halo, boundary_condition)
    return LocalMultiHaloArray(Float64, (nx, ny), halo;
        boundary_conditions=ideal_hydro_boundary_conditions(boundary_condition))
end

function _ideal_hydro_threaded_state(nx, ny, halo, tile_dims, boundary_condition)
    all(d -> (nx, ny)[d] % tile_dims[d] == 0, 1:2) ||
        error("--nx and --ny must be divisible by --tile-dims")
    tile_size = (nx ÷ tile_dims[1], ny ÷ tile_dims[2])
    return ThreadedMultiHaloArray(Float64, tile_size, halo;
        dims=tile_dims,
        boundary_conditions=ideal_hydro_boundary_conditions(boundary_condition))
end

function _ideal_hydro_mpi_state(nx, ny, halo, boundary_condition)
    topology = CartesianTopology(MPI.COMM_WORLD, (0, 0);
        periodic=(boundary_condition == :periodic, boundary_condition == :periodic))

    nx % topology.dims[1] == 0 ||
        error("--nx=$nx must be divisible by MPI topology dim $(topology.dims[1])")
    ny % topology.dims[2] == 0 ||
        error("--ny=$ny must be divisible by MPI topology dim $(topology.dims[2])")

    owned_cells = (nx ÷ topology.dims[1], ny ÷ topology.dims[2])
    return MultiHaloArray(Float64, owned_cells, halo, topology;
        boundary_conditions=ideal_hydro_boundary_conditions(boundary_condition))
end

function _run_ideal_hydro_sample!(make_state; steps, cfl, gamma)
    u = make_state()
    _, _, initial, final = run_ideal_hydro_2d!(
        u; gamma, cfl, steps, adaptive=false, reltol=1e-5, abstol=1e-7)

    @assert final.min_rho > 0
    @assert final.min_pressure > 0
    @assert isfinite(final.max_speed)
    @assert isapprox(final.mass, initial.mass; rtol=1e-10, atol=1e-10)
    @assert isapprox(final.energy, initial.energy; rtol=1e-10, atol=1e-10)
    return final
end

function _inference_ok(f)
    try
        @inferred f()
        return true
    catch
        return false
    end
end

function _diagnose_ideal_hydro(label, make_state, gamma; comm=nothing, rank=0)
    u = make_state()
    du = similar(u)
    nx, ny = global_size(u[:rho])
    p = (; gamma, dx=1 / nx, dy=1 / ny)

    fill_pressure_bump!(u; gamma)
    ideal_hydro_rhs!(du, u, p, 0.0)
    max_signal_speed(u, gamma)

    fill_inferred = _inference_ok(() -> fill_pressure_bump!(u; gamma))
    rhs_inferred = _inference_ok(() -> ideal_hydro_rhs!(du, u, p, 0.0))
    speed_inferred = _inference_ok(() -> max_signal_speed(u, gamma))

    fill_alloc = allocation_bytes!(() -> fill_pressure_bump!(u; gamma), 1; comm)
    rhs_alloc = allocation_bytes!(() -> ideal_hydro_rhs!(du, u, p, 0.0), 1; comm)
    speed_alloc = allocation_bytes!(() -> max_signal_speed(u, gamma), 1; comm)

    if rank == 0
        @printf("%-26s alloc fill=%d B rhs=%d B max_speed=%d B\n",
            label * "_diagnostics", fill_alloc, rhs_alloc, speed_alloc)
        @printf("%-26s inferred fill=%s rhs=%s max_speed=%s\n",
            label * "_diagnostics", fill_inferred, rhs_inferred, speed_inferred)
    end

    return (; fill_alloc, rhs_alloc, speed_alloc, fill_inferred, rhs_inferred, speed_inferred)
end

function _benchmark_ideal_hydro_case!(rows, name, make_state, metadata, options;
        comm=nothing, rank=0)
    samples = option_int(options, "samples", 5)
    warmups = option_int(options, "warmups", 1)
    steps = option_int(options, "steps", 6)
    cfl = parse(Float64, option_string(options, "cfl", "0.25"))
    gamma = parse(Float64, option_string(options, "gamma", "1.4"))

    sample!() = _run_ideal_hydro_sample!(make_state; steps, cfl, gamma)
    times = benchmark_times!(sample!, samples, warmups; comm)
    alloc = allocation_bytes!(sample!, warmups; comm)

    if rank == 0
        print_summary(name, times; allocations=alloc)
        push!(rows, benchmark_record("ideal_hydro", name, times;
            metadata=copy(metadata), allocations=alloc))
    end

    if option_bool(options, "diagnostics", true)
        _diagnose_ideal_hydro(name, make_state, gamma; comm, rank)
    end

    return nothing
end

function main()
    options = parse_args()
    comm = ensure_mpi()
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)
    cases = _option_cases(options)

    nx = option_int(options, "nx", 128)
    ny = option_int(options, "ny", 128)
    halo = option_int(options, "halo", 1)
    steps = option_int(options, "steps", 6)
    cfl = parse(Float64, option_string(options, "cfl", "0.25"))
    gamma = parse(Float64, option_string(options, "gamma", "1.4"))
    boundary_condition = Symbol(option_string(options, "boundary", "periodic"))
    tile_dims = option_tuple(options, "tile-dims", 2, (2, 2))

    if rank == 0
        println("Ideal hydro benchmark")
        println("  global size:          ", (nx, ny))
        println("  steps per sample:     ", steps)
        println("  samples:              ", option_int(options, "samples", 5))
        println("  warmups:              ", option_int(options, "warmups", 1))
        println("  cfl:                  ", cfl)
        println("  gamma:                ", gamma)
        println("  boundary:             ", boundary_condition)
        println("  cases:                ", join(sort!(collect(cases)), ","))
        println("  ranks:                ", nproc)
        println("  Julia threads/rank:   ", Threads.nthreads())
        println("  threaded tile dims:   ", tile_dims)
        println()
    end

    metadata = Dict{String,Any}(
        "global_size" => joined_tuple((nx, ny)),
        "steps" => steps,
        "cfl" => cfl,
        "gamma" => gamma,
        "boundary" => boundary_condition,
        "ranks" => nproc,
        "threads_per_rank" => Threads.nthreads(),
        "tile_dims" => joined_tuple(tile_dims),
    )
    rows = Dict{String,Any}[]

    if rank == 0 && (:local in cases)
        _benchmark_ideal_hydro_case!(
            rows,
            "local",
            () -> _ideal_hydro_local_state(nx, ny, halo, boundary_condition),
            metadata,
            options,
        )
    end

    if rank == 0 && (:threaded in cases)
        _benchmark_ideal_hydro_case!(
            rows,
            "threaded",
            () -> _ideal_hydro_threaded_state(nx, ny, halo, tile_dims, boundary_condition),
            metadata,
            options,
        )
    end

    if :mpi in cases
        _benchmark_ideal_hydro_case!(
            rows,
            "mpi",
            () -> _ideal_hydro_mpi_state(nx, ny, halo, boundary_condition),
            metadata,
            options;
            comm,
            rank,
        )
    end

    rank == 0 && maybe_write_csv(options, rows)
    MPI.Barrier(comm)
    return nothing
end

main()
