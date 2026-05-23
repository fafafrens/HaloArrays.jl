include("common.jl")

function diagnostic_topology(::Val{N}, comm, periodic::Bool) where {N}
    dims = MPI.Dims_create(MPI.Comm_size(comm), ntuple(_ -> 0, Val(N))) |> Tuple
    return CartesianTopology(comm, Tuple(Int.(dims)); periodic=ntuple(_ -> periodic, Val(N)))
end

function diagnostic_case!(rows, name, f, samples, warmups, metadata; comm, rank)
    times = benchmark_times!(f, samples, warmups; comm)
    allocations = allocation_bytes!(f, warmups; comm)
    if rank == 0
        print_summary(name, times; allocations)
        push!(rows, benchmark_record("mpi_diagnostics", name, times; metadata=copy(metadata), allocations))
    end
    return nothing
end

function split_exchange!(halo)
    start_halo_exchange!(halo)
    finish_halo_exchange!(halo)
    return halo
end

function run_periodic_cases!(rows, ::Val{N}, comm, rank, owned_size, halo_width, samples, warmups, metadata) where {N}
    topology = diagnostic_topology(Val(N), comm, true)
    halo = HaloArray(Float64, owned_size, halo_width, topology; boundary_condition=:periodic)
    fill_benchmark_data!(halo)

    case_metadata = merge(copy(metadata), Dict{String,Any}(
        "topology" => joined_tuple(topology.dims),
        "mode" => "periodic",
    ))

    diagnostic_case!(rows, "periodic_halo_exchange", () -> halo_exchange!(halo),
        samples, warmups, case_metadata; comm, rank)
    diagnostic_case!(rows, "periodic_start_finish_exchange", () -> split_exchange!(halo),
        samples, warmups, case_metadata; comm, rank)
    diagnostic_case!(rows, "periodic_synchronize_halo", () -> synchronize_halo!(halo),
        samples, warmups, case_metadata; comm, rank)
    return nothing
end

function run_physical_cases!(rows, ::Val{N}, comm, rank, owned_size, halo_width, boundary_mode, samples, warmups, metadata) where {N}
    topology = diagnostic_topology(Val(N), comm, false)
    halo = HaloArray(Float64, owned_size, halo_width, topology; boundary_condition=boundary_mode)
    fill_benchmark_data!(halo)

    case_metadata = merge(copy(metadata), Dict{String,Any}(
        "topology" => joined_tuple(topology.dims),
        "mode" => string(boundary_mode),
    ))

    diagnostic_case!(rows, "physical_halo_exchange", () -> halo_exchange!(halo),
        samples, warmups, case_metadata; comm, rank)
    diagnostic_case!(rows, "physical_boundary_condition", () -> boundary_condition!(halo),
        samples, warmups, case_metadata; comm, rank)
    diagnostic_case!(rows, "physical_synchronize_halo", () -> synchronize_halo!(halo),
        samples, warmups, case_metadata; comm, rank)
    return nothing
end

function main()
    comm = ensure_mpi()
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)

    options = parse_args()
    ndims = option_int(options, "ndims", 2)
    halo_width = option_int(options, "halo", 1)
    samples = option_int(options, "samples", 30)
    warmups = option_int(options, "warmups", 5)
    owned_size = option_owned_size(options, ndims, 64)
    boundary_mode = Symbol(option_string(options, "boundary", "repeating"))

    if boundary_mode === :periodic
        error("--boundary is for physical boundaries and must not be periodic")
    end

    if rank == 0
        println("MPI diagnostics benchmark")
        println("  ranks:       ", nproc)
        println("  ndims:       ", ndims)
        println("  owned size:  ", owned_size)
        println("  halo width:  ", halo_width)
        println("  physical boundary: ", boundary_mode)
        println("  samples:     ", samples)
        println("  warmups:     ", warmups)
        println()
    end

    metadata = Dict{String,Any}(
        "ranks" => nproc,
        "ndims" => ndims,
        "owned_size" => joined_tuple(owned_size),
        "halo_width" => halo_width,
    )
    rows = Dict{String,Any}[]

    run_periodic_cases!(rows, Val(ndims), comm, rank, owned_size, halo_width, samples, warmups, metadata)
    run_physical_cases!(rows, Val(ndims), comm, rank, owned_size, halo_width, boundary_mode, samples, warmups, metadata)

    if rank == 0
        maybe_write_csv(options, rows)
    end

    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
