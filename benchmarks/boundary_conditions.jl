include("common.jl")

using Base.Threads: nthreads

function boundary_modes(options)
    raw = option_string(options, "modes", "repeating,reflecting,antireflecting,periodic")
    modes = Symbol.(strip.(split(raw, ",")))
    allowed = (:repeating, :reflecting, :antireflecting, :periodic)
    unknown = setdiff(modes, allowed)
    isempty(unknown) || error("Unknown boundary modes: $(join(unknown, ", "))")
    return modes
end

function mode_topology(::Val{N}, comm, mode::Symbol) where {N}
    is_periodic = mode === :periodic
    dims = MPI.Dims_create(MPI.Comm_size(comm), ntuple(_ -> 0, Val(N))) |> Tuple
    return CartesianTopology(comm, Tuple(Int.(dims)); periodic=ntuple(_ -> is_periodic, Val(N)))
end

function make_threaded_boundary_halo(::Val{N}, owned_size, halo_width, tile_dims, mode::Symbol) where {N}
    tile_size = tile_size_from_owned_size(owned_size, tile_dims)
    return ThreadedHaloArray(Float64, tile_size, halo_width; dims=tile_dims, boundary_condition=mode)
end

function local_case!(rows, name, f, samples, warmups, timer, metadata)
    times = benchmark_times!(f, samples, warmups, timer)
    allocations = allocation_bytes!(f, warmups)
    print_summary(name, times; allocations)
    push!(rows, benchmark_record("boundary_conditions", name, times; metadata=copy(metadata), allocations))
    return nothing
end

function mpi_case!(rows, name, f, samples, warmups, metadata; comm, rank)
    times = benchmark_times!(f, samples, warmups; comm)
    allocations = allocation_bytes!(f, warmups; comm)
    if rank == 0
        print_summary(name, times; allocations)
        push!(rows, benchmark_record("boundary_conditions", name, times; metadata=copy(metadata), allocations))
    end
    return nothing
end

function run_rank_local_cases!(rows, ::Val{N}, owned_size, halo_width, tile_dims, modes, samples, warmups, timer, metadata) where {N}
    for mode in modes
        local_u = LocalHaloArray(Float64, owned_size, halo_width; boundary_condition=mode)
        threaded_u = make_threaded_boundary_halo(Val(N), owned_size, halo_width, tile_dims, mode)

        fill_benchmark_data!(local_u)
        fill_benchmark_data!(threaded_u)

        mode_metadata = merge(copy(metadata), Dict{String,Any}("mode" => string(mode)))

        local_case!(rows, "local_$(mode)_boundary_condition", () -> boundary_condition!(local_u),
            samples, warmups, timer, mode_metadata)
        local_case!(rows, "local_$(mode)_synchronize_halo", () -> synchronize_halo!(local_u),
            samples, warmups, timer, mode_metadata)
        local_case!(rows, "threaded_$(mode)_boundary_condition", () -> boundary_condition!(threaded_u),
            samples, warmups, timer, mode_metadata)
        local_case!(rows, "threaded_$(mode)_synchronize_halo", () -> synchronize_halo!(threaded_u),
            samples, warmups, timer, mode_metadata)
    end
    return nothing
end

function run_mpi_cases!(rows, ::Val{N}, comm, rank, owned_size, halo_width, modes, samples, warmups, metadata) where {N}
    for mode in modes
        topology = mode_topology(Val(N), comm, mode)
        mpi_u = HaloArray(Float64, owned_size, halo_width, topology; boundary_condition=mode)
        fill_benchmark_data!(mpi_u)

        mode_metadata = merge(copy(metadata), Dict{String,Any}(
            "mode" => string(mode),
            "topology" => joined_tuple(topology.dims),
        ))

        mpi_case!(rows, "mpi_$(mode)_boundary_condition", () -> boundary_condition!(mpi_u),
            samples, warmups, mode_metadata; comm, rank)
        mpi_case!(rows, "mpi_$(mode)_synchronize_halo", () -> synchronize_halo!(mpi_u),
            samples, warmups, mode_metadata; comm, rank)
    end
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
    tile_dims = option_tuple(options, "tile-dims", ndims, 2)
    modes = boundary_modes(options)
    timer = Symbol(option_string(options, "timer", "manual"))

    if rank == 0
        println("Boundary-condition benchmark")
        println("  ranks:       ", nproc)
        println("  ndims:       ", ndims)
        println("  owned size:  ", owned_size)
        println("  tile dims:   ", tile_dims)
        println("  tile size:   ", tile_size_from_owned_size(owned_size, tile_dims))
        println("  Julia threads: ", nthreads())
        println("  halo width:  ", halo_width)
        println("  modes:       ", join(modes, ", "))
        println("  samples:     ", samples)
        println("  warmups:     ", warmups)
        println("  local/thread timer: ", timer)
        println()
    end

    metadata = Dict{String,Any}(
        "ranks" => nproc,
        "ndims" => ndims,
        "owned_size" => joined_tuple(owned_size),
        "halo_width" => halo_width,
        "tile_dims" => joined_tuple(tile_dims),
        "tile_size" => joined_tuple(tile_size_from_owned_size(owned_size, tile_dims)),
        "threads" => nthreads(),
    )
    rows = Dict{String,Any}[]

    if rank == 0
        run_rank_local_cases!(rows, Val(ndims), owned_size, halo_width, tile_dims, modes, samples, warmups, timer, metadata)
    end

    MPI.Barrier(comm)
    run_mpi_cases!(rows, Val(ndims), comm, rank, owned_size, halo_width, modes, samples, warmups, metadata)

    if rank == 0
        maybe_write_csv(options, rows)
    end

    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
