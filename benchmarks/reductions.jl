include("common.jl")

function benchmark_case!(rows, name, f, samples, warmups, metadata; comm, rank)
    sink = Ref{Any}()
    times = benchmark_times!(samples, warmups; comm=comm) do
        sink[] = f()
    end

    if rank == 0
        print_summary(name, times)
        push!(rows, benchmark_record("reductions", name, times; metadata=copy(metadata)))
    end
    return sink[]
end

function make_threaded_halo(::Val{N}, owned_size, halo_width, tile_dims) where {N}
    tile_size = tile_size_from_owned_size(owned_size, tile_dims)
    return ThreadedHaloArray(Float64, tile_size, halo_width; dims=tile_dims, boundary_condition=:periodic)
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

    topology = make_periodic_topology(comm, ndims)

    mpi_u = HaloArray(Float64, owned_size, halo_width, topology; boundary_condition=:periodic)
    mpi_v = similar(mpi_u)
    local_u = LocalHaloArray(Float64, owned_size, halo_width; boundary_condition=:periodic)
    local_v = similar(local_u)
    threaded_u = make_threaded_halo(Val(ndims), owned_size, halo_width, tile_dims)
    threaded_v = similar(threaded_u)

    for halo in (mpi_u, mpi_v, local_u, local_v, threaded_u, threaded_v)
        fill_benchmark_data!(halo)
    end

    mpi_fields = MultiHaloArray((; u=mpi_u, v=mpi_v))
    local_fields = MultiHaloArray((; u=local_u, v=local_v))
    threaded_fields = MultiHaloArray((; u=threaded_u, v=threaded_v))

    if rank == 0
        println("Reduction benchmark")
        println("  ranks:       ", nproc)
        println("  topology:    ", topology.dims)
        println("  ndims:       ", ndims)
        println("  owned size:  ", owned_size)
        println("  halo width:  ", halo_width)
        println("  tile dims:   ", tile_dims)
        println("  samples:     ", samples)
        println("  warmups:     ", warmups)
        println()
    end

    metadata = Dict{String,Any}(
        "ranks" => nproc,
        "topology" => joined_tuple(topology.dims),
        "ndims" => ndims,
        "owned_size" => joined_tuple(owned_size),
        "halo_width" => halo_width,
        "tile_dims" => joined_tuple(tile_dims),
    )
    rows = Dict{String,Any}[]

    cases = [
        ("mpi_mapreduce", () -> mapreduce(abs2, +, mpi_u)),
        ("mpi_all", () -> all(isfinite, mpi_u)),
        ("mpi_any", () -> any(x -> x < 0, mpi_u)),
        ("mpi_multi_mapreduce", () -> mapreduce(abs2, +, mpi_fields)),
        ("mpi_multi_all", () -> all(isfinite, mpi_fields)),
        ("mpi_multi_any", () -> any(x -> x < 0, mpi_fields)),
        ("local_mapreduce", () -> mapreduce(abs2, +, local_u)),
        ("local_all", () -> all(isfinite, local_u)),
        ("local_any", () -> any(x -> x < 0, local_u)),
        ("local_multi_mapreduce", () -> mapreduce(abs2, +, local_fields)),
        ("local_multi_all", () -> all(isfinite, local_fields)),
        ("local_multi_any", () -> any(x -> x < 0, local_fields)),
        ("threaded_mapreduce", () -> mapreduce(abs2, +, threaded_u)),
        ("threaded_all", () -> all(isfinite, threaded_u)),
        ("threaded_any", () -> any(x -> x < 0, threaded_u)),
        ("threaded_multi_mapreduce", () -> mapreduce(abs2, +, threaded_fields)),
        ("threaded_multi_all", () -> all(isfinite, threaded_fields)),
        ("threaded_multi_any", () -> any(x -> x < 0, threaded_fields)),
    ]

    for (name, f) in cases
        benchmark_case!(rows, name, f, samples, warmups, metadata; comm=comm, rank=rank)
    end

    if rank == 0
        maybe_write_csv(options, rows)
    end

    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
