include("common.jl")

function remove_output!(base, comm)
    if MPI.Comm_rank(comm) == 0
        rm(base * ".h5"; force=true)
    end
    MPI.Barrier(comm)
    return nothing
end

function benchmark_case!(rows, name, f, samples, warmups, metadata; comm, rank)
    sink = Ref{Any}()
    times = benchmark_times!(samples, warmups; comm=comm) do
        sink[] = f()
    end

    if rank == 0
        print_summary(name, times)
        push!(rows, benchmark_record("gather_hdf5", name, times; metadata=copy(metadata)))
    end
    return sink[]
end

function main()
    comm = ensure_mpi()
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)

    options = parse_args()
    ndims = option_int(options, "ndims", 2)
    halo_width = option_int(options, "halo", 1)
    samples = option_int(options, "samples", 10)
    warmups = option_int(options, "warmups", 2)
    owned_size = option_owned_size(options, ndims, 64)
    output = option_string(options, "output", joinpath(tempdir(), "haloarrays_bench"))

    topology = make_periodic_topology(comm, ndims)
    halo = HaloArray(Float64, owned_size, halo_width, topology; boundary_condition=:periodic)
    fill_benchmark_data!(halo)

    gather_base = output * "_gather_save"
    append_base = output * "_append"

    remove_output!(gather_base, comm)
    remove_output!(append_base, comm)

    if rank == 0
        println("Gather/HDF5 benchmark")
        println("  ranks:       ", nproc)
        println("  topology:    ", topology.dims)
        println("  ndims:       ", ndims)
        println("  owned size:  ", owned_size)
        println("  halo width:  ", halo_width)
        println("  output base: ", output)
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
    )
    rows = Dict{String,Any}[]

    benchmark_case!(rows, "gather_haloarray", () -> begin
        gather_haloarray(halo; root=0)
    end, samples, warmups, metadata; comm=comm, rank=rank)

    benchmark_case!(rows, "gather_and_save_haloarray", () -> begin
        remove_output!(gather_base, comm)
        gather_and_save_haloarray(gather_base, halo; root=0)
    end, samples, warmups, metadata; comm=comm, rank=rank)

    benchmark_case!(rows, "append_haloarray_to_file", () -> begin
        remove_output!(append_base, comm)
        append_haloarray_to_file!(append_base, "field", halo)
    end, samples, warmups, metadata; comm=comm, rank=rank)

    if rank == 0
        maybe_write_csv(options, rows)
    end

    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
