include("common.jl")

function make_halo(::Val{N}, owned_size, halo_width, topology) where {N}
    return HaloArray(Float64, owned_size, halo_width, topology; boundary_condition=:periodic)
end

function exchange_methods()
    return Dict(
        "exchange" => HaloArrays.halo_exchange_wait!,
        "blocking" => halo_exchange!,
        "waitall" => HaloArrays.halo_exchange_waitall!,
        "waitall_unsafe" => HaloArrays.halo_exchange_waitall_unsafe!,
        "async" => HaloArrays.halo_exchange_async!,
        "async_unsafe" => HaloArrays.halo_exchange_async_unsafe!,
        "public_split" => h -> begin
            start_halo_exchange!(h)
            finish_halo_exchange!(h)
        end,
        "split_async" => h -> begin
            HaloArrays.start_halo_exchange_async!(h)
            HaloArrays.end_halo_exchange_wait!(h)
        end,
        "split_async_unsafe" => h -> begin
            HaloArrays.start_halo_exchange_async_unsafe!(h)
            HaloArrays.end_halo_exchange_async_wait_unsafe!(h)
        end,
    )
end

function selected_methods(options)
    methods = exchange_methods()
    raw = get(options, "methods", "")
    isempty(raw) && return [
        "blocking",
        "waitall",
        "waitall_unsafe",
        "async",
        "async_unsafe",
        "public_split",
        "split_async",
        "split_async_unsafe",
    ]

    names = String.(split(raw, ","))
    unknown = setdiff(names, collect(keys(methods)))
    isempty(unknown) || error("Unknown methods: $(join(unknown, ", "))")
    return names
end

function sanity_check!(halo, names, methods, comm)
    fill_benchmark_data!(halo)
    methods["waitall_unsafe"](halo)
    reference = copy(parent(halo))

    ok = true
    for name in names
        fill_benchmark_data!(halo)
        methods[name](halo)
        ok &= parent(halo) == reference
    end

    all_ok = MPI.Allreduce(ok, &, comm)
    all_ok || error("At least one exchange method produced a different halo result")
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

    topology = make_periodic_topology(comm, ndims)
    halo = make_halo(Val(ndims), owned_size, halo_width, topology)
    methods = exchange_methods()
    names = selected_methods(options)

    sanity_check!(halo, names, methods, comm)

    if rank == 0
        println("Halo exchange benchmark")
        println("  ranks:       ", nproc)
        println("  topology:    ", topology.dims)
        println("  ndims:       ", ndims)
        println("  owned size:  ", owned_size)
        println("  halo width:  ", halo_width)
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

    for name in names
        fill_benchmark_data!(halo)
        times = benchmark_times!(samples, warmups; comm=comm) do
            methods[name](halo)
        end
        if rank == 0
            print_summary(name, times)
            push!(rows, benchmark_record("halo_exchange", name, times; metadata=copy(metadata)))
        end
    end

    if rank == 0
        maybe_write_csv(options, rows)
    end

    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
