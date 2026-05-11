using MPI
using HaloArrays

function parse_args(args)
    options = Dict{String,String}()
    for arg in args
        startswith(arg, "--") || continue
        key_value = split(arg[3:end], "=", limit=2)
        if length(key_value) == 1
            options[key_value[1]] = "true"
        else
            options[key_value[1]] = key_value[2]
        end
    end
    return options
end

function option_int(options, name, default)
    return parse(Int, get(options, name, string(default)))
end

function option_tuple(options, name, ndims, default)
    raw = get(options, name, "")
    if isempty(raw)
        return ntuple(_ -> default, Val(ndims))
    end

    values = parse.(Int, split(raw, ","))
    length(values) == ndims || error("--$name must have $ndims comma-separated values")
    return Tuple(values)
end

function median_value(values)
    sorted = sort(values)
    n = length(sorted)
    mid = n ÷ 2
    return isodd(n) ? sorted[mid + 1] : (sorted[mid] + sorted[mid + 1]) / 2
end

function quantile_value(values, q)
    sorted = sort(values)
    index = clamp(ceil(Int, q * length(sorted)), 1, length(sorted))
    return sorted[index]
end

function reset_halo!(halo)
    fill!(parent(halo), -1.0)
    fill_from_global_indices!(halo) do I
        return sum((d + 10) * I[d] for d in 1:length(I))
    end
    return halo
end

function make_topology(comm, ndims)
    nproc = MPI.Comm_size(comm)
    dims = MPI.Dims_create(nproc, ntuple(_ -> 0, Val(ndims)))
    return CartesianTopology(comm, Tuple(Int.(dims)); periodic=ntuple(_ -> true, Val(ndims)))
end

function make_halo(::Val{N}, local_size, halo_width, topology) where {N}
    return HaloArray(Float64, local_size, halo_width, topology; boundary_condition=:periodic)
end

function exchange_methods()
    return Dict(
        "exchange" => halo_exchange!,
        "wait" => halo_exchange_wait!,
        "waitall" => halo_exchange_waitall!,
        "waitall_unsafe" => halo_exchange_waitall_unsafe!,
        "async" => halo_exchange_async!,
        "async_unsafe" => halo_exchange_async_unsafe!,
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
        "waitall",
        "waitall_unsafe",
        "async",
        "async_unsafe",
        "split_async",
        "split_async_unsafe",
    ]

    names = String.(split(raw, ","))
    unknown = setdiff(names, collect(keys(methods)))
    isempty(unknown) || error("Unknown methods: $(join(unknown, ", "))")
    return names
end

function sanity_check!(halo, names, methods, comm)
    reset_halo!(halo)
    methods["waitall_unsafe"](halo)
    reference = copy(parent(halo))

    ok = true
    for name in names
        reset_halo!(halo)
        methods[name](halo)
        ok &= parent(halo) == reference
    end

    all_ok = MPI.Allreduce(ok, &, comm)
    all_ok || error("At least one exchange method produced a different halo result")
    return nothing
end

function benchmark_method!(halo, method, samples, warmups, comm)
    for _ in 1:warmups
        MPI.Barrier(comm)
        method(halo)
    end

    times = Vector{Float64}(undef, samples)
    for sample in 1:samples
        MPI.Barrier(comm)
        t0 = MPI.Wtime()
        method(halo)
        MPI.Barrier(comm)
        elapsed = MPI.Wtime() - t0
        times[sample] = MPI.Allreduce(elapsed, MPI.MAX, comm)
    end
    return times
end

function print_summary(name, times)
    println(rpad(name, 22),
        " min=", round(1e6 * minimum(times), digits=2), " us",
        " median=", round(1e6 * median_value(times), digits=2), " us",
        " p90=", round(1e6 * quantile_value(times, 0.90), digits=2), " us",
        " max=", round(1e6 * maximum(times), digits=2), " us")
end

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)

    options = parse_args(ARGS)
    ndims = option_int(options, "ndims", 2)
    halo_width = option_int(options, "halo", 1)
    samples = option_int(options, "samples", 30)
    warmups = option_int(options, "warmups", 5)
    local_size = option_tuple(options, "local-size", ndims, 64)

    topology = make_topology(comm, ndims)
    halo = make_halo(Val(ndims), local_size, halo_width, topology)
    methods = exchange_methods()
    names = selected_methods(options)

    sanity_check!(halo, names, methods, comm)

    if rank == 0
        println("Halo exchange benchmark")
        println("  ranks:       ", nproc)
        println("  topology:    ", topology.dims)
        println("  ndims:       ", ndims)
        println("  local size:  ", local_size)
        println("  halo width:  ", halo_width)
        println("  samples:     ", samples)
        println("  warmups:     ", warmups)
        println()
    end

    for name in names
        reset_halo!(halo)
        times = benchmark_method!(halo, methods[name], samples, warmups, comm)
        rank == 0 && print_summary(name, times)
    end

    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
