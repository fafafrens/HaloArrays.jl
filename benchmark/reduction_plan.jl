include("common.jl")

# Dimensional reduction, one-shot vs planned:
#   oneshot — mapreduce_haloarray_dims (== sum(u; dims=…)): builds a transient
#             DimReductionPlan per call (2× MPI.Cart_sub) and hands the result
#             its sub-communicator (freed here each sample via free!)
#   plan    — DimReductionPlan built once, then one MPI.Reduce per reduce!
#             into a preallocated output
#
#   mpiexec -n 4 julia --project=benchmark benchmark/reduction_plan.jl \
#       [--ndims=2] [--owned-size=256,256] [--halo=1] [--samples=30] [--warmups=5]

function benchmark_case!(rows, name, f, samples, warmups, metadata; comm, rank)
    sink = Ref{Any}()
    times = benchmark_times!(samples, warmups; comm=comm) do
        sink[] = f()
    end
    if rank == 0
        print_summary(name, times)
        push!(rows, benchmark_record("reduction_plan", name, times; metadata=copy(metadata)))
    end
    return times
end

function main()
    comm = ensure_mpi()
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)

    options = parse_args()
    ndims_ = option_int(options, "ndims", 2)
    halo_width = option_int(options, "halo", 1)
    samples = option_int(options, "samples", 30)
    warmups = option_int(options, "warmups", 5)
    interior_size = option_owned_size(options, ndims_, 256)

    topology = make_periodic_topology(comm, ndims_)
    u = HaloArray(Float64, interior_size, halo_width, topology; boundary_condition=:periodic)
    fill_benchmark_data!(u)

    if rank == 0
        println("Dimensional-reduction benchmark: mapreduce_haloarray_dims vs DimReductionPlan")
        println("  ranks:       ", nproc)
        println("  topology:    ", topology.dims)
        println("  ndims:       ", ndims_)
        println("  owned size:  ", interior_size)
        println("  halo width:  ", halo_width)
        println("  samples:     ", samples)
        println("  warmups:     ", warmups)
        println()
    end

    metadata = Dict{String,Any}(
        "ranks" => nproc,
        "topology" => joined_tuple(topology.dims),
        "ndims" => ndims_,
        "interior_size" => joined_tuple(interior_size),
        "halo_width" => halo_width,
    )
    rows = Dict{String,Any}[]

    for dim in 1:ndims_
        # One-time plan construction cost (itself collective), reported separately.
        MPI.Barrier(comm)
        t0 = MPI.Wtime()
        plan = DimReductionPlan(u, dim)
        MPI.Barrier(comm)
        build_time = MPI.Allreduce(MPI.Wtime() - t0, MPI.MAX, comm)
        rank == 0 && println("plan_build dims=", dim, "            once=",
            round(1e6 * build_time, digits=2), " us")

        t_old = benchmark_case!(rows, "oneshot_per_call dims=$dim",
            () -> free!(mapreduce_haloarray_dims(identity, +, u, dim)),
            samples, warmups, metadata; comm=comm, rank=rank)
        t_plan = benchmark_case!(rows, "plan_reduce! dims=$dim",
            () -> reduce!(plan, identity, +, u),
            samples, warmups, metadata; comm=comm, rank=rank)

        if rank == 0
            speedup = median_value(t_old) / median_value(t_plan)
            println("  → plan speedup (median): ", round(speedup, digits=1), "x\n")
        end
        free!(plan)
    end

    if rank == 0
        maybe_write_csv(options, rows)
    end

    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
