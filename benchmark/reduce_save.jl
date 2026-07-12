include("common.jl")
using HDF5   # activates the HDF5 I/O extension (collective write)

# Saving a reduced quantity (e.g. a profile = sum over one axis) of a
# distributed HaloArray, three ways — end to end (reduction + HDF5 write):
#
#   gather_reduce_save : gather the WHOLE global array to root, reduce there,
#                        root writes the profile. Comm ∝ prod(global_size),
#                        serial write on root.
#   reduce_gather_save : mapreduce_haloarray_dims reduces in place across MPI
#                        sub-communicators (Cart_sub + one MPI.Reduce), then the
#                        (small) reduced array is gathered to root and written.
#   reduce_save        : reduce in place, then write the distributed reduced
#                        result COLLECTIVELY — every rank writes its own block,
#                        NO gather. Comm ∝ prod(reduced) and the I/O is parallel.
#
# The reduced result is identical; the new paths move `global_size[dim]` times
# less data, and `reduce_save` skips the gather bottleneck entirely.
#
# Both new paths are far faster than gather_reduce_save (which moves the whole
# array to one rank): ~5-8x here. Between them, which wins is scale-dependent —
# gathering the small reduced profile to root and writing it serially beats the
# collective write when the reduced result and rank count are modest (collective
# HDF5 has a fixed per-call overhead). `reduce_save` (no gather, parallel I/O)
# pulls ahead only at large scale, where the gather-to-root serialization and the
# single-rank write become the bottleneck — on a few laptop ranks it is
# overhead-bound (cf. the weak-scaling note in the README).
#
#   mpiexec -n 4 julia --project=benchmark benchmark/reduce_save.jl \
#       [--ndims=2] [--owned-size=256,256] [--dim=2] [--samples=20] [--warmups=3]

function main()
    comm  = ensure_mpi()
    rank  = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)

    options = parse_args()
    ndims_  = option_int(options, "ndims", 2)
    halo    = option_int(options, "halo", 1)
    samples = option_int(options, "samples", 20)
    warmups = option_int(options, "warmups", 3)
    owned   = option_owned_size(options, ndims_, 256)
    rdim    = option_int(options, "dim", ndims_)     # reduce this spatial axis

    topology = make_periodic_topology(comm, ndims_)
    u = HaloArray(Float64, owned, halo, topology; boundary_condition=:periodic)
    fill_benchmark_data!(u)
    gsize = global_size(u)

    dir = tempdir()
    fA  = joinpath(dir, "rs_gather_reduce_$(nproc).h5")
    fB  = joinpath(dir, "rs_reduce_gather_$(nproc).h5")
    fC  = joinpath(dir, "rs_reduce_collective_$(nproc)")   # append_ adds ".h5"

    function root_write(file, data)
        if rank == 0 && data !== nothing
            h5open(file, "w") do fid
                write(fid, "profile", data)
            end
        end
        return nothing
    end

    # old: gather the full array to root, reduce on root, write on root.
    gather_reduce_save = function ()
        g = gather_haloarray(u; root=0)
        r = rank == 0 ? dropdims(sum(g; dims=rdim); dims=rdim) : nothing
        root_write(fA, r)
    end
    # reduce in place, gather the small result to root, write on root.
    reduce_gather_save = function ()
        mr = mapreduce_haloarray_dims(identity, +, u, rdim)
        r  = is_active(mr) ? gather_haloarray(parent(mr); root=0) : nothing
        root_write(fB, r)
        free!(mr)
    end
    # reduce in place, write the distributed result collectively (NO gather).
    reduce_save = function ()
        mr = mapreduce_haloarray_dims(identity, +, u, rdim)
        append_haloarray_to_file!(fC, "profile", mr)     # per-rank block, collective
        free!(mr)
    end

    # correctness: the full-gather reduction and the in-place reduction agree on root.
    g = gather_haloarray(u; root=0)
    a = rank == 0 ? dropdims(sum(g; dims=rdim); dims=rdim) : nothing
    mr = mapreduce_haloarray_dims(identity, +, u, rdim)
    b  = is_active(mr) ? gather_haloarray(parent(mr); root=0) : nothing
    free!(mr)
    if rank == 0 && a !== nothing && b !== nothing
        (size(a) == size(b) && a ≈ b) || error("reduce_save: reduction paths disagree")
    end

    reduced_cells = prod(gsize) ÷ gsize[rdim]
    if rank == 0
        println("Reduce-and-save benchmark (reduction + HDF5 write)")
        println("  ranks:        ", nproc, "   topology ", topology.dims)
        println("  global size:  ", gsize, "  (", prod(gsize), " cells)")
        println("  reduce dim:   ", rdim, "  (extent ", gsize[rdim], ")")
        println("  data moved:   new paths move ~", gsize[rdim],
                "× less (full ", prod(gsize), " vs reduced ", reduced_cells, " cells)")
        println("  samples/warmups: ", samples, "/", warmups, "\n")
    end

    t_grs = benchmark_times!(gather_reduce_save, samples, warmups; comm=comm)
    t_rgs = benchmark_times!(reduce_gather_save, samples, warmups; comm=comm)
    t_rs  = benchmark_times!(reduce_save,        samples, warmups; comm=comm)

    if rank == 0
        print_summary("gather_reduce_save", t_grs)
        print_summary("reduce_gather_save", t_rgs)
        print_summary("reduce_save (no gather)", t_rs)
        base = median_value(t_grs)
        println("  → speedup vs gather_reduce_save (median): ",
                "reduce_gather_save ", round(base / median_value(t_rgs), digits=1), "×,  ",
                "reduce_save ", round(base / median_value(t_rs), digits=1), "×")

        meta = Dict{String,Any}("ranks" => nproc, "global" => joined_tuple(gsize), "dim" => rdim)
        rows = [benchmark_record("reduce_save", "gather_reduce_save", t_grs; metadata=copy(meta)),
                benchmark_record("reduce_save", "reduce_gather_save", t_rgs; metadata=copy(meta)),
                benchmark_record("reduce_save", "reduce_save",        t_rs;  metadata=copy(meta))]
        maybe_write_csv(options, rows)
        for f in (fA, fB, fC * ".h5"); rm(f; force=true); end
    end

    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
