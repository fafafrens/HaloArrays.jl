# MPI halo-exchange benchmark: exchange cost vs message size, and how much of
# it the split start/finish API can hide behind computation.
#
# Runs under mpiexec — use the launcher:  julia --project=benchmark benchmark/run_mpi.jl 4
# HALO_BENCH_QUICK=1 shrinks the problem for a fast smoke run.
#
# For each cube size n³ it reports (max over ranks, median of samples):
#   exchange   blocking halo_exchange!
#   work       the standalone compute kernel (a saxpy sized ~ one stencil sweep)
#   seq        exchange followed by work (no overlap)
#   overlap    start_halo_exchange! ; work ; finish_halo_exchange!
#   hidden     fraction of the exchange hidden by overlap: (seq − overlap) / exchange

using MPI
using HaloArrays
using Printf
using Statistics

MPI.Init()
const COMM   = MPI.COMM_WORLD
const RANK   = MPI.Comm_rank(COMM)
const NRANKS = MPI.Comm_size(COMM)
const QUICK  = get(ENV, "HALO_BENCH_QUICK", "0") == "1"

# Median over samples on each rank, then max over ranks: the slowest rank paces
# the whole simulation, so that is the honest number.
function bench_max(f; warmup=5, samples=30)
    for _ in 1:warmup
        f()
    end
    ts = Vector{Float64}(undef, samples)
    for i in 1:samples
        MPI.Barrier(COMM)
        t0 = MPI.Wtime()
        f()
        ts[i] = MPI.Wtime() - t0
    end
    return MPI.Allreduce(median(ts), MPI.MAX, COMM)
end

function bench_size(n)
    u = HaloArray(Float64, (n, n, n), 1; boundary_condition=:periodic)
    fill_from_global_indices!(I -> Float64(sum(I)), u)

    # Ghost-independent compute stand-in, sized like one sweep over the interior.
    w = ones(n, n, n)
    work! = () -> (w .= 1.000001 .* w .+ 1e-9; nothing)

    t_ex  = bench_max(() -> halo_exchange!(u))
    t_wk  = bench_max(work!)
    t_seq = bench_max(() -> (halo_exchange!(u); work!()))
    t_ovl = bench_max(() -> (start_halo_exchange!(u); work!(); finish_halo_exchange!(u)))

    face_bytes = n^2 * sizeof(Float64)
    hidden = (t_seq - t_ovl) / t_ex
    if RANK == 0
        @printf("  %4d³  face %8.1f KiB   exchange %9.2f µs   work %9.2f µs   seq %9.2f µs   overlap %9.2f µs   hidden %5.1f %%\n",
            n, face_bytes / 1024, 1e6 * t_ex, 1e6 * t_wk, 1e6 * t_seq, 1e6 * t_ovl, 100 * hidden)
    end
end

function main()
    sizes = QUICK ? [16, 32] : [16, 32, 64, 128]
    if RANK == 0
        println("3-D halo exchange, halo width 1, Float64, periodic, $NRANKS ranks")
        println("(times are max over ranks of the per-rank median)")
    end
    foreach(bench_size, sizes)
end

main()
