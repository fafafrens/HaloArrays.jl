# HaloArrays benchmarks

Small, self-contained performance harness. Two axes:

- **`stencil.jl`** — single-process stencil throughput (Mcell/s) of the 2-D heat
  kernel from `examples/heat/common.jl`, on `LocalHaloArray` vs
  `ThreadedHaloArray`. Run with different `-t` to see thread scaling.
- **`exchange_mpi.jl`** — MPI halo-exchange cost vs message size, and how much
  of it the split `start_halo_exchange!` / `finish_halo_exchange!` pair hides
  behind computation. Launched via `run_mpi.jl`.

## Setup (once)

From the repository root:

```bash
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

## Run

```bash
# stencil throughput, serial then 8 threads
julia --project=benchmark        benchmark/stencil.jl
julia --project=benchmark -t 8   benchmark/stencil.jl

# MPI exchange + overlap, 4 ranks
julia --project=benchmark benchmark/run_mpi.jl 4
```

Set `HALO_BENCH_QUICK=1` for a fast smoke run (small sizes, short loops).

## Reading the numbers

- `stencil.jl` reports sustained **Mcell/s** for `synchronize_halo!` + one
  explicit Euler step. The threaded/local ratio at large sizes is the honest
  thread-scaling number; at small sizes tiling overhead dominates.
- `exchange_mpi.jl` reports, per cube size (max over ranks, median of samples):
  the blocking exchange, the standalone compute kernel, the two run
  sequentially, and the overlapped version — `hidden` is the fraction of the
  exchange cost that overlap recovered.

## Caveat: laptops lie about scaling

On a typical laptop (shared memory bandwidth, heterogeneous cores,
oversubscribed ranks) weak scaling will **not** look flat and the overlap gain
is erratic — that reflects the machine, not the algorithm. Treat local runs as
smoke tests and regression checks; measure scaling on a cluster with one rank
per core and a real interconnect.
