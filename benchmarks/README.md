# Benchmarks

Run benchmarks from the repository root. MPI benchmarks report the maximum
elapsed time across ranks for each sample.

Common options:

- `--ndims=2`
- `--halo=1`
- `--samples=30`
- `--warmups=5`
- `--csv=path/to/results.csv`

MPI benchmarks use `--owned-size` for the per-rank local domain size.
The local/threaded heat solver benchmark uses `--size` for the full local
problem size.
`--local-size` is still accepted by `halo_exchange.jl` as a deprecated alias for
`--owned-size`, but new commands should use `--owned-size`.

## Halo Exchange

```sh
mpiexec -n 4 julia --project=. benchmarks/halo_exchange.jl --owned-size=128,128
mpiexec -n 4 julia --project=. benchmarks/halo_exchange.jl --ndims=3 --owned-size=64,64,64 --methods=blocking,waitall_unsafe,waitall,async_unsafe
```

`blocking` is the public `halo_exchange!` path. The other method names benchmark
compatibility wrappers and implementation variants.

## Reductions

Benchmarks `mapreduce`, `all`, and `any` for `HaloArray`, `LocalHaloArray`,
`ThreadedHaloArray`, and compatible `MultiHaloArray` cases.

```sh
mpiexec -n 4 julia --project=. benchmarks/reductions.jl --owned-size=128,128 --tile-dims=2,2
```

## Gather And HDF5

Benchmarks MPI gather and HDF5 write paths.

```sh
mpiexec -n 4 julia --project=. benchmarks/gather_hdf5.jl --owned-size=64,64 --output=/private/tmp/haloarrays_bench
```

The `--output` option is a path prefix. The script writes files with suffixes
for gather/save and append cases.

## Threaded Operations

Benchmarks threaded halo synchronization, boundary conditions, reductions, and a
heat-step style stencil for `ThreadedHaloArray`.

```sh
julia --project=. benchmarks/threaded.jl --owned-size=128,128 --tile-dims=2,2
```

## Threaded Synchronization Variants

Compares benchmark-only implementations of threaded halo synchronization:
the production serial tile loop, an `OhMyThreads.@tasks` loop, and a
`Base.Threads.@threads :static` loop. This is useful for checking whether
parallelizing halo copies helps for a given tile size and halo width.

```sh
JULIA_NUM_THREADS=4 julia --project=. benchmarks/threaded_sync_variants.jl --owned-size=2048,2048 --tile-dims=4,1 --halo=8 --boundary=repeating --timer=benchmarktools
```

## Boundary Conditions

Benchmarks `boundary_condition!` and `synchronize_halo!` for `LocalHaloArray`,
`ThreadedHaloArray`, and MPI `HaloArray`. The output includes allocation bytes,
reported as the maximum across MPI ranks for MPI cases.

```sh
julia --project=. benchmarks/boundary_conditions.jl --owned-size=128,128 --tile-dims=2,2 --modes=repeating,reflecting,antireflecting,periodic
mpiexec -n 4 julia --project=. benchmarks/boundary_conditions.jl --owned-size=128,128 --tile-dims=2,2 --modes=repeating,periodic
```

Use `--timer=benchmarktools` for the rank-local `LocalHaloArray` and
`ThreadedHaloArray` cases. MPI cases always use the manual barrier/max-time
timer.

## MPI Diagnostics

Benchmarks MPI exchange and synchronization paths with allocation reporting.
This is meant to catch hidden allocations or type-instability symptoms in the
low-level communication paths.

```sh
mpiexec -n 4 julia --project=. benchmarks/mpi_diagnostics.jl --owned-size=128,128 --boundary=repeating
```

The periodic cases benchmark pure neighbor exchange. The physical-boundary cases
benchmark exchange, boundary fill, and the combined `synchronize_halo!` path.

## MPI Heat Solver

Solves the same periodic heat equation workload with MPI `HaloArray`,
`LocalHaloArray`, and `ThreadedHaloArray`. `--owned-size` is the MPI owned size
per rank; the local and threaded cases use the equivalent global problem size.
For example, with 2 MPI ranks and topology `(2, 1)`, `--owned-size=256,512`
gives global problem size `(512, 512)`.
The output includes both the full `--steps` solve and single-step split timings
for halo synchronization, stencil work, and synchronization plus stencil.
`--timer=benchmarktools` uses BenchmarkTools for the rank-local
`LocalHaloArray` and `ThreadedHaloArray` cases; MPI cases always use the manual
barrier/max-time timer.
Use `--timer=manual` or `--timer=benchmarktools` for the rank-local cases.
Set `JULIA_NUM_THREADS` when comparing the rank-local `ThreadedHaloArray`
numbers printed by this benchmark.

```sh
JULIA_NUM_THREADS=2 mpiexec -n 2 julia --project=. benchmarks/heat_solver_mpi.jl --owned-size=256,512 --tile-dims=2,1 --steps=10
```

## Local And Threaded Heat Solver

Compares `LocalHaloArray` and `ThreadedHaloArray` without MPI. Here `--size`
is the full local problem size, so there is no owned/global distinction.
Use `--timer=manual` or `--timer=benchmarktools`.

```sh
JULIA_NUM_THREADS=2 julia --project=. benchmarks/heat_solver_local_threaded.jl --size=512,512 --tile-dims=2,1 --timer=benchmarktools
```
