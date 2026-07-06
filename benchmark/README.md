# HaloArrays benchmarks

One directory, two tiers:

- **Quick-start harnesses** — `stencil.jl` (single-process stencil throughput,
  Mcell/s, `LocalHaloArray` vs `ThreadedHaloArray`) and `exchange_mpi.jl`
  (MPI halo-exchange cost vs message size and how much the split
  `start_/finish_halo_exchange!` overlap hides; launched via `run_mpi.jl`).
  No options to think about — start here.
- **CLI micro-suite** — per-subsystem benchmarks (halo exchange variants,
  boundary conditions, reductions, gather/HDF5, heat solvers, threaded
  synchronization, thread backends, full hydro steps) with
  `--samples/--warmups/--csv` options for regression tracking.

## Setup (once)

From the repository root:

```bash
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

Unless a section says otherwise, run everything with `--project=benchmark`.

## Quick start

```bash
# stencil throughput, serial then 8 threads
julia --project=benchmark        benchmark/stencil.jl
julia --project=benchmark -t 8   benchmark/stencil.jl

# MPI exchange + overlap, 4 ranks
julia --project=benchmark benchmark/run_mpi.jl 4
```

Set `HALO_BENCH_QUICK=1` for a fast smoke run (small sizes, short loops).

- `stencil.jl` reports sustained **Mcell/s** for `synchronize_halo!` + one
  explicit Euler step. The threaded/local ratio at large sizes is the honest
  thread-scaling number; at small sizes tiling overhead dominates.
- `exchange_mpi.jl` reports, per cube size (max over ranks, median of samples):
  the blocking exchange, the standalone compute kernel, the two run
  sequentially, and the overlapped version — `hidden` is the fraction of the
  exchange cost that overlap recovered.

## Caveat: laptops lie about scaling

On a typical laptop (shared memory bandwidth, heterogeneous cores,
oversubscribed ranks) weak scaling will **not** look flat, the overlap gain
is erratic, and threaded timings swing ±2× with OS thread placement — that
reflects the machine, not the algorithm. Treat local runs as smoke tests and
regression checks; measure scaling on a cluster with one rank per core and a
real interconnect.

---

# CLI micro-suite

Common options: `--ndims=2`, `--halo=1`, `--samples=30`, `--warmups=5`,
`--csv=path/to/results.csv`. MPI benchmarks use `--owned-size` for the
per-rank local domain size and report the maximum elapsed time across ranks
per sample. `--timer=benchmarktools` (where supported) needs BenchmarkTools,
which the benchmark environment provides.

## Halo Exchange

```sh
mpiexec -n 4 julia --project=benchmark benchmark/halo_exchange.jl --owned-size=128,128
mpiexec -n 4 julia --project=benchmark benchmark/halo_exchange.jl --ndims=3 --owned-size=64,64,64 --methods=blocking,waitall_unsafe,waitall,async_unsafe
```

`blocking` is the public `halo_exchange!` path. The other method names benchmark
compatibility wrappers and implementation variants.

Reference (4 ranks, 2×2, 128², 8-core M-series; median): `blocking` ~49 µs,
`public_split` ~51 µs, `async` ~55 µs, `waitall` ~75 µs. All variants land in
the ~50–75 µs range; the async/split paths mainly help when overlapped with
compute, not in this back-to-back microbenchmark.

## Reductions

Benchmarks `mapreduce`, `all`, and `any` for `HaloArray`, `LocalHaloArray`,
`ThreadedHaloArray`, and compatible `MultiHaloArray` cases.

```sh
mpiexec -n 4 julia --project=benchmark benchmark/reductions.jl --owned-size=128,128 --tile-dims=2,2
```

Reference (4 ranks, 128²; median): `mpi_mapreduce` ~68 µs, `threaded` all/any
~52 µs; the 3-field `multi` variants are ~2× (≈125–166 µs).

## Ideal Hydro

Benchmarks the 2D ideal-hydrodynamics example with `LocalMultiHaloArray`,
`ThreadedMultiHaloArray`, and MPI `MultiHaloArray`. The output includes full-run
allocation bytes and diagnostics for the package-owned fill, RHS, and
wave-speed reduction kernels.

Run from the `examples` environment (the hydro solver pulls in DiffEqBase /
OrdinaryDiffEq):

```sh
JULIA_NUM_THREADS=8 julia --project=examples benchmark/ideal_hydro.jl --cases=local,threaded --nx=128 --ny=128 --tile-dims=2,2
mpiexec -n 4 julia --project=examples benchmark/ideal_hydro.jl --cases=mpi --nx=128 --ny=128
```

Reference (8 threads, 128²; median full run): local ~62 ms, threaded ~47 ms
(~1.3×). Unlike the tiny halo sync, a full hydro step has enough per-tile work
to amortize the task-spawn overhead, so threading pays here.

## Relativistic hydro steps

Full-step benchmarks of the relativistic examples (serial vs threaded vs MPI),
with per-kernel splits:

```sh
julia -t 8 --project=benchmark benchmark/bench_relhydro_mu0_2d.jl         # serial vs threaded
mpiexec -n 8 julia --project=benchmark benchmark/bench_relhydro_mu0_2d.jl # serial vs MPI
julia -t 8 --project=benchmark benchmark/bench_cylindrical_threaded.jl
```

Reference (M-series, 8 threads/ranks): 1024² 2-D μ=0 serial ~160 ms/step,
threaded 6.7×, MPI 7.25× — compute-heavy steps scale, unlike thin sweeps.

## Gather And HDF5

Benchmarks MPI gather and HDF5 write paths.

```sh
mpiexec -n 4 julia --project=benchmark benchmark/gather_hdf5.jl --owned-size=64,64 --output=/private/tmp/haloarrays_bench
```

Reference (4 ranks, 64²; median): `gather_haloarray` ~285 µs,
`gather_and_save` ~1.0 ms, `append_haloarray_to_file` ~4.8 ms — the HDF5 file
write dominates by an order of magnitude.

The `--output` option is a path prefix. The script writes files with suffixes
for gather/save and append cases.

For MPI+HDF5 runs, launch through the launcher bundled with MPI.jl so MPI.jl,
HDF5_jll, and `mpiexec` agree on the MPI library:

```sh
julia --project=benchmark -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 julia --project=benchmark benchmark/gather_hdf5.jl --owned-size=64,64 --output=/private/tmp/haloarrays_bench`)'
```

If you use a system MPI instead, every MPI-linked library in the process must
be the same implementation (a PMIx error means `mpiexec` and Julia's MPI
disagree; a crash involving both a system `libmpi` and an artifact `libmpi`
means system MPI was mixed with an MPI-enabled JLL such as HDF5_jll). For
Homebrew OpenMPI on macOS with `brew install hdf5-mpi`:

```sh
julia --project=benchmark -e 'using MPI; MPI.MPIPreferences.use_system_binary(; mpiexec="mpiexec", extra_paths=["/opt/homebrew/lib"])'
julia --project=benchmark -e 'using HDF5; HDF5.API.set_libraries!("/opt/homebrew/lib/libhdf5.dylib", "/opt/homebrew/lib/libhdf5_hl.dylib")'
```

## Threaded Operations

Benchmarks threaded halo synchronization, boundary conditions, reductions, and a
heat-step style stencil for `ThreadedHaloArray`.

```sh
julia --project=benchmark benchmark/threaded.jl --owned-size=128,128 --tile-dims=2,2
```

## Thread Backends

Compares the `ThreadBackend` implementations — `OhMyThreadsBackend` (default),
`SerialBackend`, and `PolyesterBackend` — on the operations that dispatch through
the trait (`tile_foreach` / `tile_mapreduce`): `synchronize_halo_threads!`,
`boundary_condition_threads!`, `fill!`, `mapreduce`, and broadcast. Each case
reports timing and per-call allocations. **Start Julia with `-t N`** or the
backends cannot be distinguished.

```sh
julia --project=benchmark -t 4 benchmark/thread_backends.jl --owned-size=256,256 --tile-dims=4,1
```

Useful options:

- `--backends=ohmythreads,serial,polyester` (subset/order to run)
- `--tile-dims=4,1` (set `prod(tile_dims)` to the thread count for best parallelism)
- `--owned-size=256,256`, `--samples=30`, `--warmups=5`
- `--csv=/tmp/thread_backends.csv`

Indicative results (Apple M-series, 4 threads, 256×256, `tile-dims=4,1`; median):
`PolyesterBackend` has the lowest overhead and near-zero allocation on the small
per-tile kernels (synchronize/boundary/fill/broadcast), while `OhMyThreadsBackend`
pays task-spawn overhead (a few KB/call); on `mapreduce` both parallel backends
beat `SerialBackend`, which cannot parallelize the reduction. Choose the backend
per workload via the `thread_backend=` keyword on `ThreadedHaloArray`.

## Metal Colored Cells

Benchmarks a 2D Metal red-black cell stencil using a naive full launch with a
parity branch versus `CellCheckerboard` compressed launches. The optional
manual kernels are hardcoded reference implementations used to check that the
generic helper path compiles to comparable code.

Requires `Metal.jl` and `KernelAbstractions.jl`; the repository's `examples`
environment provides both:

```sh
julia --project=examples benchmark/metal_colored_cells.jl --sizes=128,256,512,1024
```

Useful options: `--sizes=…`, `--steps=50`, `--samples=10`, `--warmups=3`,
`--include-manual=true`, `--csv=/tmp/metal_colored_cells.csv`.

Reference (Apple M-series GPU; median): the `CellCheckerboard`
compressed launch is ~1.24× faster than the naive full launch with a parity
branch. The hardcoded manual kernels are ~0.9× (slightly slower), confirming
the generic helper path is competitive.

## Threaded Synchronization Variants

Compares benchmark-only implementations of threaded halo synchronization:
the production serial tile loop, an `OhMyThreads.@tasks` loop, and a
`Base.Threads.@threads :static` loop. This is useful for checking whether
parallelizing halo copies helps for a given tile size and halo width.

```sh
JULIA_NUM_THREADS=4 julia --project=benchmark benchmark/threaded_sync_variants.jl --owned-size=2048,2048 --tile-dims=4,1 --halo=8 --boundary=repeating --timer=benchmarktools
```

### Reference results (8 threads, Apple M-series; median, indicative only)

`serial` is the production `synchronize_halo!`; `threads` is
`synchronize_halo_threads!` (an `OhMyThreads` `tforeach`).

| Config | tiles | serial | threads | winner |
| --- | ---: | ---: | ---: | --- |
| 2D 64², halo 1 | 4 | 0.9 µs | 24 µs | serial 25× |
| 2D 256², halo 1 | 16 | 6.6 µs | 140 µs | serial 21× |
| 2D 1024², halo 1 | 16 | 21 µs | 142 µs | serial 7× |
| 2D 1024², halo 5 | 16 | 62 µs | 145 µs | serial 2.3× |
| 2D 2048², halo 3 | 64 | 153 µs | 181 µs | serial 1.2× |
| 3D 128³, halo 2 | 8 | 318 µs | 274 µs | threads 1.16× |

The task-based `threads`/`tasks` variants carry a large fixed spawn+join
cost (~130–180 µs here) and allocate ~4 KB per call, so the serial loop
wins until the per-exchange work exceeds ~300 µs (3-D, wide halos, many
tiles). The same overhead shows up in the threaded *stencil* step
(`heat_solver_local_threaded.jl`): a 64² threaded heat step is ~120 µs
vs ~4 µs for the local serial step. This is why `synchronize_halo!`
(and the tutorials) default to serial.

**Allocation-free threading:** [Polyester.jl](https://github.com/JuliaSIMD/Polyester.jl)
`@batch` uses a persistent preallocated worker pool instead of spawning
tasks. On the same 16-tile micro-workload it measured ~0.4 µs and ~80 B
(vs ~135 µs / ~3.6 KB for `tforeach` and `@threads`), i.e. it can beat
even the serial loop. A `@batch`-based halo sync would push the crossover
down to much smaller tiles — at the cost of Polyester's no-nesting
constraint (do not nest `@batch` inside another threaded region).

## Boundary Conditions

Benchmarks `boundary_condition!` and `synchronize_halo!` for `LocalHaloArray`,
`ThreadedHaloArray`, and MPI `HaloArray`. The output includes allocation bytes,
reported as the maximum across MPI ranks for MPI cases.

```sh
julia --project=benchmark benchmark/boundary_conditions.jl --owned-size=128,128 --tile-dims=2,2 --modes=repeating,reflecting,antireflecting,periodic
mpiexec -n 4 julia --project=benchmark benchmark/boundary_conditions.jl --owned-size=128,128 --tile-dims=2,2 --modes=repeating,periodic
```

Use `--timer=benchmarktools` for the rank-local `LocalHaloArray` and
`ThreadedHaloArray` cases. MPI cases always use the manual barrier/max-time
timer.

Reference (128², 8 threads; median, 0 B alloc for every case): local
`synchronize_halo!` ~0.67 µs, threaded ~1.5 µs; `antireflecting` is the
cheapest mode (~0.25 µs local). The threaded sync is a small constant factor
slower here because it walks the tile collection serially.

## MPI Diagnostics

Benchmarks MPI exchange and synchronization paths with allocation reporting.
This is meant to catch hidden allocations or type-instability symptoms in the
low-level communication paths.

```sh
mpiexec -n 4 julia --project=benchmark benchmark/mpi_diagnostics.jl --owned-size=128,128 --boundary=repeating
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

```sh
JULIA_NUM_THREADS=2 mpiexec -n 2 julia --project=benchmark benchmark/heat_solver_mpi.jl --owned-size=256,512 --tile-dims=2,1 --steps=10
```

## Local And Threaded Heat Solver

Compares `LocalHaloArray` and `ThreadedHaloArray` without MPI. Here `--size`
is the full local problem size, so there is no owned/global distinction.
(`stencil.jl` above answers the same question with fewer knobs; this variant
adds split timings and the CLI/CSV machinery.)

```sh
JULIA_NUM_THREADS=2 julia --project=benchmark benchmark/heat_solver_local_threaded.jl --size=512,512 --tile-dims=2,1 --timer=benchmarktools
```
