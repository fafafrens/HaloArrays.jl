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

For MPI benchmarks that use HDF5, prefer the launcher bundled through MPI.jl so
MPI.jl, HDF5_jll, and `mpiexec` all use the same MPI library:

```sh
julia --project=. -e 'using MPI; MPI.MPIPreferences.use_jll_binary("OpenMPI_jll")'
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --project=. -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 julia --project=. benchmarks/gather_hdf5.jl --owned-size=64,64 --output=/private/tmp/haloarrays_bench`)'
```

The first command writes a local `LocalPreferences.toml`; restart Julia after
running it. The second command refreshes artifacts such as the MPI-enabled HDF5
build for the selected MPI ABI. This repository lists `MPIPreferences` in
`[extras]` so local preferences are visible to MPI.jl.

If you install a system MPI, make sure every MPI-linked library in the process
uses that same implementation. For Homebrew OpenMPI on macOS with
`brew install hdf5-mpi`:

```sh
julia --project=. -e 'using MPI; MPI.MPIPreferences.use_system_binary(; mpiexec="mpiexec", extra_paths=["/opt/homebrew/lib"])'
julia --project=. -e 'using HDF5; HDF5.API.set_libraries!("/opt/homebrew/lib/libhdf5.dylib", "/opt/homebrew/lib/libhdf5_hl.dylib")'
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

A PMIx error usually means `mpiexec` and the MPI library loaded by Julia are from
different MPI implementations. A crash involving both `/opt/homebrew/.../libmpi`
and `~/.julia/artifacts/.../libmpi` usually means system MPI was mixed with an
MPI-enabled JLL such as HDF5_jll.

When using system HDF5, build it with parallel HDF5 enabled and link it against
the same MPI installation used by MPI.jl. Check with `otool -L
/path/to/system/libhdf5.dylib` on macOS, or `ldd /path/to/system/libhdf5.so` on
Linux, before running the benchmark.

## Halo Exchange

```sh
mpiexec -n 4 julia --project=. benchmarks/halo_exchange.jl --owned-size=128,128
mpiexec -n 4 julia --project=. benchmarks/halo_exchange.jl --ndims=3 --owned-size=64,64,64 --methods=blocking,waitall_unsafe,waitall,async_unsafe
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
mpiexec -n 4 julia --project=. benchmarks/reductions.jl --owned-size=128,128 --tile-dims=2,2
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
JULIA_NUM_THREADS=8 julia --project=examples benchmarks/ideal_hydro.jl --cases=local,threaded --nx=128 --ny=128 --tile-dims=2,2
mpiexec -n 4 julia --project=examples benchmarks/ideal_hydro.jl --cases=mpi --nx=128 --ny=128
```

Reference (8 threads, 128²; median full run): local ~62 ms, threaded ~47 ms
(~1.3×). Unlike the tiny halo sync, a full hydro step has enough per-tile work
to amortize the task-spawn overhead, so threading pays here.

## Gather And HDF5

Benchmarks MPI gather and HDF5 write paths.

```sh
mpiexec -n 4 julia --project=. benchmarks/gather_hdf5.jl --owned-size=64,64 --output=/private/tmp/haloarrays_bench
```

Reference (4 ranks, 64²; median): `gather_haloarray` ~285 µs,
`gather_and_save` ~1.0 ms, `append_haloarray_to_file` ~4.8 ms — the HDF5 file
write dominates by an order of magnitude.

The `--output` option is a path prefix. The script writes files with suffixes
for gather/save and append cases.

## Threaded Operations

Benchmarks threaded halo synchronization, boundary conditions, reductions, and a
heat-step style stencil for `ThreadedHaloArray`.

```sh
julia --project=. benchmarks/threaded.jl --owned-size=128,128 --tile-dims=2,2
```

## Metal Colored Cells

Benchmarks a 2D Metal red-black cell stencil using a naive full launch with a
parity branch versus `ColoredCellKernelRegion` compressed launches. The optional
manual kernels are hardcoded reference implementations used to check that the
generic helper path compiles to comparable code.

This benchmark requires `Metal.jl` and `KernelAbstractions.jl` in the active
Julia environment. One way to create a local benchmark environment is:

```sh
julia --project=/private/tmp/halo-metal-probe -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.add(["Metal", "KernelAbstractions"])'
julia --project=/private/tmp/halo-metal-probe benchmarks/metal_colored_cells.jl --sizes=128,256,512,1024
```

Useful options:

- `--sizes=128,256,512,1024`
- `--steps=50`
- `--samples=10`
- `--warmups=3`
- `--include-manual=true`
- `--csv=/tmp/metal_colored_cells.csv`

The repository's `examples` environment already provides `Metal` and
`KernelAbstractions`, so `julia --project=examples benchmarks/metal_colored_cells.jl`
works without a separate probe environment.

Reference (Apple M-series GPU; median): the `ColoredCellKernelRegion`
compressed launch is ~1.24× faster than the naive full launch with a parity
branch. The hardcoded manual kernels are ~0.9× (slightly slower), confirming
the generic helper path is competitive.

## Threaded Synchronization Variants

Compares benchmark-only implementations of threaded halo synchronization:
the production serial tile loop, an `OhMyThreads.@tasks` loop, and a
`Base.Threads.@threads :static` loop. This is useful for checking whether
parallelizing halo copies helps for a given tile size and halo width.

```sh
JULIA_NUM_THREADS=4 julia --project=. benchmarks/threaded_sync_variants.jl --owned-size=2048,2048 --tile-dims=4,1 --halo=8 --boundary=repeating --timer=benchmarktools
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
julia --project=. benchmarks/boundary_conditions.jl --owned-size=128,128 --tile-dims=2,2 --modes=repeating,reflecting,antireflecting,periodic
mpiexec -n 4 julia --project=. benchmarks/boundary_conditions.jl --owned-size=128,128 --tile-dims=2,2 --modes=repeating,periodic
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
