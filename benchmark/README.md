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

### Reference results (Apple M4 Max, 12 P + 4 E cores; indicative only)

HaloArrays 0.16.1, Julia 1.13.3, macOS 26.7, 2026-11-24. One run each, on a
machine that was not idle (load average about 3), so read them as orders of
magnitude.

`stencil.jl`, 2-D heat stencil, halo 1, Float64, periodic, in **Mcell/s**:

| size | Local | Threaded 1 | 2 threads | 4 threads | 8 threads | 16 threads |
|---|---|---|---|---|---|---|
| 256² | 1660 | 1470 | 1577 | 2980 | 2965 | 502 |
| 1024² | 1678 | 1472 | 2710 | 4749 | 8155 | 5473 |
| 2048² | 1541 | 1379 | 2483 | 4861 | 9286 | 9149 |

At 2048² eight tiles run 6.0× the local array. One tile costs about 11 % against
`LocalHaloArray`. Sixteen threads spill onto the efficiency cores and gain
nothing; at 256² the 16-row tiles make it three times slower than serial. The
tile split is `(1, nthreads)`, so a thread count that does not divide the size
(12 here) skips the threaded case.

`run_mpi.jl` (`exchange_mpi.jl`), 3-D exchange, halo 1, Float64, periodic,
MPICH 5.0.2 from `MPICH_jll` (the benchmark environment's default, not the
Homebrew OpenMPI), blocking `exchange` time in µs and `hidden` fraction:

| per-rank size | face | 2 ranks | 4 ranks | 8 ranks |
|---|---|---|---|---|
| 16³ | 2 KiB | 17 (76 %) | 20 (6 %) | 31 (4 %) |
| 32³ | 8 KiB | 14 (13 %) | 44 (48 %) | 66 (12 %) |
| 64³ | 32 KiB | 58 (5 %) | 74 (−11 %) | 135 (1 %) |
| 128³ | 128 KiB | 185 (−11 %) | 288 (5 %) | 529 (10 %) |

The overlap gain is noise on this machine (see the caveat below): the `hidden`
column changes sign from one size to the next.

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
mpiexec -n 4 julia --project=benchmark benchmark/halo_exchange.jl --ndims=3 --owned-size=64,64,64 --methods=blocking,split
```

`blocking` is `halo_exchange!` (one `Waitall`); `split` is the
`start_halo_exchange!` / `finish_halo_exchange!` pair (one `Wait` per face).

Reference (4 ranks, M4 Max, v0.16.1; median): 2×2, 128²: `blocking` ~34 µs,
`split` ~38 µs; 2×2×1, 64³: `blocking` ~95 µs, `split` ~99 µs. The split path
mainly helps when overlapped with compute, not in this back-to-back
microbenchmark.

## Reductions

Benchmarks `mapreduce`, `all`, and `any` for `HaloArray`, `LocalHaloArray`,
`ThreadedHaloArray`, and compatible `MultiHaloArray` cases.

```sh
mpiexec -n 4 julia --project=benchmark benchmark/reductions.jl --owned-size=128,128 --tile-dims=2,2
```

Reference (4 ranks, 128², M4 Max, v0.16.1; median): `mpi_mapreduce` ~51 µs,
`mpi_all`/`mpi_any` ~48–51 µs; `local` and `threaded` `mapreduce` ~34–35 µs,
all/any ~30–35 µs. The 3-field `multi` variants are ~1.3× on the serial
backends (≈39–48 µs) and ~1.6× on MPI (≈75–84 µs).

## Ideal Hydro

Benchmarks the 2D ideal-hydrodynamics example with a `MultiHaloArray` of
local, threaded, and MPI fields. The output includes full-run
allocation bytes and diagnostics for the package-owned fill, RHS, and
wave-speed reduction kernels.

Run from the `examples` environment (the hydro solver pulls in DiffEqBase /
OrdinaryDiffEq):

```sh
JULIA_NUM_THREADS=8 julia --project=examples benchmark/ideal_hydro.jl --cases=local,threaded --nx=128 --ny=128 --tile-dims=2,2
mpiexec -n 4 julia --project=examples benchmark/ideal_hydro.jl --cases=mpi --nx=128 --ny=128
```

Reference (M4 Max, v0.16.1, 128²; median full run of 6 steps): local ~23 ms,
threaded (8 threads, 2×2 tiles) ~11 ms (~2.1×), MPI on 4 ranks ~10 ms (~2.2×).
Before collection `fill!` and broadcast became one tile sweep, and the example
moved to `tile_foreach`, the threaded run was ~15–16 ms (~1.5×).
The fill, RHS and wave-speed kernels are inferred and allocation-free on the
local array; the threaded ones allocate 2–5 KB per call with the default
`ThreadsBackend` (task spawns; RHS 5.1 KB, down from 11.5 KB), nothing with
`SerialBackend` and under 1 KB with `PolyesterBackend`. Unlike
the tiny halo sync, a full hydro step has enough per-tile work to amortize the
task-spawn overhead, so threading pays here.

## Relativistic hydro steps

Full-step benchmarks of the relativistic examples (serial vs threaded vs MPI),
with per-kernel splits:

```sh
julia -t 8 --project=benchmark benchmark/bench_relhydro_mu0_2d.jl         # serial vs threaded
mpiexec -n 8 julia --project=benchmark benchmark/bench_relhydro_mu0_2d.jl # serial vs MPI
julia -t 8 --project=benchmark benchmark/bench_cylindrical_threaded.jl
```

Reference (M4 Max, v0.16.1, 8 threads/ranks): 1024² 2-D μ=0 serial
~160 ms/step, threaded 6.8× (23.6 ms), MPI 6.0× (26.7 ms, 4×2 ranks);
cylindrical, `nx=12000`: serial 8.1 ms/step, threaded 5.8× (1.4 ms) —
compute-heavy steps scale, unlike thin sweeps.

## Gather And HDF5

Benchmarks MPI gather and HDF5 write paths.

```sh
mpiexec -n 4 julia --project=benchmark benchmark/gather_hdf5.jl --owned-size=64,64 --output=/private/tmp/haloarrays_bench
```

Reference (4 ranks, 64², M4 Max, v0.16.1; median): `gather_haloarray` ~183 µs,
`gather_haloarray+h5write` ~0.38 ms, `append_haloarray!` ~2.6 ms — the
collective HDF5 append dominates by an order of magnitude.

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

Reference (M4 Max, v0.16.1, 8 threads, 128², 2×2 tiles; median):
`synchronize_halo` ~0.46 µs, `boundary_condition` ~0.17 µs, `mapreduce` ~21 µs
(7–140 µs spread), `stencil_step` ~24 µs.

## Thread Backends

Compares the `ThreadBackend` implementations — `ThreadsBackend` (default),
`OhMyThreadsBackend`, `SerialBackend`, and `PolyesterBackend` — on the operations that dispatch through
the trait (`tile_foreach` / `tile_mapreduce`): `synchronize_halo!(u; threads=true)`,
`boundary_condition!(u; threads=true)`, `fill!`, `mapreduce`, and broadcast. Each case
reports timing and per-call allocations. **Start Julia with `-t N`** or the
backends cannot be distinguished.

```sh
julia --project=benchmark -t 4 benchmark/thread_backends.jl --owned-size=256,256 --tile-dims=4,1
```

Useful options:

- `--backends=threads,ohmythreads,serial,polyester` (subset/order to run)
- `--tile-dims=4,1` (set `prod(tile_dims)` to the thread count for best parallelism)
- `--owned-size=256,256`, `--samples=30`, `--warmups=5`
- `--csv=/tmp/thread_backends.csv`

Reference (M4 Max, v0.16.1, 4 threads, 256×256, `tile-dims=4,1`; median µs,
allocation per call):

| operation | `threads` | `ohmythreads` | `serial` | `polyester` |
|---|---|---|---|---|
| synchronize | 13.4 (1.3 KB) | 14.5 (2.4 KB) | 1.6 (0 B) | 1.8 (80 B) |
| boundary | 13.3 (1.3 KB) | 14.7 (2.4 KB) | 0.29 (0 B) | 0.25 (80 B) |
| fill | 15.4 (1.3 KB) | 14.0 (2.5 KB) | 5.9 (0 B) | 2.4 (96 B) |
| mapreduce | 24.0 (1.4 KB) | 14.6 (2.4 KB) | 39.1 (0 B) | 11.4 (80 B) |
| broadcast | 15.5 (1.5 KB) | 14.6 (2.8 KB) | 8.7 (0 B) | 2.6 (160 B) |

`PolyesterBackend` has the lowest overhead and near-zero allocation on the small
per-tile kernels, while the task-based `ThreadsBackend` and `OhMyThreadsBackend`
pay a fixed ~13–15 µs spawn and 1–3 KB per call, more than the serial loop
costs on everything here except `mapreduce`. On `mapreduce` every parallel
backend beats `SerialBackend`, which cannot parallelize the reduction. Choose
the backend per workload via the `thread_backend=` keyword on
`ThreadedHaloArray`.

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

Reference (M4 Max GPU, v0.16.1; median µs per step, speedup vs the naive full
launch with a parity branch):

| size | naive | `CellCheckerboard` dim 1 | dim 2 | manual kernels |
|---|---|---|---|---|
| 128² | 26.6 | 20.7 (1.3×) | 19.6 (1.4×) | 0.75–1.15× |
| 256² | 26.6 | 13.6 (2.0×) | 13.2 (2.0×) | 1.1–1.3× |
| 512² | 59.4 | 25.5 (2.3×) | 24.2 (2.5×) | 1.1× |
| 1024² | 196.3 | 62.0 (3.2×) | 58.1 (3.4×) | 1.2× |

The compressed launch gains grow with the size, from 1.3× to 3.4×. The
hardcoded manual kernels reach only ~1.1–1.2×, so the generic helper path is
well ahead of them. Two runs agree to a few percent from 512² up; at 128² the
dim-1 speedup moved between 1.3× and 1.6×.

## GPU Linear Cell Window

`gpu_linear_cell_window.jl` times the two ways of addressing a halo array from a
GPU stencil kernel, in 2D, 3D and 4D: Cartesian (`cell_index`, neighbours at
`I ± eμ`) and linear-index (`linear_cell_window`, neighbours at `i ± sμ`), and
the main-branch `mod(·, 2)` colored map against the current `& 1`. Every variant
does the same arithmetic in the same order and the script checks that the
results are bit-identical. Metal (`--project=examples`) or CUDA:

```sh
julia --project=examples benchmark/gpu_linear_cell_window.jl
# CUDA: an environment with CUDA.jl, once, from the repository root
julia -e 'using Pkg; Pkg.activate("haloarrays-cuda"; shared=true); Pkg.add(["CUDA", "KernelAbstractions", "MPI"]); Pkg.develop(path=".")'
julia --project=@haloarrays-cuda benchmark/gpu_linear_cell_window.jl --backend=cuda --eltypes=Float32,Float64
```

Options: `--dims=2,3,4`, `--sizes=1024,128,48` (one per dim), `--eltypes=…`,
`--steps=20`, `--samples=10`, `--warmups=3`, `--csv=…`.

Reference (Apple M4 Max, Float32, default sizes; median speedups, results
bit-identical):

| case | colored: `& 1` vs `mod` | colored: linear vs `mod` | full pass: linear vs Cartesian |
|---|---|---|---|
| 2D 1024² | 1.6× | 2.1× | 1.6× |
| 3D 128³ | 1.3× | 2.1× | 2.5× |
| 4D 48⁴ | 1.2× | 2.1× | 3.5× |

The full-pass gain grows with the dimension (more neighbours per site, no
parity work to dilute it); smaller gains are expected on GPUs with native 64-bit
integer multiply (CUDA).

Rerun on the same machine with v0.16.1 (Julia 1.13.3, current Metal.jl,
2026-11-24), the linear-index gains are smaller and no longer grow with the
dimension; results still bit-identical:

| case | colored: `& 1` vs `mod` | colored: linear vs `mod` | full pass: linear vs Cartesian | full linear, ns/site |
|---|---|---|---|---|
| 2D 1024² | 1.5× | 1.7× | 1.4–2.0× | 0.050 |
| 3D 128³ | 1.2× | 1.7× | 1.5× | 0.098 |
| 4D 48⁴ | 1.1× | 1.3× | 1.4× | 0.211 |

Two runs agree to 1 % in 3D and 4D; in 2D the Cartesian full pass is noisy
(0.071 and 0.099 ns/site), hence the range.

## Checkerboard: in-place vs out-of-place

Measures the cost of the red-black coloring in a checkerboard stencil sweep, on
both the CPU (plain loops) and the Metal GPU (KernelAbstractions). Three ways to
do one full-grid update:

- `inplace` — two colored passes, read+write the same array (red-black
  Gauss-Seidel; the coloring is what makes the in-place update race-free);
- `outplace` — two colored passes, read `u` write `un` (still two passes);
- `jacobi` — one pass over the whole grid, read `u` write `un` (out-of-place
  needs no coloring, so the update is a single pass/launch).

```sh
julia --project=benchmark -t 8 benchmark/checkerboard_inout.jl --sizes=256,512,1024,2048
julia --project=examples  -t 8 benchmark/checkerboard_inout.jl --sizes=256,512,1024,2048   # also runs Metal
```

The CPU path always runs; the Metal variants run only if Metal loads (so
`--project=examples` measures both backends, `--project=benchmark` the CPU).

Reference (Apple M-series; Mcell/s, speedup vs in-place): the **single-pass
Jacobi is ~2-3× faster than the two-pass checkerboard on both backends** — on
the GPU because it is one kernel launch instead of two (launch latency dominates
small GPU workloads), on the CPU because a contiguous `@simd` pass replaces the
checkerboard's stride-2 access (which defeats vectorization). It shows on the CPU
even single-threaded, so it is the access pattern, not thread overhead. In-place
vs out-of-place at the same two-pass structure is within noise (~0.9×): the
coloring is the cost, not the aliasing. The trade is Jacobi's slower
per-iteration convergence and 2× the memory.

Rerun (M4 Max, v0.16.1, 8 threads, Metal; Mcell/s, speedup vs in-place):

| size | CPU in-place | CPU out-of-place | CPU Jacobi | GPU in-place | GPU out-of-place | GPU Jacobi |
|---|---|---|---|---|---|---|
| 256² | 1164 | 1350 (1.16×) | 2696 (2.3×) | 3781 | 3647 (0.96×) | 6377 (1.69×) |
| 512² | 4797 | 4684 (0.98×) | 9246 (1.9×) | 8335 | 12035 (1.44×) | 15510 (1.86×) |
| 1024² | 11349 | 11308 (1.00×) | 27502 (2.4×) | 22366 | 20849 (0.93×) | 24313 (1.09×) |
| 2048² | 15644 | 14485 (0.93×) | 43193 (2.8×) | 13715 | 13116 (0.96×) | 13257 (0.97×) |

The CPU claim holds (Jacobi 1.9–2.8×). On the GPU the single launch gains
1.2–1.9× up to 512², where the numbers vary from run to run (a second run gave
1.24× and 1.25×, with GPU in-place at 512² at 12578 instead of 8335), and
nothing from 1024² up, where the launch latency no longer dominates; at 2048²
the 8-thread CPU Jacobi pass is three times faster than the GPU. From 1024² up
the two runs agree to 1 %.

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
`synchronize_halo!(u; threads=true)` (an `OhMyThreads` `tforeach`; measured
before `ThreadsBackend` became the default).

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

**Rerun on v0.16.1** (M4 Max, Julia 1.13.3, 8 threads, `--boundary=repeating`,
manual timer; `tasks` is the `OhMyThreads.@tasks` loop, `threads` the
`@threads :static` loop; median):

| Config | tiles | serial | tasks | threads | winner |
| --- | ---: | ---: | ---: | ---: | --- |
| 2D 64², halo 1 | 4 | 0.25 µs | 17 µs | 15 µs | serial 58× |
| 2D 256², halo 1 | 16 | 2.5 µs | 21 µs | 17 µs | serial 7× |
| 2D 1024², halo 1 | 16 | 7.7 µs | 22 µs | 18 µs | serial 2.3× |
| 2D 1024², halo 5 | 16 | 20 µs | 17 µs | 20 µs | tasks 1.2× |
| 2D 2048², halo 3 | 64 | 59 µs | 28 µs | 26 µs | threads 2.3× |
| 3D 128³, halo 2 | 8 | 98 µs | 52 µs | 39 µs | threads 2.5× |

The README command above (4 threads, 2048², `tile-dims=4,1`, halo 8): serial
28 µs, tasks 13 µs, threads 21 µs. The serial sync is about three times faster
than in the table above and the fixed spawn+join cost is now ~15–20 µs (1–4 KB
per call) instead of ~130–180 µs, so the crossover moved down to roughly 20 µs
of exchange work: threading the sync pays from 1024² with a wide halo upward.
The serial default is still right for small tiles and thin halos.

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

Reference (128², 8 threads, M4 Max, v0.16.1; median, 0 B alloc for every case):
local `synchronize_halo!` ~0.13–0.17 µs in every mode, threaded (2×2 tiles)
~0.46 µs (periodic ~0.58 µs), a one-rank MPI `HaloArray` ~0.29 µs (periodic
~0.83 µs). With the per-cell rules the modes cost the same; none stands out.
The threaded sync is a small constant factor slower because it walks the tile
collection serially. On 4 ranks the MPI `synchronize_halo!` is ~36–38 µs and
`boundary_condition!` ~20 µs, both dominated by the timer's barrier and the
exchange rather than the fill.

## MPI Diagnostics

Benchmarks MPI exchange and synchronization paths with allocation reporting.
This is meant to catch hidden allocations or type-instability symptoms in the
low-level communication paths.

```sh
mpiexec -n 4 julia --project=benchmark benchmark/mpi_diagnostics.jl --owned-size=128,128 --boundary=repeating
```

The periodic cases benchmark pure neighbor exchange. The physical-boundary cases
benchmark exchange, boundary fill, and the combined `synchronize_halo!` path.

Reference (4 ranks, 128², M4 Max, v0.16.1; median, 0 B alloc for every case):
periodic `halo_exchange!` ~34 µs, start/finish ~34 µs, `synchronize_halo!`
~36 µs; physical: exchange ~36 µs, `boundary_condition!` ~23 µs,
`synchronize_halo!` ~38 µs.

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

Reference (M4 Max, v0.16.1, 512² global, 10 steps; median): local 1.59 ms, MPI
on 2 ranks 0.89 ms (1.8×), threaded on 2 tiles 1.13 ms (1.4×). Per step: the
stencil is ~156 µs local, ~93 µs per MPI rank, ~111 µs threaded; the halo
synchronisation is 0.6 µs local, 1.6 µs threaded, ~17 µs on MPI.

## Local And Threaded Heat Solver

Compares `LocalHaloArray` and `ThreadedHaloArray` without MPI. Here `--size`
is the full local problem size, so there is no owned/global distinction.
(`stencil.jl` above answers the same question with fewer knobs; this variant
adds split timings and the CLI/CSV machinery.)

```sh
JULIA_NUM_THREADS=2 julia --project=benchmark benchmark/heat_solver_local_threaded.jl --size=512,512 --tile-dims=2,1 --timer=benchmarktools
```

Reference (M4 Max, v0.16.1, 10 steps; median): local 1.55 ms, threaded on 2
tiles 1.11 ms (1.4×), results identical (max error 0.0). The manual and
BenchmarkTools timers agree to 1 %.
