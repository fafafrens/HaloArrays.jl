# Examples

Runnable examples live in the [`examples/`](https://github.com/fafafrens/HaloArrays.jl/tree/main/examples)
directory and are grouped by topic: `tutorials/`, `heat/`, `finite_volume/`,
`hydro/`, `lattice/`, and `poisson/`. See
[`examples/README.md`](https://github.com/fafafrens/HaloArrays.jl/blob/main/examples/README.md)
for the full list and run commands.

## Tutorials

Step-by-step, self-contained walkthroughs (start here):

| File | What it covers |
|---|---|
| `tutorials/local.jl` | Storage layout, boundary conditions, `CellRanges`/`FaceRanges`, heat equation, `LocalMultiHaloArray`, `ThreadedHaloArray`, `ArrayOfHaloArray` |
| `tutorials/mpi.jl` | `CartesianTopology`, `HaloArray`, halo exchange (blocking and async), global reductions, gather, distributed heat equation |
| `tutorials/threaded.jl` | `ThreadedHaloArray` tile layout, tile loop pattern, synchronisation, threaded Burgers, `ThreadedMultiHaloArray` |
| `tutorials/broadcast.jl` | Interior-only broadcast semantics, in-place vs allocating, collections |
| `tutorials/gpu.jl` | Moving a `LocalHaloArray` to Metal/GPU, KernelAbstractions kernels, kernel regions |
| `tutorials/diffeq.jl` | `OrdinaryDiffEq.jl` integration, the `synchronize_halo!` contract in the RHS |

```bash
julia --project=. examples/tutorials/local.jl
julia --project=. -t 4 examples/tutorials/threaded.jl
mpiexec -n 4 julia --project=. examples/tutorials/mpi.jl
julia --project=examples examples/tutorials/diffeq.jl
```

## Worked solvers

- **`heat/`** — finite-difference heat diffusion (local 1-D/2-D/3-D, MPI, threaded, CPU-vs-GPU).
- **`finite_volume/`** — Burgers, linear advection, a *coupled* characteristic
  boundary condition, and **special relativistic hydrodynamics** (1-D → 3-D,
  charge-free μ=0 and charge-carrying (T, μ) formulations).
- **`hydro/`** — 2-D non-relativistic ideal hydrodynamics (local, threaded, MPI, plotting).
- **`lattice/`** — scalar-field and SU(2) Wilson lattice Monte Carlo (threaded and Metal).
- **`poisson/`** — matrix-free CG/BiCGStab/GMRES on a halo array; the same operator
  and solvers run serially and across MPI ranks because `dot`/`norm` are global reductions.

```bash
julia --project=. examples/heat/local.jl
julia --project=. examples/finite_volume/relativistic_hydro_mu0_2d.jl
julia --project=examples examples/poisson/operator.jl
mpiexec -n 4 julia --project=examples examples/poisson/mpi.jl
```

## Tests and benchmarks

```bash
# unit tests
julia --project=. -e 'using Pkg; Pkg.test()'

# MPI tests (2, 3, 4 ranks)
HALOARRAYS_RUN_UNIT_TESTS=false mpiexec -n 4 julia --project=. test/runtests.jl
```

Benchmarks live in [`benchmark/`](https://github.com/fafafrens/HaloArrays.jl/tree/main/benchmark):
two quick-start throughput harnesses (stencil Mcell/s, MPI exchange/overlap)
plus a CLI micro-suite (halo exchange, boundary conditions, reductions,
gather/HDF5, heat solvers, threaded synchronization, thread backends) — see its
`README.md`.
