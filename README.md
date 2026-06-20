# HaloArrays.jl

[![CI](https://github.com/fafafrens/HaloArrays.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/fafafrens/HaloArrays.jl/actions/workflows/ci.yml)
[![Docs](https://github.com/fafafrens/HaloArrays.jl/actions/workflows/docs.yml/badge.svg)](https://github.com/fafafrens/HaloArrays.jl/actions/workflows/docs.yml)
[![docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://fafafrens.github.io/HaloArrays.jl/dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*One halo-array API for serial, shared-memory, and distributed stencil & PDE codes in Julia.*

HaloArrays.jl gives you arrays with **ghost (halo) cells** and a uniform interface
for filling them — whether the data lives in one process, is tiled across threads,
or is decomposed across MPI ranks. Write a stencil once against `interior_view`,
`synchronize_halo!`, and global `size`/`axes`, and run it **unchanged on any
backend**. `synchronize_halo!` is the only place halos are filled, so the hot path
stays local and halo validity is predictable.

**The idea:** automate the two error-prone parts of a parallel stencil code — halo
synchronization and boundary conditions — and otherwise stay out of your way. Beyond
`synchronize_halo!`, everything is a plain array: drop to `parent(u)` for the raw
storage and use `interior_range`/`interior_view` (plus the face/cell range utilities)
to address your own slice, so your kernel is just a loop over arrays with no framework
to fight.

## Features

- **Three interchangeable backends behind one API** — `LocalHaloArray` (single
  process), `ThreadedHaloArray` (shared-memory tiles), `HaloArray` (MPI over a
  `CartesianTopology`).
- **Explicit semantics** — global `size`/`axes`, `interior_size`/`interior_view`,
  `parent`; no hidden communication in `getindex`.
- **Multi-field containers** (`MultiHaloArray`, `ArrayOfHaloArray`) exchanged at once.
- **Boundary conditions** — periodic, reflecting, antireflecting, repeating,
  custom, and *coupled* (characteristic) conditions across fields.
- **Global reductions** — `sum`/`maximum`/`dot`/`norm` Allreduce (MPI) or
  tile-reduce (threaded) automatically; plus `gather` and HDF5 output.
- **GPU-ready** via KernelAbstractions and `Adapt` (`cu(halo)`); pluggable thread
  backends (OhMyThreads / Serial / Polyester); composes as an OrdinaryDiffEq state
  and a matrix-free Krylov vector.

## Installation

```julia
using Pkg
Pkg.add("HaloArrays")
```

MPI-backed features require an MPI runtime (OpenMPI or MPICH).

## Quick start

```julia
using HaloArrays
u = LocalHaloArray(Float64, (64, 64), 1; boundary_condition=:periodic)
interior_view(u) .= 1.0      # write interior cells
synchronize_halo!(u)         # fill ghost cells (here: periodic wrap)
```

A stencil written against `interior_view`/`synchronize_halo!` goes from serial to
threaded to distributed **just by swapping the constructor** — the loop body never
changes:

```julia
LocalHaloArray(Float64, (128, 128), 1; boundary_condition=:periodic)
ThreadedHaloArray(Float64, (64, 64), 1; dims=(2, 2), boundary_condition=:periodic)
HaloArray(Float64, (128, 128), 1, CartesianTopology(comm, (0, 0); periodic=(true, true)))
```

See [`examples/heat`](examples/heat/) for one heat solver run unchanged across all
three backends (and a CPU-vs-GPU KernelAbstractions version).

## Documentation

- **[Documentation site](https://fafafrens.github.io/HaloArrays.jl)** — guide + full API reference.
- **[examples/](examples/)** — worked solvers: heat, finite volume, hydro, relativistic hydro, lattice field theory, Poisson.

Every exported symbol has a docstring — in the REPL try `?synchronize_halo!`, `?ThreadedHaloArray`.

## Tests

```julia
julia --project=. -e 'using Pkg; Pkg.test()'                                      # unit
HALOARRAYS_RUN_UNIT_TESTS=false mpiexec -n 4 julia --project=. test/runtests.jl   # MPI
```

CI runs unit tests (1 / 2 / 4 threads) and MPI tests (2 / 4 ranks).

## Related work & acknowledgements

I wrote HaloArrays.jl to fit my own needs, but the packages below are well done and
more mature — if your needs differ, one of them may suit you better. HaloArrays.jl
was directly inspired by [Chmy.jl](https://github.com/PTsolvers/Chmy.jl) and sits
alongside Julia's excellent parallel-arrays ecosystem:

- **[Chmy.jl](https://github.com/PTsolvers/Chmy.jl)** — architecture-agnostic
  (CPU/GPU) fields on structured grids with distributed MPI; a direct inspiration
  for this package — the way it exchanges halos between ranks is, I think, brilliant.
- **[ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl)** —
  high-performance stencil kernels on CPU/GPU (pairs with ImplicitGlobalGrid.jl
  for distributed halo updates).
- **[ImplicitGlobalGrid.jl](https://github.com/omlins/ImplicitGlobalGrid.jl)** —
  distributed global grids with MPI halo updates (the companion to ParallelStencil).
- **[Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl)** — its `Field`
  abstraction is a halo-region array that runs multi-architecture (CPU/GPU) and
  distributed; an influential "fields with halos" design.
- **[PencilArrays.jl](https://github.com/jipolanco/PencilArrays.jl)** — distributed
  N-D arrays with pencil decompositions and halo exchange (and parallel FFTs via
  [PencilFFTs.jl](https://github.com/jipolanco/PencilFFTs.jl)).
- **[MPIHaloArrays.jl](https://github.com/smillerc/MPIHaloArrays.jl)** — arrays with
  halo regions over MPI; closest in spirit and name.
- **[PartitionedArrays.jl](https://github.com/fverdugo/PartitionedArrays.jl)** —
  partitioned vectors and sparse matrices for distributed linear algebra.

Where HaloArrays.jl focuses: *one* halo-array API that is identical across serial,
threaded, and MPI backends (swap only the constructor), GPU-ready via
KernelAbstractions and `Adapt`, with built-in global reductions and a vector-space
interface so the same array drops straight into LinearSolve/Krylov and
OrdinaryDiffEq.

## License

MIT. See [LICENSE](./LICENSE).
