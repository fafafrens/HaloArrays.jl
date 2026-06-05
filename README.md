# HaloArrays.jl

*One halo-array API for serial, shared-memory, and distributed stencil & PDE codes in Julia.*

HaloArrays.jl gives you arrays with **ghost (halo) cells** and a single, uniform
interface for filling them — whether the data lives in one process, is tiled
across threads, or is decomposed across MPI ranks. Write a stencil or solver
once against `interior_view`, `synchronize_halo!`, and global `size`/`axes`
semantics, and run it **unchanged on any backend**.

The design keeps *logical* and *owned* data distinct and keeps communication out
of indexing: `synchronize_halo!` is the only place halos are filled, so halo
validity is predictable and the hot path stays local.

## Features

- **Three interchangeable backends behind one API** — `LocalHaloArray` (single
  process), `ThreadedHaloArray` (shared-memory tiles), `HaloArray` (MPI over a
  `CartesianTopology`).
- **Explicit semantics** — global `size`/`axes`, `owned_size`/`owned_axes`,
  `interior_view`, `parent`; no hidden communication in `getindex`.
- **Multi-field containers** (`MultiHaloArray`, `ArrayOfHaloArray`, …) that
  exchange every field at once.
- **Boundary conditions** — periodic, reflecting, antireflecting, repeating,
  custom, and *coupled* (characteristic) conditions across fields.
- **Global reductions** — `sum`, `maximum`, `dot`, `norm`, … that Allreduce (MPI)
  or tile-reduce (threaded) automatically — plus `gather` and HDF5 output.
- **Cell & face loop helpers** and kernel regions, GPU-ready via KernelAbstractions.
- **Pluggable thread backends** (OhMyThreads / Serial / Polyester) and composes
  with the ecosystem (OrdinaryDiffEq state, matrix-free Krylov vector).

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
interior_view(u) .= 1.0      # write owned cells
synchronize_halo!(u)         # fill ghost cells (here: periodic wrap)
```

The identical stencil code goes distributed just by swapping the constructor —
`synchronize_halo!` becomes an MPI halo exchange, nothing else changes:

```julia
using MPI, HaloArrays
MPI.Init()
topo = CartesianTopology(MPI.COMM_WORLD, (0, 0); periodic=(true, true))
u = HaloArray(Float64, (64, 64), 1, topo; boundary_condition=:periodic)
interior_view(u) .= MPI.Comm_rank(MPI.COMM_WORLD)
synchronize_halo!(u)
MPI.Finalize()
```

## Documentation

- **[Documentation site](https://fafafrens.github.io/HaloArrays.jl)** — guide and
  full API reference.
- **[Guide](https://fafafrens.github.io/HaloArrays.jl/guide/)** — ownership
  semantics, halo exchange, boundary conditions, the backends, multi-field
  containers, and the loop/kernel helpers.
- **[examples/](examples/)** — runnable tutorials and worked solvers (heat,
  finite volume, hydro, relativistic hydro, lattice field theory, Poisson); see
  [`examples/README.md`](examples/README.md).

Every exported type and function has a docstring — in the REPL try
`?synchronize_halo!`, `?ThreadedHaloArray`, `?Reflecting`.

## Tests

```julia
julia --project=. -e 'using Pkg; Pkg.test()'        # unit tests
```

```bash
HALOARRAYS_RUN_UNIT_TESTS=false mpiexec -n 4 julia --project=. test/runtests.jl   # MPI tests
```

CI runs unit tests (1 / 2 / 4 threads) and MPI tests (2, 3, 4 ranks).

## License

MIT. See [LICENSE](./LICENSE).
