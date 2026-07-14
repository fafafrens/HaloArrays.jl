# HaloArrays.jl

*One halo-array API for serial, shared-memory, and distributed stencil & PDE codes in Julia.*

HaloArrays.jl gives you arrays with **ghost (halo) cells** and a single, uniform
interface for filling them — whether the data lives in one process, is tiled
across threads, or is decomposed across MPI ranks. Write a stencil or solver
once against `interior_view`, `synchronize_halo!`, and global `size`/`axes`
semantics, and run it **unchanged on any backend**.

The design keeps *logical* and *interior* data distinct and keeps communication out
of indexing: `synchronize_halo!` is the only place halos are filled, so halo
validity is predictable and the hot path stays local.

## Features

- **Three interchangeable backends behind one API**
  - [`LocalHaloArray`](@ref) — single process, no MPI
  - [`ThreadedHaloArray`](@ref) — shared memory, tiled across threads
  - [`HaloArray`](@ref) — distributed over an MPI [`CartesianTopology`](@ref)
- **Explicit semantics** — global `size`/`axes`, [`interior_size`](@ref)/[`interior_axes`](@ref),
  [`interior_view`](@ref), `parent`; no hidden communication in `getindex`.
- **Multi-field containers** ([`MultiHaloArray`](@ref), `LocalMultiHaloArray`,
  `ThreadedMultiHaloArray`, [`ArrayOfHaloArray`](@ref)) that exchange every field at once.
- **Boundary conditions** — periodic, reflecting, antireflecting, repeating,
  custom, and *coupled* (characteristic) conditions across fields.
- **Global reductions** — `sum`, `maximum`, `minimum`, `dot`, `norm`, … that
  Allreduce (MPI) or tile-reduce (threaded) automatically — plus `gather` and HDF5 output.
- **Cell & face loop helpers** and kernel regions, GPU-ready via KernelAbstractions.
- **Composes with the ecosystem** — a halo array is an `AbstractArray` with global
  reductions, so it works as an OrdinaryDiffEq state and as the vector in a
  matrix-free Krylov solve.

## Installation

```julia
using Pkg
Pkg.add("HaloArrays")
```

MPI-backed features require an MPI runtime (OpenMPI or MPICH).

## At a glance

```julia
using HaloArrays
u = LocalHaloArray(Float64, (64, 64), 1; boundary_condition=:periodic)
interior_view(u) .= 1.0      # write interior cells
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

## Where to go next

- **[Guide](guide.md)** — concepts: ownership semantics, halo exchange, boundary
  conditions, the backends, multi-field containers, and the loop/kernel helpers.
- **[Examples](examples.md)** — runnable tutorials and worked solvers (heat,
  finite volume, hydro, relativistic hydro, lattice field theory, Poisson,
  Schrödinger dynamics).
- **[API reference](api/types.md)** — every exported type and function.
