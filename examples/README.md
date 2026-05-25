# Examples

This folder contains small runnable examples for local, threaded, MPI-backed,
and DiffEq workflows.

## Setup

The package test project is enough for the local and MPI heat examples:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

DiffEq examples use optional packages. From the repository root, create the
examples environment once:

```bash
julia --project=examples -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
```

The generated `examples/Manifest.toml` is intentionally ignored by git.

## Local Heat Diffusion

These examples use `LocalHaloArray` with the same interior/halo API as the MPI
array:

```bash
julia --project=. examples/heat_diffusion_local_1d.jl
julia --project=. examples/heat_diffusion_local_2d.jl
julia --project=. examples/heat_diffusion_local_3d.jl
```

## Local and Threaded Heat Diffusion

`local_and_threaded_halo_arrays.jl` runs the same 2D heat update with
`LocalHaloArray` and `ThreadedHaloArray`.

```bash
julia --project=. examples/local_and_threaded_halo_arrays.jl
```

## MPI Heat Diffusion

These examples use `HaloArray`, MPI Cartesian topologies, periodic halo
exchange, and the shared stencil code in `heat_diffusion_common.jl`.

```bash
mpiexec -n 4 julia --project=. examples/heat_diffusion_mpi_1d.jl
mpiexec -n 4 julia --project=. examples/heat_diffusion_mpi_2d.jl
mpiexec -n 4 julia --project=. examples/heat_diffusion_mpi_3d.jl
```

To save a gathered final snapshot to HDF5:

```bash
mpiexec -n 4 julia --project=. -e 'include("examples/heat_diffusion_mpi_2d.jl"); run_mpi_heat_2d(save_hdf5=true)'
```

## Burgers Finite Volume

These examples solve 1D inviscid Burgers with a conservative finite-volume
update and Rusanov face fluxes.

```bash
julia --project=. examples/burgers_1d.jl
mpiexec -n 4 julia --project=. examples/burgers_mpi_1d.jl
```

The same semi-discrete finite-volume RHS can also be stepped by OrdinaryDiffEq:

```bash
julia --project=examples examples/burgers_diffeq_1d.jl
mpiexec -n 4 julia --project=examples examples/burgers_diffeq_1d.jl
```

## Linear Advection With DiffEq

This example solves periodic 1D linear advection with an upwind finite-volume
flux and steps the semi-discrete system with OrdinaryDiffEq.

```bash
julia --project=examples examples/linear_advection_diffeq_1d.jl
mpiexec -n 4 julia --project=examples examples/linear_advection_diffeq_1d.jl
```

## DiffEq Examples

The DiffEq examples use `DiffEqBase` and `OrdinaryDiffEq` from
`examples/Project.toml`.

```bash
julia --project=examples examples/ode_diffeq.jl
mpiexec -n 2 julia --project=examples examples/ode_diffeq.jl
julia --project=examples examples/local_and_threaded_diffeq.jl
julia --project=examples examples/heat_diffusion_diffeq.jl
```

## Notes

- If the machine has fewer cores than MPI ranks, add `--oversubscribe` to
  `mpiexec`.
- The heat examples share the finite-difference kernels in
  `heat_diffusion_common.jl`.
- The examples avoid plotting packages by default so they run in headless
  environments.
