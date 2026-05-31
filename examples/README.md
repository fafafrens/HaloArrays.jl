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

## Local CPU and Metal GPU Heat Diffusion

`local_cpu_gpu_heat_2d.jl` solves the same 2D periodic heat problem with a
CPU-backed `LocalHaloArray` and a Metal-backed `LocalHaloArray`. The GPU update
uses KernelAbstractions, colored face regions for flux accumulation, and the
owned-cell region API for the final cell update.

```bash
julia --project=examples examples/local_cpu_gpu_heat_2d.jl
```

This example requires an Apple GPU supported by Metal.jl.

## Metal Monte Carlo

These examples use `LocalHaloArray` storage on Metal devices together with
colored cell regions for checkerboard updates.

```bash
julia --project=examples examples/local_metal_phi4_2d.jl
julia --project=examples examples/local_metal_phi4_2d_philox.jl
julia --project=examples examples/local_metal_su2_wilson_2d_arrayofhaloarray_philox.jl
```

The Philox examples use stateless per-site random numbers, so they avoid
allocating random-number arrays during each sweep. These examples require an
Apple GPU supported by Metal.jl.

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

## Ideal Hydro

These examples start a 2D non-relativistic ideal hydrodynamics setup using
conservative variables, periodic halos, first-order Rusanov fluxes, and
`OrdinaryDiffEq`.

```bash
julia --project=examples examples/ideal_hydro_local_2d.jl
JULIA_NUM_THREADS=4 julia --project=examples examples/ideal_hydro_threaded_2d.jl
mpiexec -n 4 julia --project=examples examples/ideal_hydro_mpi_2d.jl
```

To write an SVG comparing the initial and final density/pressure fields:

```bash
julia --project=examples examples/ideal_hydro_plot_2d.jl
```

The threaded hydro example chooses a 2D tile decomposition that divides the
problem size and whose tile count is equal to `Threads.nthreads()`. If
`tile_dims` is supplied manually, it must satisfy
`prod(tile_dims) == Threads.nthreads()`.

## Linear Advection With DiffEq

This example solves periodic 1D linear advection with an upwind finite-volume
flux and steps the semi-discrete system with OrdinaryDiffEq.

```bash
julia --project=examples examples/linear_advection_diffeq_1d.jl
mpiexec -n 4 julia --project=examples examples/linear_advection_diffeq_1d.jl
```

## Scalar Field Heat Bath

These examples run a simple 2D free scalar field heat-bath update with
checkerboard sweeps and periodic halo synchronization. The local/threaded
example also shows named fields with `MultiHaloArray` and replica fields with
`ArrayOfHaloArray`; the MPI example shows the same collection wrappers across
MPI ranks.

```bash
julia --project=. examples/scalar_field_heatbath_local_threaded_2d.jl
mpiexec -n 4 julia --project=. examples/scalar_field_heatbath_mpi_2d.jl
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
