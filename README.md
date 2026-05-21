# HaloArrays.jl

Distributed halo-array utilities for Julia stencil and PDE codes.

The package provides MPI-backed halo arrays, local no-MPI halo arrays, grouped multi-field arrays, boundary-condition helpers, reductions, gather, and HDF5 output utilities.

## Installation

```julia
using Pkg
Pkg.add("HaloArrays")
```

For development in this repository:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

MPI-backed features require an MPI runtime (OpenMPI or MPICH).

## Main Types

- `HaloArray`: MPI-backed array with owned interior cells and halo cells.
- `LocalHaloArray`: no-MPI halo array for local problems and boundary-condition-only workflows.
- `ThreadedHaloArray`: thread-local tiled halo array for shared-memory workflows.
- `ThreadedMultiHaloArray`: named collection of threaded halo fields.
- `MultiHaloArray`: named collection of MPI-backed halo fields.
- `LocalMultiHaloArray`: named collection of local halo fields.
- `CartesianTopology`: MPI Cartesian topology helper.

`size(u)` and `axes(u)` describe the global logical domain for every halo
container. Local owned data is accessed through `owned_size(u)`, `owned_axes(u)`,
and `interior_view(u)`.

Scalar indexing on `HaloArray` uses global indices, but it is local-only: it
works only for global cells owned by the current MPI rank and throws otherwise.
It is intended for diagnostics and setup, not stencil kernels or hot loops. Use
`interior_view`, `parent`, `owned_to_global_index`, and `global_to_storage_index`
when you need explicit local/global behavior.

## Typical Workflow

Most stencil updates follow this sequence:

1. Update only the interior region (`interior_view(u)`).
2. Call `synchronize_halo!(u)` before any stencil read that touches halo cells.
3. Repeat for each time step.

This keeps interior ownership explicit and halo validity predictable.

## Global and Local Semantics

The package separates logical array shape from owned storage:

- `size(u)` / `axes(u)`: global logical array shape.
- `owned_size(u)` / `owned_axes(u)`: cells owned by this rank or local backend.
- `interior_view(u)`: writable owned cells, excluding halos.
- `parent(u)`: raw storage including halos.
- `storage_size(u)`: raw storage size including halos.

For MPI-backed `HaloArray`, scalar `u[I...]` does not communicate. If `I` is not
owned by the current rank, it errors. This keeps expensive communication out of
generic indexing and makes performance-critical code use local views explicitly.

## Halo Exchange

The public exchange API is intentionally small:

```julia
halo_exchange!(u)          # blocking MPI exchange
start_halo_exchange!(u)    # begin async exchange
finish_halo_exchange!(u)   # finish async exchange
synchronize_halo!(u)       # make halos valid for stencil use
```

For `HaloArray`, `synchronize_halo!` performs exchange and then applies physical boundary conditions where the topology has `MPI.PROC_NULL` neighbors. For `LocalHaloArray`, it only applies boundary conditions.

Use the split async API (`start_halo_exchange!` + `finish_halo_exchange!`) when you want to overlap communication with independent computation.

## Boundary Conditions

Built-in boundary conditions are:

```julia
Reflecting()
Antireflecting()
Repeating()
Periodic()
```

Convenient symbols are also accepted:

```julia
:reflecting
:antireflecting
:repeating
:periodic
```

Examples:

```julia
HaloArray(Float64, (64, 64), 1, topology; boundary_condition=:periodic)

HaloArray(
    Float64,
    (64, 64),
    1,
    topology;
    boundary_condition=((Reflecting(), Repeating()), (:periodic, :periodic)),
)
```

Custom boundary conditions can be passed as a subtype or instance of `AbstractBoundaryCondition`; symbol registration is not used.

## Quick Example

```julia
using HaloArrays

u = LocalHaloArray(Float64, (64, 64), 1; boundary_condition=:periodic)
interior_view(u) .= 1.0
synchronize_halo!(u)
```

MPI-backed arrays require an MPI topology:

```julia
using MPI, HaloArrays

MPI.Init()
topology = CartesianTopology(MPI.COMM_WORLD, (0, 0); periodic=(true, true))
u = HaloArray(Float64, (64, 64), 1, topology; boundary_condition=:periodic)

interior_view(u) .= MPI.Comm_rank(MPI.COMM_WORLD)
synchronize_halo!(u)
MPI.Finalize()
```

## Local and Threaded Arrays

`LocalHaloArray` is the simplest option when running on a single process:

```julia
using HaloArrays

u = LocalHaloArray(Float64, (64, 64, 64), 2; boundary_condition=:repeating)
interior_view(u) .= 1.0
synchronize_halo!(u)
```

`ThreadedHaloArray` splits the domain into local tiles and exchanges halos across tiles using threads:

```julia
using HaloArrays

u = ThreadedHaloArray(Float64, (32, 32, 32), 2; dims=(2, 2, 2), boundary_condition=:periodic)
for tile in interior_view(u)
    tile .= 1.0
end
synchronize_halo!(u)
```

For multiple threaded fields, use `ThreadedMultiHaloArray`:

```julia
using HaloArrays

state = ThreadedMultiHaloArray(
    Float64,
    (32, 32, 32),
    2;
    dims=(2, 2, 2),
    boundary_conditions=(
        rho=:periodic,
        vel=((Reflecting(), Reflecting()), (:periodic, :periodic), (:periodic, :periodic)),
    ),
)

synchronize_halo!(state)
```

## When Multi Arrays Are Useful

`MultiHaloArray` and `ThreadedMultiHaloArray` are useful when a solver evolves multiple fields on the same grid (for example `rho`, `u`, `v`, `w`, `p`, `T`).

Use a multi-array container when:

1. All fields share the same geometry and halo width.
2. You want one synchronization call for the full state (`synchronize_halo!(state)`).
3. You want to reduce boilerplate and avoid passing many separate arrays.

Keep independent arrays when fields require different halo widths, different grid layouts, or different topologies.

## Core Utility Functions

- Domain and layout: `interior_size`, `storage_size`, `halo_width`, `global_size`, `interior_range`
- Index mapping: `owned_to_global_index`, `global_to_storage_index`
- Data movement and reduction: `gather_haloarray`, `mapreduce_haloarray_dims`
- HDF5 helpers: `create_haloarray_output_file`, `write_haloarray_timestep!`, `gather_and_save_haloarray`

`owned_to_global_index(u, I)` expects an owned interior index, excluding halo
offsets. `global_to_storage_index(u, I)` returns the full parent-storage index for
owned global cells and `nothing` for cells owned by another rank.

## Examples

Optional DiffEq examples use their own environment:

```bash
julia --project=examples -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
```

Local heat-diffusion examples:

```bash
julia --project=. examples/heat_diffusion_local_1d.jl
julia --project=. examples/heat_diffusion_local_2d.jl
julia --project=. examples/heat_diffusion_local_3d.jl
```

MPI heat-diffusion examples:

```bash
mpiexec -n 4 julia --project=. examples/heat_diffusion_mpi_1d.jl
mpiexec -n 4 julia --project=. examples/heat_diffusion_mpi_2d.jl
mpiexec -n 4 julia --project=. examples/heat_diffusion_mpi_3d.jl
```

The local, threaded, and MPI examples share their finite-difference update in
`examples/heat_diffusion_common.jl`.

DiffEq/OrdinaryDiffEq example:

```bash
julia --project=examples examples/ode_diffeq.jl
mpiexec -n 2 julia --project=examples examples/ode_diffeq.jl
julia --project=examples examples/local_and_threaded_diffeq.jl
```

On machines with fewer cores than ranks, use `--oversubscribe`:

```bash
mpiexec --oversubscribe -n 4 julia --project=. examples/heat_diffusion_mpi_2d.jl
```

## Tests

Unit tests:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

MPI tests:

```bash
HALOARRAYS_RUN_UNIT_TESTS=false mpiexec -n 2 julia --project=. test/runtests.jl
HALOARRAYS_RUN_UNIT_TESTS=false mpiexec -n 3 julia --project=. test/runtests.jl
HALOARRAYS_RUN_UNIT_TESTS=false mpiexec -n 4 julia --project=. test/runtests.jl
```

GitHub Actions runs unit tests and MPI tests with 2, 3, and 4 ranks.

## Benchmarks

Halo exchange benchmarks:

```bash
mpiexec -n 4 julia --project=. benchmarks/halo_exchange.jl
mpiexec -n 4 julia --project=. benchmarks/halo_exchange.jl --ndims=3 --local-size=64,64,64 --samples=50
```

## License

MIT. See [LICENSE](./LICENSE).
