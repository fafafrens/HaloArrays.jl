# HaloArrays.jl

Distributed halo-array utilities for Julia stencil and PDE codes.

The package provides MPI-backed halo arrays, local no-MPI halo arrays, grouped multi-field arrays, boundary-condition helpers, reductions, gather, and HDF5 output utilities.

## Main Types

- `HaloArray`: MPI-backed array with owned interior cells and halo cells.
- `LocalHaloArray`: no-MPI halo array for local problems and boundary-condition-only workflows.
- `MultiHaloArray`: named collection of MPI-backed halo fields.
- `LocalMultiHaloArray`: named collection of local halo fields.
- `CartesianTopology`: MPI Cartesian topology helper.

`HaloArray` intentionally does not behave like a global Julia array for indexing. Use `interior_view`, `parent`, `local_to_global_index`, and `global_to_local_index` for explicit local/global behavior. `LocalHaloArray` supports array-like indexing over its interior.

## Halo Exchange

The public exchange API is intentionally small:

```julia
halo_exchange!(u)          # blocking MPI exchange
start_halo_exchange!(u)    # begin async exchange
finish_halo_exchange!(u)   # finish async exchange
synchronize_halo!(u)       # make halos valid for stencil use
```

For `HaloArray`, `synchronize_halo!` performs exchange and then applies physical boundary conditions where the topology has `MPI.PROC_NULL` neighbors. For `LocalHaloArray`, it only applies boundary conditions.

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
```

## Examples

Local heat-diffusion examples:

```bash
julia --project=. examples/heat_diffusion_local_1d.jl
julia --project=. examples/heat_diffusion_local_2d.jl
julia --project=. examples/heat_diffusion_local_3d.jl
```

MPI heat-diffusion examples:

```bash
mpiexec -n 4 julia --project=. examples/tes_heat.jl
mpiexec -n 4 julia --project=. examples/tes_heat_2d.jl
mpiexec -n 4 julia --project=. examples/tes_heat_3d.jl
```

The local examples share their finite-difference update in `examples/heat_diffusion_common.jl`.

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
