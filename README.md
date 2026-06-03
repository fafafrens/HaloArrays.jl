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

- **Three interchangeable backends behind one API**
  - `LocalHaloArray` — single process, no MPI
  - `ThreadedHaloArray` — shared memory, tiled across threads
  - `HaloArray` — distributed over an MPI `CartesianTopology`
- **Explicit semantics** — global `size`/`axes`, `owned_size`/`owned_axes`,
  `interior_view`, `parent`; no hidden communication in `getindex`.
- **Multi-field containers** (`MultiHaloArray`, `LocalMultiHaloArray`,
  `ThreadedMultiHaloArray`, `ArrayOfHaloArray`) that exchange every field at once.
- **Boundary conditions** — periodic, reflecting, antireflecting, repeating, or custom.
- **Global reductions** — `sum`, `maximum`, `minimum`, `dot`, `norm`, … that
  Allreduce (MPI) or tile-reduce (threaded) automatically — plus `gather` and
  HDF5 output.
- **Cell & face loop helpers** and kernel regions, GPU-ready via KernelAbstractions.
- **Composes with the ecosystem** — a halo array is an `AbstractArray` with global
  reductions, so it works as an OrdinaryDiffEq state and as the vector in a
  matrix-free Krylov solve. The *same* code solves a Poisson problem serially and
  across MPI ranks: see [`examples/poisson/operator.jl`](examples/poisson/operator.jl)
  and [`examples/poisson/mpi.jl`](examples/poisson/mpi.jl).

## At a glance

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

Every exported type and function carries a docstring — in the REPL try
`?HaloArray`, `?synchronize_halo!`, `?interior_view`, `?ThreadedHaloArray`, or
`?Reflecting`. The sections below are the narrative guide; the docstrings are the
reference.

## Contents

- [Installation](#installation) · [Main types](#main-types) · [Typical workflow](#typical-workflow)
- Concepts: [Global vs. local semantics](#global-and-local-semantics) · [Halo exchange](#halo-exchange) · [Boundary conditions](#boundary-conditions)
- Backends: [Local & threaded arrays](#local-and-threaded-arrays) · [Multi-field containers](#when-multi-arrays-are-useful) · [Backend traits](#backend-traits) · [Thread backends](#thread-backends)
- Loops & kernels: [Face loops](#face-loops) · [Cell loops](#cell-loops) · [Kernel regions](#kernel-regions) · [Core utilities](#core-utility-functions)
- [Tutorials](#tutorials) · [Examples](#examples) · [Tests](#tests) · [Benchmarks](#benchmarks)

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
- `halo_backend`: backend trait helper for dispatching on MPI, local, or
  threaded storage.

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

For `ThreadedHaloArray`, the default `halo_exchange!`, `boundary_condition!`,
and `synchronize_halo!` use a serial tile loop because this is allocation-free
and fastest for small halo surfaces. Explicit threaded variants are also
available:

```julia
halo_exchange_threads!(u)
boundary_condition_threads!(u)
synchronize_halo_threads!(u)
```

Use these only after benchmarking the target problem. They can help for large
halo surfaces, but they add thread scheduling overhead.

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

## Backend Traits

Use `halo_backend(u)` when an algorithm needs separate implementations for MPI,
local, and threaded storage while still accepting collection wrappers. It returns
one of:

```julia
MPIHaloBackend()
LocalHaloBackend()
ThreadedHaloBackend()
```

The trait is defined for `HaloArray`, `LocalHaloArray`, `ThreadedHaloArray`,
`MultiHaloArray`, `ArrayOfHaloArray`, and `MaybeHaloArray`. Collections require
all fields to use the same backend, so backend dispatch is unambiguous.

```julia
update!(du, u, p) = update!(halo_backend(u), du, u, p)

update!(::Union{MPIHaloBackend,LocalHaloBackend}, du, u, p) = serial_update!(du, u, p)
update!(::ThreadedHaloBackend, du, u, p) = threaded_update!(du, u, p)
```

## Thread Backends

`halo_backend` describes *where* data lives; a **`ThreadBackend`** describes *how*
a `ThreadedHaloArray`'s per-tile work is dispatched across threads. Choose it at
construction with the `thread_backend` keyword (default `OhMyThreadsBackend()`):

```julia
u = ThreadedHaloArray(Float64, (32, 32), 1; dims=(2, 2),
                      boundary_condition=:periodic,
                      thread_backend=SerialBackend())

thread_backend(u)   # SerialBackend()
```

Built-in backends:

| Backend | Notes |
|---|---|
| `OhMyThreadsBackend()` | Default. Task-based, supports schedulers, composes/nests. |
| `SerialBackend()`      | Runs per-tile work on the calling thread — handy for debugging races and deterministic runs. |
| `PolyesterBackend()`   | Low-overhead `@batch`. Requires `using Polyester` (loaded via the `HaloArraysPolyesterExt` extension); constructing it otherwise raises a clear error. |

The backend is part of the array's concrete type, so dispatch is resolved at
compile time, and it propagates through `similar`, broadcast, and reductions —
every threaded operation (`synchronize_halo_threads!`, `fill!`, broadcast, `sum`,
`dot`, `any`/`all`, …) honours it automatically.

All threaded work funnels through just two methods. Define them for your own
`<:ThreadBackend` and the whole package uses it — no other changes needed:

```julia
struct MyBackend <: ThreadBackend end
HaloArrays.tile_foreach(::MyBackend, f, itr; scheduler=nothing) = ...   # parallel foreach
HaloArrays.tile_mapreduce(::MyBackend, f, op, itr; scheduler=nothing) = ...  # parallel mapreduce
```

## Face Loops

`FaceRanges(u)` gives the index ranges needed by finite-volume style face
updates. The ranges are expressed in parent-storage indices, so they are meant
for kernels that work directly with `parent(u)` and `parent(du)`.

```julia
ranges = FaceRanges(u)
e = get_unit_vector(ranges, dim)
udata = parent(u)
dudata = parent(du)

for IL in get_left_face(ranges, dim)
    IR = IL + e
    flux = numerical_flux(udata[IL], udata[IR])
    dudata[IR] += flux
end

for IL in get_internal_face(ranges)
    IR = IL + e
    flux = numerical_flux(udata[IL], udata[IR])
    dudata[IL] -= flux
    dudata[IR] += flux
end

for IL in get_right_face(ranges, dim)
    IR = IL + e
    flux = numerical_flux(udata[IL], udata[IR])
    dudata[IL] -= flux
end
```

The helper works for `HaloArray`, `LocalHaloArray`, `ThreadedHaloArray`, and
the collection wrappers. For collections, the face ranges describe the spatial
part only; select a field first, then apply the ranges to that field.

Use the colored face helpers when a face update writes both adjacent owned
cells in-place and must avoid same-color write conflicts:

```julia
for color in 0:1
    for IL in get_colored_internal_face(ranges, dim, color)
        IR = IL + e
        flux = numerical_flux(udata[IL], udata[IR])
        dudata[IL] -= flux
        dudata[IR] += flux
    end
end
```

## Cell Loops

`CellRanges(u)` gives the owned-cell range in parent-storage indices. For
ordinary out-of-place stencil updates, use the full owned-cell range:

```julia
ranges = CellRanges(u)
udata = parent(u)
vdata = parent(v)

for I in get_owned_cells(ranges)
    vdata[I] = stencil_value(udata, I)
end
```

For nearest-neighbor in-place red-black updates, use the colored cell subranges.
Each color is returned as a tuple of strided `CartesianIndices`, so the inner CPU
loop has no parity branch:

```julia
for color in 0:1
    for cells in get_colored_owned_cell_ranges(ranges, color)
        for I in cells
            udata[I] = stencil_value(udata, I)
        end
    end
end
```

Cell colors use `mod(sum(Tuple(I)), 2)`. This two-color pattern is intended for
nearest-neighbor stencils. Wider stencils may need a different coloring.

## Kernel Regions

The range APIs are also available as compact launch metadata for GPU-style or
KernelAbstractions-style kernels.

For face kernels, use `FaceKernelRegion` or `ColoredFaceKernelRegion`:

```julia
region = get_colored_internal_face_region(FaceRanges(u), dim, color)

J = @index(Global, Cartesian)
IL = region.first + CartesianIndex((Tuple(J) .- 1) .* Tuple(region.stride))
IR = IL + region.offset
```

For cell kernels, use `CellKernelRegion` or `ColoredCellKernelRegion`. The
colored cell region compresses one launch dimension by two, reconstructs the
physical cell with `cell_index`, and uses `is_cell_index_inbounds` for the final
upper-bound check:

```julia
region = get_colored_owned_cell_region(CellRanges(u), color, Dim(2))

I = cell_index(region, @index(Global, NTuple))
if is_cell_index_inbounds(region, I)
    udata[I...] = stencil_value(udata, I)
end
```

This lets GPU kernels launch roughly half as many threads per color while
keeping the checkerboard rule shared with the CPU cell-range API.

## Core Utility Functions

- Domain and layout: `interior_size`, `storage_size`, `halo_width`, `global_size`, `interior_range`
- Index mapping: `owned_to_global_index`, `global_to_storage_index`
- Face loops: `FaceRanges`, `get_left_face`, `get_internal_face`, `get_right_face`
- Cell loops: `CellRanges`, `get_owned_cells`, `get_colored_owned_cell_ranges`
- Kernel regions: `FaceKernelRegion`, `CellKernelRegion`, `ColoredFaceKernelRegion`, `ColoredCellKernelRegion`
- Backend dispatch: `halo_backend`, `MPIHaloBackend`, `LocalHaloBackend`, `ThreadedHaloBackend`
- Data movement and reduction: `gather_haloarray`, `mapreduce_haloarray_dims`
- HDF5 helpers: `create_haloarray_output_file`, `write_haloarray_timestep!`, `gather_and_save_haloarray`

`owned_to_global_index(u, I)` expects an owned interior index, excluding halo
offsets. `global_to_storage_index(u, I)` returns the full parent-storage index for
owned global cells and `nothing` for cells owned by another rank.

## Tutorials

Step-by-step tutorials are provided in the `examples/` directory.
Each file is self-contained and runnable.

| File | What it covers |
|---|---|
| [`tutorials/local.jl`](examples/tutorials/local.jl) | Storage layout, boundary conditions, `CellRanges`/`FaceRanges`, heat equation, `LocalMultiHaloArray`, `ThreadedHaloArray`, `ArrayOfHaloArray` |
| [`tutorials/mpi.jl`](examples/tutorials/mpi.jl) | `CartesianTopology`, `HaloArray`, halo exchange (blocking and async), global reductions, gather, multi-field MPI, distributed heat equation |
| [`tutorials/threaded.jl`](examples/tutorials/threaded.jl) | `ThreadedHaloArray` tile layout, tile loop pattern, synchronisation, threaded Burgers equation, `ThreadedMultiHaloArray` |
| [`tutorials/broadcast.jl`](examples/tutorials/broadcast.jl) | Interior-only semantics, in-place and allocating broadcast, mixing with scalars and plain arrays, `MultiHaloArray` and `ThreadedHaloArray` broadcast, unsupported patterns |
| [`tutorials/gpu.jl`](examples/tutorials/gpu.jl) | Moving a `LocalHaloArray` to Metal/GPU, `KernelAbstractions` kernels, `CellKernelRegion`, `ColoredCellKernelRegion`, `FaceKernelRegion`, GPU heat equation |
| [`tutorials/diffeq.jl`](examples/tutorials/diffeq.jl) | `OrdinaryDiffEq.jl` integration, `synchronize_halo!` contract in the RHS, scalar decay, heat equation with Tsit5, multi-field ODE state, `ThreadedHaloArray` as ODE state |

**Running the tutorials:**

```bash
# No MPI — runs on a single process
julia --project=. examples/tutorials/local.jl
julia --project=. -t 4 examples/tutorials/broadcast.jl
julia --project=. -t 4 examples/tutorials/threaded.jl

# MPI — requires mpiexec
mpiexec -n 4 julia --project=. examples/tutorials/mpi.jl

# GPU — requires Metal.jl (macOS) or equivalent KernelAbstractions backend
julia --project=. examples/tutorials/gpu.jl

# OrdinaryDiffEq — requires the examples environment
julia --project=examples -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=examples examples/tutorials/diffeq.jl
```

## Examples

Optional DiffEq examples use their own environment:

```bash
julia --project=examples -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
```

Examples are grouped by topic: `heat/`, `finite_volume/`, `hydro/`, `lattice/`,
and `poisson/` (plus `tutorials/`). Heat diffusion (the simplest stencil) on
local arrays — `heat/local.jl` runs the 1-D, 2-D, and 3-D cases, and
`heat/local_vs_threaded.jl` solves the same 2-D problem on `LocalHaloArray` vs
`ThreadedHaloArray`, by hand and via OrdinaryDiffEq:

```bash
julia --project=. examples/heat/local.jl
julia --project=examples -t 4 examples/heat/local_vs_threaded.jl
```

MPI heat diffusion (`heat/mpi.jl` runs the 1-D, 2-D, and 3-D cases):

```bash
mpiexec -n 4 julia --project=. examples/heat/mpi.jl
```

The local, threaded, and MPI heat examples share their finite-difference update
in `examples/heat/common.jl`.

On machines with fewer cores than ranks, use `--oversubscribe`:

```bash
mpiexec --oversubscribe -n 4 julia --project=. examples/heat/mpi.jl
```

Matrix-free linear operators — wrap a stencil as a `SciMLOperators.FunctionOperator`
and solve a Poisson problem with the coordinate-free Krylov solvers in
[`examples/poisson/krylov_solvers.jl`](examples/poisson/krylov_solvers.jl) (CG, BiCGStab, GMRES).
The same operator and solvers run serially and across MPI ranks, because `dot`/`norm`
are global reductions:

```bash
julia --project=examples examples/poisson/operator.jl       # serial, 3 solvers, O(h²) check
mpiexec -n 4 julia --project=examples examples/poisson/mpi.jl  # identical solve, distributed
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
mpiexec -n 4 julia --project=. benchmarks/halo_exchange.jl --ndims=3 --owned-size=64,64,64 --samples=50
```

See [`benchmarks/README.md`](benchmarks/README.md) for boundary, reduction,
gather/HDF5, heat-solver, and threaded synchronization benchmarks.

## License

MIT. See [LICENSE](./LICENSE).
