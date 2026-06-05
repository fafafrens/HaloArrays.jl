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

An array with one ghost layer — write the owned cells, then fill the halo:

```julia
using HaloArrays
u = LocalHaloArray(Float64, (64, 64), 1; boundary_condition=:periodic)
interior_view(u) .= 1.0      # write owned cells
synchronize_halo!(u)         # fill ghost cells (here: periodic wrap)
```

### One solver, every backend

Write a stencil once and run it **unchanged** on a single process, on
shared-memory tiles, and across MPI ranks. The only backend-specific code is how
the owned-cell loop is traversed — flat arrays iterate their `parent`, tiled
arrays iterate per tile — so two `heat_step!` methods cover all three, and
multiple dispatch picks the right one:

```julia
using HaloArrays, MPI
MPI.Init()

# Flat backends (LocalHaloArray + the MPI HaloArray) share one method;
# the tiled ThreadedHaloArray gets a second. The kernel is shared.
function heat_step!(out, u, α, dt, dx)
    synchronize_halo!(u)
    _heat_kernel!(parent(out), parent(u), interior_range(u), α, dt, dx, Val(ndims(u)))
end
function heat_step!(out::ThreadedHaloArray, u::ThreadedHaloArray, α, dt, dx)
    synchronize_halo!(u)
    for t in 1:tile_count(u)
        _heat_kernel!(tile_parent(out, t), tile_parent(u, t), interior_range(u), α, dt, dx, Val(ndims(u)))
    end
end
function _heat_kernel!(out, u, rng, α, dt, dx, ::Val{N}) where {N}
    e = CartesianIndex.(versors(Val(N)))
    @inbounds for I in CartesianIndices(rng)
        lap = sum(u[I + e[d]] - 2u[I] + u[I - e[d]] for d in 1:N) / dx^2
        out[I] = u[I] + α * dt * lap
    end
end

# The same solver on three different arrays — just loop over them.
for u in (
    LocalHaloArray(Float64, (128, 128), 1; boundary_condition=:periodic),
    ThreadedHaloArray(Float64, (64, 64), 1; dims=(2, 2), boundary_condition=:periodic),
    HaloArray(Float64, (128, 128), 1,
        CartesianTopology(MPI.COMM_SELF, (1, 1); periodic=(true, true));
        boundary_condition=:periodic),
)
    v = similar(u)
    fill_from_global_indices!(u) do I                  # centred Gaussian
        exp(-((I[1] - 64)^2 + (I[2] - 64)^2) / 200)
    end
    for _ in 1:200
        heat_step!(v, u, 0.1, 0.1, 1.0)
        u, v = v, u
    end
    println(rpad(nameof(typeof(u)), 18), "mean = ", sum(u) / length(u))
end
```

```text
LocalHaloArray    mean = 0.038349519684809985
ThreadedHaloArray mean = 0.038349519684810080
HaloArray         mean = 0.038349519684809985
```

Same answer on every backend (the last digits differ only by floating-point
reduction order). Run it with `julia -t 4` to use the threads; swap
`MPI.COMM_SELF` for `MPI.COMM_WORLD` and launch with `mpiexec -n 4` to
distribute across ranks — the loop body does not change.

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
