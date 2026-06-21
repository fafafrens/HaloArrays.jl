# Guide

## Main types

- [`HaloArray`](@ref): MPI-backed array with local interior cells and halo cells.
- [`LocalHaloArray`](@ref): no-MPI halo array for local problems and boundary-condition-only workflows.
- [`ThreadedHaloArray`](@ref): thread-local tiled halo array for shared-memory workflows.
- `ThreadedMultiHaloArray`: named collection of threaded halo fields.
- [`MultiHaloArray`](@ref): named collection of MPI-backed halo fields.
- `LocalMultiHaloArray`: named collection of local halo fields.
- [`ArrayOfHaloArray`](@ref): index-addressed collection of fields.
- [`CartesianTopology`](@ref): MPI Cartesian topology helper.

`size(u)` and `axes(u)` describe the global logical domain for every halo
container. Local interior data is accessed through [`interior_size`](@ref),
[`interior_axes`](@ref), and [`interior_view`](@ref).

Scalar indexing on `HaloArray` uses global indices, but it is local-only: it
works only for global cells in the current MPI rank's interior region and throws otherwise.
It is intended for diagnostics and setup, not stencil kernels or hot loops. Use
`interior_view`, `parent`, `interior_to_global_index`, and `global_to_storage_index`
when you need explicit local/global behavior.

## Typical workflow

Most stencil updates follow this sequence:

1. Update only the interior region ([`interior_view`](@ref)`(u)`).
2. Call [`synchronize_halo!`](@ref)`(u)` before any stencil read that touches halo cells.
3. Repeat for each time step.

This keeps interior ownership explicit and halo validity predictable.

## Global and local semantics

The package separates logical array shape from interior storage:

- `size(u)` / `axes(u)`: global logical array shape.
- [`interior_size`](@ref)`(u)` / [`interior_axes`](@ref)`(u)`: cells local to this rank or local backend.
- [`interior_view`](@ref)`(u)`: writable interior cells, excluding halos.
- `parent(u)`: raw storage including halos.
- [`storage_size`](@ref)`(u)`: raw storage size including halos.

For MPI-backed `HaloArray`, scalar `u[I...]` does not communicate. If `I` is not
in the current rank's interior region, it errors. This keeps expensive communication out of
generic indexing and makes performance-critical code use local views explicitly.

## Halo exchange

The public exchange API is intentionally small:

```julia
halo_exchange!(u)          # blocking MPI exchange
start_halo_exchange!(u)    # begin async exchange
finish_halo_exchange!(u)   # finish async exchange
synchronize_halo!(u)       # make halos valid for stencil use
```

For `HaloArray`, [`synchronize_halo!`](@ref) performs the exchange and then
applies physical boundary conditions where the topology has `MPI.PROC_NULL`
neighbors. For `LocalHaloArray`, it only applies boundary conditions.

Use the split async API ([`start_halo_exchange!`](@ref) + [`finish_halo_exchange!`](@ref))
to overlap communication with independent computation.

For `ThreadedHaloArray`, the default `halo_exchange!`, `boundary_condition!`, and
`synchronize_halo!` use a serial tile loop because this is allocation-free and
fastest for small halo surfaces. Explicit threaded variants
(`halo_exchange_threads!`, `boundary_condition_threads!`,
`synchronize_halo_threads!`) are available; reach for them only after
benchmarking, when the halo surface is large.

## Boundary conditions

Built-in boundary conditions are [`Reflecting`](@ref), [`Antireflecting`](@ref),
[`Repeating`](@ref), [`Periodic`](@ref), and [`NoBoundaryCondition`](@ref); the
symbols `:reflecting`, `:antireflecting`, `:repeating`, `:periodic`,
`:noboundary` are also accepted.

```julia
HaloArray(Float64, (64, 64), 1, topology; boundary_condition=:periodic)

HaloArray(Float64, (64, 64), 1, topology;
    boundary_condition=((Reflecting(), Repeating()), (:periodic, :periodic)))
```

Custom boundary conditions can be passed as a subtype or instance of
`AbstractBoundaryCondition`.

### Custom per-field conditions with `FunctionBC`

For a one-off rule you don't want to make a type for, wrap a function in
[`FunctionBC`](@ref). It runs inside `synchronize_halo!` like a built-in, on
physical edges only, and works on every backend (single, MPI, threaded). Your
function is called per `(side, dim)` face as `f(ghost, edge, side, dim, hw, origin)`,
where `ghost` is the slab to write and `edge` is the adjacent interior slab to read
(same shape). Because the two straddle the wall, the standard conditions are short,
side-independent one-liners — and they're geometry-agnostic, so you pass any grid
spacing `Δ` yourself:

```julia
# Dirichlet — fix the wall value u₀:           (ghost + edge)/2 = u₀
dirichlet(u₀)     = FunctionBC((g, e, s, d, hw, o) -> (g .= 2 .* u₀ .- e))
# Neumann — fix the outward normal flux q:     (ghost − edge)/Δ = q
neumann(q, Δ)     = FunctionBC((g, e, s, d, hw, o) -> (g .= e .+ Δ .* q))
# Robin — α·u + β·∂u/∂n = γ at the wall:
robin(α, β, γ, Δ) = FunctionBC((g, e, s, d, hw, o) -> (g .= (γ .- (α/2 - β/Δ) .* e) ./ (α/2 + β/Δ)))
```

These compose the way you'd expect: `robin(1, 0, u₀, Δ)` is `dirichlet(u₀)`,
`robin(0, 1, q, Δ)` is `neumann(q, Δ)`, and `dirichlet(0)` coincides with the
built-in [`Antireflecting`](@ref) (zero-flux Neumann likewise coincides with
[`Repeating`](@ref)). Pass them per side like any BC, e.g.
`boundary_condition = ((dirichlet(300.0), neumann(0.0, dx)), (:periodic, :periodic))`.

`origin` is the **global** `CartesianIndex` of `ghost[1]` (the package computes the
MPI-rank / tile offset for you), so a *position-dependent* condition is a broadcast
that stays correct under decomposition — and GPU-safe, since each lane derives its
own global index:

```julia
inflow = FunctionBC() do g, e, s, d, hw, o
    g .= profile.(Tuple.((o - oneunit(o)) .+ CartesianIndices(g)))   # value varies along the face
end
```

Three kinds, one mechanism: built-in singletons, `FunctionBC` (custom **per-field**),
and coupled (**cross-field**, below).

### Coupled boundary conditions

Some schemes — characteristic reconstruction, for instance — need *all* fields'
interior edges together to fill the ghosts (the ghost state of each field depends
on the others). Mark those `(dim, side)` with [`NoBoundaryCondition`](@ref) so
`synchronize_halo!` skips them, then fill them from the whole state with a coupled
boundary condition: subtype [`AbstractCoupledBoundaryCondition`](@ref) and
implement [`apply_coupled_bc!`](@ref).

```julia
struct MyBC <: AbstractCoupledBoundaryCondition end
function HaloArrays.apply_coupled_bc!(bc::MyBC, state, s::Side{S}, d::Dim{D}) where {S,D}
    for field in eachfield(state)              # iterate the collection's fields
        edge  = get_send_view(s, d, field)     # interior cells at the boundary (read)
        ghost = get_recv_view(s, d, field)     # ghost cells (write)
        # ... transform across fields, then write `ghost` ...
    end
end

synchronize_halo!(state)          # periodic/reflecting edges, per field
apply_coupled_bc!(MyBC(), state)  # fills the NoBoundaryCondition physical edges
```

The two-argument `apply_coupled_bc!(bc, state)` visits every face that is both a
physical boundary ([`is_physical_boundary`](@ref)) and configured
`NoBoundaryCondition`. Works on `MultiHaloArray` and `ArrayOfHaloArray` with
`LocalHaloArray`/MPI fields (threaded fields use a per-tile five-argument method).
See `examples/finite_volume/acoustics_characteristic_1d.jl`.

## Local and threaded arrays

`LocalHaloArray` is the simplest option on a single process:

```julia
u = LocalHaloArray(Float64, (64, 64, 64), 2; boundary_condition=:repeating)
interior_view(u) .= 1.0
synchronize_halo!(u)
```

`ThreadedHaloArray` splits the domain into local tiles and exchanges halos across
tiles using threads:

```julia
u = ThreadedHaloArray(Float64, (32, 32, 32), 2; dims=(2, 2, 2), boundary_condition=:periodic)
synchronize_halo!(u)
```

### When multi-field containers are useful

`MultiHaloArray` / `ThreadedMultiHaloArray` (named) and `ArrayOfHaloArray`
(indexed) help when a solver evolves several fields on one grid (`rho`, `u`, `v`,
`p`, …). Use one when all fields share geometry and halo width and you want a
single `synchronize_halo!(state)` for the whole state. Keep independent arrays
when fields need different halo widths, layouts, or topologies.

## Backend traits

Use [`halo_backend`](@ref)`(u)` when an algorithm needs separate implementations
for MPI, local, and threaded storage while still accepting collection wrappers.
It returns `MPIHaloBackend()`, `LocalHaloBackend()`, or `ThreadedHaloBackend()`.

```julia
update!(du, u, p) = update!(halo_backend(u), du, u, p)
update!(::Union{MPIHaloBackend,LocalHaloBackend}, du, u, p) = serial_update!(du, u, p)
update!(::ThreadedHaloBackend, du, u, p) = threaded_update!(du, u, p)
```

## Thread backends

`halo_backend` describes *where* data lives; a [`ThreadBackend`](@ref) describes
*how* a `ThreadedHaloArray`'s per-tile work is dispatched. Choose it at
construction with the `thread_backend` keyword (default `OhMyThreadsBackend()`):

```julia
u = ThreadedHaloArray(Float64, (32, 32), 1; dims=(2, 2),
                      boundary_condition=:periodic, thread_backend=SerialBackend())
thread_backend(u)   # SerialBackend()
```

| Backend | Notes |
|---|---|
| [`OhMyThreadsBackend`](@ref) | Default. Task-based, supports schedulers, composes/nests. |
| [`SerialBackend`](@ref)      | Per-tile work on the calling thread — debugging races / deterministic runs. |
| [`PolyesterBackend`](@ref)   | Low-overhead `@batch`. Requires `using Polyester` (the `HaloArraysPolyesterExt` extension). |

The backend is part of the array's concrete type (compile-time dispatch) and
propagates through `similar`, broadcast, and reductions. Add your own by defining
[`tile_foreach`](@ref) and [`tile_mapreduce`](@ref) for a new `<:ThreadBackend`.

## Face loops

[`FaceRanges`](@ref)`(u)` gives the index ranges for finite-volume face updates,
in parent-storage indices (for kernels working on `parent(u)`/`parent(du)`). The
high-level [`accumulate_flux_divergence!`](@ref) does the whole left/internal/right
update for one direction:

```julia
fr = FaceRanges(u)
accumulate_flux_divergence!(parent(du), parent(u), fr, dim, inv(dx), numerical_flux)
```

Or write the loops explicitly with [`get_left_face`](@ref),
[`get_internal_face`](@ref)`(ranges, dim)` (direction-aware), and
[`get_right_face`](@ref), pairing each `IL` with `IL + get_unit_vector(ranges, dim)`.
For in-place updates that write both adjacent cells, use the colored variants
(`get_internal_face(ranges, dim, color)`) to avoid same-color conflicts.

For collections the ranges are spatial only — select a field first.

## Cell loops

[`CellRanges`](@ref)`(u)` gives the interior-cell range. For ordinary out-of-place
stencils use [`get_interior_cells`](@ref); for nearest-neighbor in-place red-black
updates use [`get_interior_cells`](@ref)`(ranges, color)` (strided
`CartesianIndices`, so the inner loop has no parity branch). Cell colors use
`mod(sum(Tuple(I)), 2)`.

## Kernel regions

The range APIs are also available as compact launch metadata for GPU /
KernelAbstractions kernels: [`FaceWindow`](@ref) /
[`FaceCheckerboard`](@ref) for faces, and [`CellWindow`](@ref) /
[`CellCheckerboard`](@ref) for cells (with [`cell_index`](@ref) and
[`is_cell_index_inbounds`](@ref) to map a launch index to a storage cell). See
`examples/tutorials/gpu.jl`.
