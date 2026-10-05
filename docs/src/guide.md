# Guide

## Main types

- [`HaloArray`](@ref): MPI-backed array with local interior cells and halo cells.
- [`LocalHaloArray`](@ref): no-MPI halo array for local problems and boundary-condition-only workflows.
- [`ThreadedHaloArray`](@ref): thread-local tiled halo array for shared-memory workflows.
- [`MultiHaloArray`](@ref): named collection of MPI-backed halo fields.
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

## Indexing

A halo array reports its **global** shape but stores only this rank's/tile's
piece, so it deliberately supports a *narrower* indexing surface than a plain
`AbstractArray` — everything it refuses, it refuses loudly, with the fast
alternative named in the error.

**What works:**

| form | behaviour |
|:-----|:----------|
| `u[i, j, …]` (all `ndims` indices, **global** coordinates) | reads/writes the cell — on MPI only for cells this rank owns (warns; diagnostics only, not for hot loops) |
| `u[i, j, …, 1]` (trailing `1`s past `ndims`) | allowed, per the `AbstractArray` contract (generic LinearAlgebra code such as `Diagonal`'s `ldiv!` relies on it) |

**What is refused, and what to use instead:**

| form | why | instead |
|:-----|:----|:--------|
| `u[3]` — linear indexing of an N-d array | would route generic code through a slow scalar path (and is ill-defined across ranks) | `interior_view(u)[3]`, or loop `eachindex(u)` / `CartesianIndices(interior_view(u))` |
| `u[:]`, `u[:, 1]`, `u[1:2, :]` — slices | would materialize cross-rank/tile data cell by cell | slice the view: `interior_view(u)[:, 1]` (this rank's cells), or [`gather_haloarray`](@ref)`(u)` for the true global array (on MPI: root only) |

The idiomatic patterns:

```julia
interior_view(u) .= data              # bulk initialisation of this rank's cells
fill_from_global_indices!(f, u)       # initial condition from GLOBAL indices, any backend
interior_view(u)[2, :]                # arbitrary slicing — it is a normal array
parent(u)[I...]                       # storage coordinates (ghosts included), for stencils
A = gather_haloarray(u)               # the assembled global array (on the MPI root)
```

Everything performance-relevant — broadcast, reductions, `dot`/`norm`, the
BLAS-1 updates — is specialized and never touches scalar indexing, so these
restrictions only surface in hand-written loops over the array itself.

The same global-vs-local rule governs **plain-array operands in an in-place
broadcast** (`u .= u .+ g`). On the shared-memory backends a plain operand is
indexed by *global* interior coordinates — a `ThreadedHaloArray` hands each
tile a view of its own global window, with the usual size-1 broadcast
expansion per dimension. On a distributed `HaloArray` a plain operand must be
**owned-size** (rank-local, `interior_view`-shaped): a global-shaped operand
would require every rank to hold the whole array or an implicit scatter, so
distribute global data explicitly (e.g. [`fill_from_global_indices!`](@ref))
and broadcast with local operands.

## Halo exchange

### Reading and writing all fields at one cell

Use `siteview(u, I[, tile])` to access all components as a lazy, writable array
shaped like the fields, without copying values or building a container of
backing arrays. These CPU
operations use **local padded-storage indices**, including ghost cells, and
perform no communication:

```julia
u = ArrayOfHaloArray(LocalHaloArray, Float64, (3,), (32,32), 1;
                     boundary_condition=:periodic)
U = zeros(3)
I = first(interior_cells(CellRanges(u)))
q = siteview(u, I)
copyto!(U, q)                # fields -> vector
copyto!(q, U)                # overwrite fields
q .+= 0.5 .* U               # accumulate into initialized fields
snapshot = copy(q)           # independent Vector
```

An `ArrayOfHaloArray` with a multidimensional field shape gives a site array of
that shape. For fields of size `(2, 2, nx, ny, nz)`, each site is a lazy 2×2
matrix:

```julia
U = ArrayOfHaloArray(LocalHaloArray, Float64, (2, 2), (16, 16, 16), 1;
                     boundary_condition=:periodic)
I = first(interior_cells(CellRanges(U)))
m = siteview(U, I)          # 2×2: m[a, b] is field (a, b) at cell I
m[1, 2] = 1.0               # writes field (1, 2) directly
m .+= 0.5 .* M              # broadcast operands must be 2×2 (or broadcastable)
copyto!(buf, m)             # a flat buffer of length 4, column-major
vec(m) .= buf               # flat view for broadcasting with flat buffers
```

Single halo arrays are also supported, yielding a one-component vector:

```julia
u = LocalHaloArray(Float64, (32,32), 1; boundary_condition=:periodic)
I = first(interior_cells(CellRanges(u)))
q = siteview(u, I)
q[1] = 2.0                  # writes the cell directly
q .*= 3                     # the cell now contains 6.0
```

A vector- or matrix-valued cell is one component; its contents are not flattened.
Collections must be flat. `size(q) == field_shape(u)`; linear indexing and
`copyto!` follow column-major field order, and named collections use declaration
order. Threaded arrays and collections require a final tile ID argument. The
number of dimensions is part of the view's type; the extents of an
`ArrayOfHaloArray` site are runtime values. The view is not contiguous memory.

Writes take effect immediately. Take a snapshot if a calculation needs the
original state throughout an update. Site views have no internal snapshot buffer.
Operations requiring Julia to make an automatic alias-protection copy of a site
view throw `ArgumentError`; explicitly copy the source instead:

```julia
q .= copy(view(q, length(q):-1:1))
```

Alias detection is conservative, so separate views sharing field storage may
require a copy even at distinct sites. Direct self-broadcast (`q .*= 2`) and
operations with independent buffers remain supported.

Synchronize halos before reading ghost
values and refresh them before subsequent stencil reads. Do not resize or
replace fields while a view is in use. Parallel writes must access disjoint
storage or be synchronized.

Construction performs no validation: callers must supply a single array or flat fields, a valid
storage index of the correct dimension, and a valid tile (explicit for threaded
storage). Ordinary scalar indexing still checks bounds; `@inbounds` can skip it.

Copying and broadcasting use Julia's standard implementations and validation.
`copyto!` checks destination capacity (a larger destination is allowed), while
broadcasting checks compatible shapes and supports singleton expansion. There
are no additional component-length or shape checks in the site-view code.
Alias detection remains enabled.

Scalar reads convert to the state's element type (promoted for collections), so
`q[k] isa eltype(q)` holds even for mixed field types; read a field directly for its
native type. `copy(q)` and `similar(q)` allocate ordinary arrays of the same element
type. Writes convert to the destination field's type.

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

All mutating drivers (`halo_exchange!`, `synchronize_halo!`,
`boundary_condition!`, `fill!`, `copyto!`, the BLAS-1 updates, …) return their
array on **every** backend, so backend-agnostic chaining like
`u = synchronize_halo!(u)` always works.

For `ThreadedHaloArray`, the default `halo_exchange!`, `boundary_condition!`, and
`synchronize_halo!` use a serial tile loop because this is allocation-free and
fastest for small halo surfaces. Explicit threaded variants
(`halo_exchange!(u; threads=true)`, `boundary_condition!(u; threads=true)`,
`synchronize_halo!(u; threads=true)`) are available; reach for them only after
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
function HaloArrays.apply_coupled_bc!(bc::MyBC, state, s::Side{S}, d::Dim{D}, tile) where {S,D}
    for field in eachfield(state)                 # iterate the collection's fields
        edge  = edge_view(field, s, d, tile)      # interior cells at the boundary (read)
        ghost = ghost_view(field, s, d, tile)     # ghost cells (write)
        # ... transform across fields, then write `ghost` ...
    end
end

synchronize_halo!(state)          # periodic/reflecting edges, per field
apply_coupled_bc!(MyBC(), state)  # fills the NoBoundaryCondition physical edges
```

One method covers every backend: the driver passes `tile = nothing` for
`LocalHaloArray`/MPI fields (whole-array views) and the boundary tile id for
`ThreadedHaloArray` fields — you just forward it to the view helpers. The
two-argument `apply_coupled_bc!(bc, state)` visits every face that is both a
physical boundary ([`is_physical_boundary`](@ref)) and configured
`NoBoundaryCondition`.
See `examples/finite_volume/acoustics_characteristic_1d.jl`.

## Local and threaded arrays

`LocalHaloArray` is the simplest option on a single process:

```julia
u = LocalHaloArray(Float64, (64, 64, 64), 2; boundary_condition=:repeating)
interior_view(u) .= 1.0
synchronize_halo!(u)
```

For position-dependent initial conditions, use
[`fill_from_global_indices!`](@ref) — the callback receives the **global** index
tuple, so the same `f` produces one consistent field regardless of how the
domain is decomposed across ranks or tiles:

```julia
fill_from_global_indices!(u) do I
    exp(-((I[1] - 32)^2 + (I[2] - 32)^2) / 50)
end
```

`ThreadedHaloArray` splits the domain into local tiles and exchanges halos across
tiles using threads:

```julia
u = ThreadedHaloArray(Float64, (32, 32, 32), 2; dims=(2, 2, 2), boundary_condition=:periodic)
synchronize_halo!(u)
```

!!! tip "Choosing the tile layout `dims`"
    Julia arrays are column-major: dimension 1 is the contiguous, SIMD/prefetch
    direction. Prefer decompositions that keep it intact — `dims[1] = 1`, balance
    the remaining dimensions (e.g. 8 threads in 3-D → `dims = (1, 2, 4)`). The
    default (`nthreads()` tiles along the **last** dimension) already follows
    this rule; splitting dimension 1 chops the contiguous runs and turns the
    inter-tile edge copies into strided gathers.

### When multi-field containers are useful

`MultiHaloArray` (named) and `ArrayOfHaloArray`
(indexed) help when a solver evolves several fields on one grid (`rho`, `u`, `v`,
`p`, …). Use one when all fields share geometry and halo width and you want a
single `synchronize_halo!(state)` for the whole state. Keep independent arrays
when fields need different halo widths, layouts, or topologies.

## Reductions

Broadcast and reductions operate on the **interior** cells (halos excluded), and
the whole-array reductions are **global** — an MPI `HaloArray` combines every
rank's contribution internally, so the same code returns the same scalar on
every backend:

```julia
sum(u)             # global sum over all interior cells
maximum(u)         # global maximum
mapreduce(abs2, +, u)
using LinearAlgebra
dot(u, v)          # global inner product
norm(u)            # global 2-norm
```

`sum`/`prod`/`maximum`/`minimum` and `mapreduce` all work; `dot`/`norm` and the
in-place BLAS‑1 updates make a halo array a drop-in Krylov/`OrdinaryDiffEq`
vector. `mapfoldl`/`mapfoldr` also work, but their strict ordering makes them
incompatible with the `dims=` keyword below (use the commutative `mapreduce`
forms there).

### Reducing along dimensions

Passing `dims=` collapses only some axes and keeps a distributed array. Every
backend returns a reduced array of **the same backend** with the reduced
dimensions **dropped** (kept dimensions keep their halo width and boundary
conditions):

```julia
# the same call, returning a reduced array of u's own backend:
#   LocalHaloArray    → LocalHaloArray
#   ThreadedHaloArray → ThreadedHaloArray (tiled by the kept dimensions)
#   HaloArray (MPI)   → MaybeHaloArray    (see below)
r = sum(u; dims=2)
r = mapreduce(abs2, +, u; dims=2)   # explicit form, same result
```

For an MPI `HaloArray` the collapsed result lives only on the **coordinate‑0
slice** of the reduced dimensions and is returned as a [`MaybeHaloArray`](@ref):
`is_active(r)` is `true` on the ranks that hold it (and always `true` for serial
backends), `interior_view(r)` reads it there, and [`free!`](@ref)`(r)` releases
the sub-communicator it owns (optional — otherwise reclaimed at `MPI.Finalize`;
call it to keep communicator use bounded when reducing in a loop). These three
behave uniformly on every return kind, so reduction-consuming code needs no
backend branches:

```julia
r = sum(u; dims=2)
if is_active(r)
    save(interior_view(r))
end
free!(r)
```

One-shot reductions promote the element type like Base (so `sum` of a `Bool`
array counts in `Int`).

### Reusing a plan in a loop

Each `sum(u; dims=…)` builds and releases MPI sub-communicators. For a reduction
that runs every step, build a [`DimReductionPlan`](@ref) **once** and
[`reduce!`](@ref) into it — one `MPI.Reduce` per call, no communicator churn, and
the same call compiles on every backend:

```julia
plan = DimReductionPlan(u, 2)          # once, outside the loop
for step in 1:nsteps
    step!(u)
    profile = reduce!(plan, identity, +, u)   # overwrites the plan's output
    is_active(profile) && save(profile)
end
free!(plan)
```

A plan fixes its output element type at construction (pass `output_eltype` to
override); a promoting reduction against it errors with a pointer to the
one-shot forms. On `LocalHaloArray`/`ThreadedHaloArray` the plan build and the
whole reduction are type-stable and allocation-free (even for a runtime
`dims::Int`).

### Reducing collections

On a [`MultiHaloArray`](@ref)/[`ArrayOfHaloArray`](@ref), `dims=` uses
**collection coordinates**: the field axes come first (`1:F`), then the shared
spatial axes (`F+1:D`) — the same order as `size(c)`. Field axes reduce
**locally** (an elementwise fold across the fields — no communication), so
reducing every field axis returns one bare `HaloArray` (a `MultiHaloArray` drops
the field names); spatial axes reduce per field and rebuild the same collection
kind. The `MaybeHaloArray` wrapper, when a spatial axis was reduced on MPI, is
always outermost.

```julia
state = MultiHaloArray((; rho, mom))    # 2-D fields → size (2, nx, ny)
sum(state; dims=1)                       # sum over fields → one HaloArray (nx, ny)
sum(state; dims=3)                       # reduce spatial-y → collection (nx,)
sum(state; dims=(1, 3))                  # both → one HaloArray (nx,)
```

`mapreduce(f, op, c; dims=dims)` and `DimReductionPlan(c, dims)` accept
collections the same way.

### Gather and output

[`gather_haloarray`](@ref)`(u)` assembles the global interior into an `Array` on
any backend (a copy on `LocalHaloArray`, stitched tiles on `ThreadedHaloArray`,
the root rank on MPI; collections give the field axes first, or a `NamedTuple`
per field). HDF5 output (weak dependency, `using HDF5`) is one function,
[`append_haloarray!`](@ref)`(file, name, u)`: it appends the interior as the next
step of a time-series dataset (time on the leading axis) into a file you opened
with HDF5.jl — collectively for a distributed array, each rank writing its own
block — and returns the dataset so you can attach attributes. A snapshot is a
single append. `append_haloarray!(filename, name, u)` opens the file for you on
the array's own communicator, which suits a `dims=` reduction result
(`MaybeHaloArray`). For a gathered global array use plain HDF5.jl:
`A = gather_haloarray(u); is_root(u) && h5write("snap.h5", "rho", A)`.
See [Arrays, layout & reductions](@ref) for the full API.

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
construction with the `thread_backend` keyword (default `ThreadsBackend()`):

```julia
u = ThreadedHaloArray(Float64, (32, 32), 1; dims=(2, 2),
                      boundary_condition=:periodic, thread_backend=SerialBackend())
thread_backend(u)   # SerialBackend()
```

| Backend | Notes |
|---|---|
| [`ThreadsBackend`](@ref)     | Default. Base tasks (`Threads.@spawn`), no extra package; composes and nests. |
| [`PolyesterBackend`](@ref)   | Lowest overhead (`@batch`, persistent workers). Requires `using Polyester`. Does not nest. |
| [`OhMyThreadsBackend`](@ref) | OhMyThreads tasks; honours the `scheduler` keyword (dynamic load balancing). Requires `using OhMyThreads`. |
| [`SerialBackend`](@ref)      | Per-tile work on the calling thread — debugging races / deterministic runs. |

The backend is part of the array's concrete type (compile-time dispatch) and
propagates through `similar`, broadcast, and reductions. Add your own by defining
[`tile_foreach`](@ref) and [`tile_mapreduce`](@ref) for a new `<:ThreadBackend`.

### Choosing a backend

All three parallel backends do the same per-tile work; they differ in the fixed
cost paid per threaded call, which matters when each call does little work
(`fill!`, broadcasts, boundary fills, the halo sync, Krylov dot products):

- `ThreadsBackend` splits the tiles into one chunk per thread, spawns all but
  the first as Base tasks and works on the first itself: about 1 µs per call
  and ~6 allocations per spawned task. It nests inside other threaded code.
- `PolyesterBackend` hands the work to persistent worker threads through
  preallocated per-thread buffers: a few hundred nanoseconds per call and
  (near) zero allocations, but a `@batch` region must not run inside another
  threaded region.
- `OhMyThreadsBackend` costs a few microseconds and ~30 allocations per call;
  choose it when you want OhMyThreads' schedulers, e.g. dynamic load balancing
  for your own uneven per-tile work through `tile_foreach(...; scheduler=...)`.

For coarse-grained work — a full stencil sweep, an RHS evaluation — the spawn
cost is amortised and the backends converge. `synchronize_halo!(u; threads=true)`
copies only thin edge slabs, so whether it beats the serial default
(`threads=false`) depends on the backend overhead: measure on your problem size.

`benchmark/thread_backends.jl` measures the backends on exactly these operations.
As a rule, no threaded backend helps a purely memory-bandwidth-bound kernel scale
past the machine's memory bandwidth — the backend choice only changes the *fixed*
per-call overhead, not the bandwidth ceiling.

## Face loops

[`FaceRanges`](@ref)`(u)` gives the index ranges for finite-volume face updates,
in parent-storage indices (for kernels working on `parent(u)`/`parent(du)`). The
high-level [`accumulate_flux_divergence!`](@ref) does the whole flux-divergence
update for one direction:

```julia
fr = FaceRanges(u)
accumulate_flux_divergence!(parent(du), parent(u), fr, dim, inv(dx), numerical_flux)
```

Or write the loop explicitly with [`interior_faces`](@ref)`(ranges, dim)` — every
face touching the interior along `dim` — pairing each lower cell `IL` with
`IL + unit_vector(ranges, dim)` and scattering the flux onto both cells (the two
boundary faces also write a ghost cell, which is in-bounds and harmless). For a
race-free *parallel* scatter, loop one checkerboard color at a time with
`interior_faces(ranges, dim, color)`.

For collections the ranges are spatial only — select a field first.

## Cell loops

[`CellRanges`](@ref)`(u)` gives the interior-cell range. For ordinary out-of-place
stencils use [`interior_cells`](@ref); for nearest-neighbor in-place red-black
updates use [`interior_cells`](@ref)`(ranges, color)` (strided
`CartesianIndices`, so the inner loop has no parity branch). A cell's color is
its **global** parity `mod(sum(global_index), 2)`, so the checkerboard stays
continuous across tile/rank seams; on a [`ThreadedHaloArray`](@ref) build the
ranges per tile — `CellRanges(u, tile_id)` — inside the tile loop.

## Kernel regions

The range APIs are also available as compact launch metadata for GPU /
KernelAbstractions kernels: [`FaceWindow`](@ref) /
[`FaceCheckerboard`](@ref) for faces, and [`CellWindow`](@ref) /
[`CellCheckerboard`](@ref) for cells (with [`cell_index`](@ref) and
[`is_cell_index_inbounds`](@ref) to map a launch index to a storage cell). See
`examples/tutorials/gpu.jl`.
