# Good habits

The package is built so that code written for one backend runs on the others.
That holds as long as the code respects a few conventions, most of which follow
from the [index spaces](indexing.md). This page states them as rules, each with
the reason and the idiomatic form.

## Own the interior, let the package fill the ghosts

Write only to the owned cells, through [`interior_view`](@ref) or
[`fill_from_global_indices!`](@ref), and call [`synchronize_halo!`](@ref)
before any read that touches a ghost cell. The ghosts are stale until then:
whatever the last synchronisation left there.

```julia
interior_view(u) .= initial
synchronize_halo!(u)              # ghosts valid from here
for step in 1:nsteps
    stencil!(du, u)               # reads ghosts
    interior_view(u) .+= dt .* interior_view(du)
    synchronize_halo!(u)          # again, before the next read
end
```

`synchronize_halo!` is the **only** place ghosts are filled, on every backend:
the exchange between ranks or tiles plus the boundary conditions at the
physical walls. Do not write ghost cells by hand. The one exception is a
coupled boundary condition, which fills the faces you marked
[`NoBoundaryCondition`](@ref) through [`apply_coupled_bc!`](@ref) after the
synchronisation.

Use the split form, [`start_halo_exchange!`](@ref) and
[`finish_halo_exchange!`](@ref), only to overlap communication with work that
does not read the ghosts. The pair is the **exchange alone**: it fills the ghosts
that face a neighbouring rank or tile and does nothing at a physical wall (on a
`LocalHaloArray` both calls are no-ops). The physical-wall ghosts stay stale
until you call [`boundary_condition!`](@ref) after the finish, which is the
second half of what `synchronize_halo!` does:

```julia
start_halo_exchange!(u)
interior_only_work!(u)        # reads no ghost cell
finish_halo_exchange!(u)
boundary_condition!(u)        # now every ghost is valid
stencil!(du, u)
```

## Initialise from global indices

A position-dependent initial condition is a function of the **global** index.
Hand it to [`fill_from_global_indices!`](@ref) and let each block fill the
cells it owns:

```julia
fill_from_global_indices!(u) do I
    exp(-((I[1] - 32)^2 + (I[2] - 32)^2) / 50)
end
```

The same `f` gives the same field on one process, on tiles, or on any number
of ranks. A loop over `u[I]` does the same on the serial kinds, slowly, and on
MPI fails for every cell another rank owns. For a collection return one value
per field.

The same applies inside a boundary rule: a condition that varies along the
wall reads [`global_index`](@ref)`(u)`, never the storage index.

## Write hot loops in storage coordinates

A stencil reads neighbours, and a neighbour is an offset in the padded block.
Loop over the block through [`interior_range`](@ref) or [`CellRanges`](@ref)
and index `parent(u)` or [`tile_parent`](@ref)`(u, t)`:

```julia
s, d = parent(u), parent(du)
e1, e2 = unit_vector(u, 1), unit_vector(u, 2)
@inbounds for I in CartesianIndices(interior_range(u))
    d[I] = (s[I + e1] + s[I - e1] + s[I + e2] + s[I - e2] - 4s[I]) * inv_h2
end
```

Never put `u[i, j]` in such a loop. Scalar indexing takes global coordinates,
checks ownership on every access and, on MPI, warns that it is local-only. It
exists for setup and for printing a value. Linear indexing and slicing of the
halo array are refused outright; slice `interior_view(u)` instead.

## Write one loop for every backend

A single-block array is a one-tile decomposition, so a loop over tiles runs on
every kind. Take the tile from [`tile_parent`](@ref) and the ranges from
`CellRanges(u, t)`, and the same function serves a `LocalHaloArray`, an MPI
`HaloArray` and a `ThreadedHaloArray`:

```julia
function laplacian!(du, u, inv_h2)
    for t in 1:tile_count(u)
        s, d = tile_parent(u, t), tile_parent(du, t)
        e1, e2 = unit_vector(u, 1), unit_vector(u, 2)
        @inbounds for I in interior_cells(CellRanges(u, t))
            d[I] = (s[I + e1] + s[I - e1] + s[I + e2] + s[I - e2] - 4s[I]) * inv_h2
        end
    end
    return du
end
```

To run the tiles in parallel replace the outer loop with
[`tile_foreach`](@ref)`(u) do t … end`; the array's [`thread_backend`](@ref)
decides how. A per-tile kernel must touch only its own tile, since tiles run
concurrently.

When you do need a backend branch, dispatch on [`halo_backend`](@ref)`(u)`
rather than testing the type, so collections and wrappers fall through.

## Prefer the bulk operations

Broadcast, `fill!`, `copyto!`, the BLAS-1 updates (`axpy!`, `axpby!`,
`rmul!`, [`swap!`](@ref)) and the reductions (`sum`, `maximum`, `mapreduce`,
`dot`, `norm`, [`dots!`](@ref)) are written once per backend: interior-only,
threaded over tiles, one kernel on GPU storage, and global on MPI with a
single reduction. They are the fast path and the portable path:

```julia
u .= u .+ dt .* du            # interior only, every backend
r = norm(u)                   # the same number on every rank
```

A plain array in an in-place broadcast is indexed by **global** interior
coordinates on the shared-memory backends, so it is domain-sized even on tiles,
and must be **owned-size**, shaped like `interior_view(u)`, on MPI. Distribute
global data with `fill_from_global_indices!` first.

## Treat collective calls as collective

On MPI, `synchronize_halo!`, every reduction, `dot`, `norm`,
[`gather_haloarray`](@ref), [`DimReductionPlan`](@ref) and
[`append_haloarray!`](@ref) are collective: every rank must call them, in the
same order. Never guard them with a rank test. Test the **result** instead:

```julia
A = gather_haloarray(u)               # every rank calls it
is_root(u) && h5write("snap.h5", "rho", A)

r = sum(u; dims=2)                    # every rank calls it
P = gather_haloarray(r)               # the assembled profile, or nothing on a rank that does not hold it
P === nothing || save(P)
free!(r)
```

[`is_root`](@ref) and [`is_active`](@ref) are `true` on the serial backends,
and `gather_haloarray` returns the array on every serial kind (stitching the
tiles of a `ThreadedHaloArray`), so the pattern needs no backend branch. Do not
reach into a reduced result with `interior_view(r)`: on a threaded result the
interior is per tile, `interior_view(r, t)`. Diagnostics that read `u[I]` on
one rank are fine; they do not communicate.

## Allocate once

Build arrays with `similar`, `zero` or `copy` of an existing one so the halo
width, boundary conditions, topology and storage carry over. Preallocate the
work arrays of a time loop outside it. A reduction with `dims=` that runs
every step belongs in a [`DimReductionPlan`](@ref) built once and driven with
[`reduce!`](@ref); call [`free!`](@ref) on plans and on the `MaybeHaloArray`
results of one-shot reductions to keep communicator use bounded.

Every mutating driver returns its array on every backend, so
`u = synchronize_halo!(u)` and `du = laplacian!(du, u, c)` chain safely.

## Keep the storage consistent

Where an array's memory lives is the `storage=` keyword of its constructor,
`storage=CUDA.zeros` or `Metal.zeros` for a device array. The exchange buffers
of an MPI `HaloArray` follow it, the fields of a collection inherit it, and
`similar` preserves it. Moving an existing array with `adapt(CuArray, u)`
moves its boundary conditions too, including the arrays a custom rule
captures. Do not mix host and device arrays in one operation: a broadcast
between them falls back to scalar indexing.

## On a GPU

- No scalar indexing anywhere in the time loop; it is one transfer per element.
- Write stencils as kernels over a [`CellWindow`](@ref) or, for nearest
  neighbours, a [`LinearCellWindow`](@ref) with [`cell_strides`](@ref); keep
  the index type `Int32` unless the storage exceeds `2^31` elements.
- Keep the ghost refresh on the device. With a GPU array package loaded,
  `synchronize_halo!` fills all rule faces in one launch and the MPI exchange
  packs its buffers with kernels. One MPI rank per GPU, with a GPU-aware MPI,
  is the supported multi-GPU layout.
- Bring data to the host once, with `gather_haloarray` or `Array(interior_view(u))`,
  not cell by cell.

## With threads

- Keep dimension 1 whole: `dims[1] = 1`, balance the rest. Splitting the
  contiguous axis turns the inter-tile copies into strided gathers.
- Leave `synchronize_halo!(u)` serial unless a benchmark shows that
  `threads=true` wins; the edge slabs are thin, and the spawn cost often
  dominates.
- Choose the [`ThreadBackend`](@ref) by per-call overhead: `ThreadsBackend` by
  default, `PolyesterBackend` for many small calls, `SerialBackend` to debug a
  race.

## With several fields

Put fields that share geometry and halo width in a [`MultiHaloArray`](@ref) or
an [`ArrayOfHaloArray`](@ref), so one `synchronize_halo!(state)` and one
`dot(state, state)` cover them all, and the Krylov solvers see one vector.
Keep separate arrays when the halo widths or layouts differ. Read and write
all fields at one cell through [`siteview`](@ref) or [`sitevector`](@ref);
iterate the fields with [`eachfield`](@ref).

## Instead of …

| instead of | write |
|:-----------|:------|
| `for i in 1:nx, j in 1:ny; u[i, j] = f(i, j); end` | `fill_from_global_indices!(I -> f(I...), u)` |
| `u[i, j]` inside a stencil | `parent(u)[I]` over `CartesianIndices(interior_range(u))` |
| `u[:, 1]`, `vec(u)`, `u[3]` | `interior_view(u)[:, 1]`, `vec(interior_view(u))`, or `gather_haloarray(u)` |
| writing `parent(u)` ghost cells by hand | `synchronize_halo!(u)`, or a boundary rule, or `apply_coupled_bc!` |
| `if rank == 0; s = sum(u); end` | `s = sum(u)` on every rank, then `is_root(u) && …` |
| `sum(u; dims=2)` inside the time loop | a `DimReductionPlan` built once, `reduce!` in the loop |
| `u isa HaloArray ? mpi_path : serial_path` | a method per `halo_backend(u)` |
| `Array(u)` on a distributed array | `gather_haloarray(u)` |
| `cu(parent(u))` and rewrapping | `storage=CUDA.zeros` at construction, or `adapt(CuArray, u)` |
