# Index spaces

A halo array describes one logical domain but stores it in blocks, each padded
with ghost cells. Every function in the API therefore works in one of a few
coordinate systems, and the first question to ask of any index is *which space
is it in*. This page names the spaces, shows how they relate, and lists which
functions take which. The working rules that follow from it are collected in
[Good habits](habits.md).

## The spaces

Take a 1-D domain of 32 cells with halo width 2, split into four blocks (MPI
ranks, or the tiles of a `ThreadedHaloArray`). The block that owns global cells
9 to 16 sees its data as:

```
storage    1   2 │ 3   4   5   6   7   8   9  10 │ 11  12
           g   g │ ·   ·   ·   ·   ·   ·   ·   · │ g   g       g = ghost cell
interior         │ 1   2   3   4   5   6   7   8 │
global           │ 9  10  11  12  13  14  15  16 │
```

| space | `1` is … | used by |
|:------|:---------|:--------|
| **global** | the first cell of the whole domain | `size`, `axes`, `eachindex`, `u[I]`, [`fill_from_global_indices!`](@ref), [`global_index`](@ref) in a boundary rule, checkerboard colours, the array [`gather_haloarray`](@ref) returns |
| **interior** (owned) | the first cell this block owns, ghosts excluded | [`interior_view`](@ref), [`interior_size`](@ref), [`interior_axes`](@ref), the input of [`interior_to_global_index`](@ref) |
| **storage** (padded) | the first ghost cell of the block | `parent`, [`tile_parent`](@ref), [`interior_range`](@ref), [`CellRanges`](@ref), [`FaceRanges`](@ref), [`siteview`](@ref), the cell geometry, the output of [`cell_index`](@ref) |
| **launch** | the first work item of a kernel | `@index(Global, NTuple)`, the input of [`cell_index`](@ref) and [`cell_linear_index`](@ref) |
| **tile** | the first tile of a [`ThreadedHaloArray`](@ref) | [`tile_parent`](@ref), [`tile_coordinates`](@ref), [`neighbor_tile_id`](@ref), the `tile` argument of every per-tile accessor |

On a [`LocalHaloArray`](@ref) the global and interior spaces coincide: the one
block is the whole domain. On an MPI [`HaloArray`](@ref), and on each tile of a
`ThreadedHaloArray`, they differ by the block's offset, and that offset is the
only thing a block knows about the rest of the domain. The storage space is
always the interior space shifted by the halo width.

Two more coordinate systems appear in specific places: the **face** coordinates
a boundary rule sees, and the **collection** coordinates of a multi-field state.
Both are described at the end of this page.

## Global

Every halo array reports the **whole domain**: `size(u)`, `axes(u)` and
`length(u)` are the same on every rank and every backend. This is what lets a
halo array stand in for an ordinary array in a Krylov solver or an ODE
integrator, whose code reasons about the vector as a whole.

`eachindex(u)` is the set of **global** indices of the cells this block owns:
the whole domain on the serial kinds, this rank's block on MPI. So

```julia
for I in eachindex(u)
    u[I]                      # valid on every rank
end
```

is correct everywhere, and `sum(u[I] for I in eachindex(u))` equals the
rank-local part of `sum(u)`.

Scalar indexing `u[i, j, …]` takes **global** coordinates and reaches only the
cells this block holds. A `LocalHaloArray` throws a `BoundsError` outside the
domain; an MPI `HaloArray` throws an `ArgumentError` for a cell another rank
owns, and warns once that scalar access is local-only. It is meant for setup
and diagnostics, not for stencils: it performs no communication and compiles to
a bounds check plus an offset, which is far slower than a loop over the
storage block. See [What scalar indexing accepts](@ref) below for the forms
that are refused outright.

The other global entry points are:

- [`fill_from_global_indices!`](@ref)`(f, u)`: sets every owned cell to `f(I)`
  with `I` the global index tuple. One `f` gives one consistent field whatever
  the decomposition.
- [`global_index`](@ref)`(u)` inside a boundary rule: the global
  `CartesianIndex` of the ghost cell being filled, for position-dependent
  conditions.
- The checkerboard colour of a cell is its **global** parity
  `mod(sum(I), 2)`, so red-black sweeps stay consistent across rank and tile
  seams.
- [`gather_haloarray`](@ref)`(u)` returns the global interior as one `Array`
  (on MPI: on the root rank only).

## Interior

The interior space is the block's **owned** cells, ghosts excluded, numbered
from 1. [`interior_view`](@ref)`(u)` is a view of exactly those cells, and it is
an ordinary array: slice it, broadcast into it, pass it to any library.

```julia
v = interior_view(u)
v .= data                     # bulk initialisation of this block's cells
v[2, :]                       # slicing: a normal SubArray
interior_size(u)              # its size, (nx_owned, ny_owned, …)
interior_axes(u)              # its axes, 1-based
```

On a `ThreadedHaloArray` the interior is **per tile**: `interior_view(u, t)`,
and [`CellRanges`](@ref)`(u, t)` for the loop ranges of tile `t`. On a
collection, `interior_view` returns a container of views, one per field.

[`interior_to_global_index`](@ref)`(u, I)` maps an interior index tuple to the
global one, `interior_to_global_index(u, t, I)` does the same for tile `t`. The
offset is constant per block, so compute it once outside a loop:

```julia
first = interior_to_global_index(u, ntuple(_ -> 1, ndims(u)))   # global index of owned cell (1, …, 1)
```

## Storage

The storage space is the padded block itself: `parent(u)` for a single-block
array, [`tile_parent`](@ref)`(u, t)` for tile `t` of a `ThreadedHaloArray`
(and for a single-block array with `t = 1`, so one loop serves both). Interior
cell `k` sits at storage index `k + halo_width(u)`; ghost cells occupy the first
and last `halo_width(u)` positions along every axis.

This is where stencils live, because here a neighbour is just an offset and the
ghosts make every neighbour read in-bounds:

```julia
h  = halo_width(u)
s  = parent(u)
e1 = unit_vector(u, 1)
for I in CartesianIndices(interior_range(u))         # interior cells, storage coordinates
    du[I] = (s[I + e1] - 2s[I] + s[I - e1]) * inv_dx2
end
```

The loop helpers all speak storage coordinates:

- [`interior_range`](@ref)`(u)`: the interior as a tuple of storage ranges,
  `(h+1):(n+h)` per axis.
- [`CellRanges`](@ref) and [`interior_cells`](@ref)`(ranges[, color])`: the
  interior cells, all of them or one checkerboard colour.
- [`FaceRanges`](@ref), [`interior_faces`](@ref) and [`unit_vector`](@ref):
  faces for finite-volume updates, each face named by its lower cell.
- [`siteview`](@ref)`(state, I[, tile])`, [`sitevector`](@ref): all fields at
  storage cell `I`.
- [`cell_center`](@ref), [`cell_width`](@ref), [`cell_volume`](@ref) and the
  other geometry functions: the cell at storage index `I`.

[`edge_view`](@ref)`(u, side, dim[, tile])` and [`ghost_view`](@ref) are views
of the `halo`-wide bands next to one wall. Like any view they carry their own
1-based indices; a boundary condition pairs them cell by cell.

## Launch

A GPU kernel receives a work-item index from `@index(Global, NTuple)`, numbered
from 1 over the launch range. The window types turn it into a storage index:

```julia
@kernel function laplacian!(du, u, region::CellWindow{2}, inv_dx2)
    J = @index(Global, NTuple)
    I = cell_index(region, J)                 # storage cell of this work item
    if is_cell_index_inbounds(region, I)
        @inbounds du[I...] = (u[I[1]+1, I[2]] - 2u[I...] + u[I[1]-1, I[2]]) * inv_dx2
    end
end
region = interior_cell_window(CellRanges(u))
laplacian!(backend, (16, 16))(parent(du), parent(u), region, inv_dx2; ndrange=region.size)
```

[`CellWindow`](@ref) covers every interior cell; [`CellCheckerboard`](@ref)
covers one colour with the launch compressed by two along one axis, which is
why kernels check [`is_cell_index_inbounds`](@ref). [`FaceWindow`](@ref) and
[`FaceCheckerboard`](@ref) do the same for faces. The ranges, the colours and
every window are described in [Cell and face loops](loops.md).

For nearest-neighbour stencils, [`linear_cell_window`](@ref) gives a
[`LinearCellWindow`](@ref) addressed by storage **linear** index:
[`cell_linear_index`](@ref)`(w, J)` converts the launch index once per cell and
[`cell_strides`](@ref)`(w)` are the neighbour offsets, `x[i + s[d]]` and
`x[i - s[d]]`. The index type is `Int32` by default.

## Tile

A `ThreadedHaloArray` is a Cartesian grid of tiles, numbered `1:tile_count(u)`
in column-major order of the grid. [`tile_coordinates`](@ref)`(u, t)` is the
tile's position in that grid, [`tile_size`](@ref)`(u)` the owned cells per tile,
and [`neighbor_tile_id`](@ref)`(u, t, dim, side)` the adjacent tile or `0` at a
non-periodic domain edge. A tile's first owned cell has global index
`(tile_coordinates(u, t) .- 1) .* tile_size(u) .+ 1`.

The single-block kinds are a one-tile decomposition: `tile_count` is 1 and
`tile_parent(u, 1)` is `parent(u)`. So the per-tile loop is the one loop that
runs unchanged on every backend:

```julia
for t in 1:tile_count(u)
    s = tile_parent(u, t)
    for I in interior_cells(CellRanges(u, t))
        # s[I], s[I + e1], …
    end
end
```

Every per-tile accessor takes the tile as its last argument: `interior_view(u,
t)`, `edge_view(u, side, dim, t)`, `siteview(state, I, t)`. A coupled boundary
condition receives `tile` from the driver, `nothing` for a whole block and the
tile id on a threaded field, and forwards it as it is.

## Face coordinates in a boundary rule

A boundary condition is a rule `bc(u, k)` for one ghost cell (see
[Custom conditions as rules](@ref) in the guide). Inside it:

- `k` is the ghost cell's distance from the wall, `1:halo_width`.
- `u(j)` is the interior cell `j` steps in from the wall along the face normal;
  `u(0)` touches the wall. The other coordinates are the ghost cell's own.
- [`far`](@ref)`(u)` is the same accessor from the opposite wall of the block.
- [`global_index`](@ref)`(u)` is the ghost cell's global `CartesianIndex`;
  `u.I` is its storage index and [`face_dim`](@ref)`(u)` the normal axis.

A face is named by [`Side`](@ref) (`1` the low end, `2` the high end) and
[`Dim`](@ref) (the axis), as in `edge_view(u, Side(2), Dim(1))`.

## Collection coordinates

A [`MultiHaloArray`](@ref) or [`ArrayOfHaloArray`](@ref) has the field axes
**first**, then the spatial axes: `size(state) == (field_shape(state)...,
size(field)...)`. `eachindex(state)` and the `dims=` keyword of the reductions
use this order, so `sum(state; dims=1)` sums over the fields of a flat
collection and `sum(state; dims=2)` over the first spatial axis.

The per-cell accessors take **spatial** indices only: `siteview(state, I)` with
`I` a storage index, `fill_from_global_indices!(f, state)` with `f(I)` receiving
the global spatial index and returning one value per field.

## Converting between spaces

| from → to | how |
|:----------|:----|
| interior → storage | add `halo_width(u)` to every coordinate; [`interior_range`](@ref)`(u)` is the whole interior in storage coordinates |
| storage → interior | subtract `halo_width(u)` |
| interior → global | [`interior_to_global_index`](@ref)`(u, I)`, or `(u, t, I)` for tile `t` |
| global → interior | `I .- first .+ 1` with `first = interior_to_global_index(u, (1, …, 1))`; `eachindex(u)` lists the global indices this block owns |
| global → storage | `u[I]` does it for one owned cell; in a loop compute `offset = first .- 1 .- halo_width(u)` once and index `parent(u)` at `I .- offset` |
| launch → storage | [`cell_index`](@ref)`(region, J)`, [`cell_linear_index`](@ref)`(window, J)` |
| tile → global | `(tile_coordinates(u, t) .- 1) .* tile_size(u)` is the tile's global offset |
| ghost cell → global | [`global_index`](@ref)`(u)` inside a rule |

## What scalar indexing accepts

A halo array reports its global shape but stores only this block's piece, so
it deliberately supports a *narrower* indexing surface than a plain
`AbstractArray`. Everything it refuses, it refuses loudly, with the fast
alternative named in the error.

**What works:**

| form | behaviour |
|:-----|:----------|
| `u[i, j, …]` (all `ndims` indices, **global** coordinates) | reads or writes the cell; on MPI only for cells this rank owns (warns; diagnostics only) |
| `u[i, j, …, 1]` (trailing `1`s past `ndims`) | allowed, per the `AbstractArray` contract (generic LinearAlgebra code such as `Diagonal`'s `ldiv!` relies on it) |

**What is refused, and what to use instead:**

| form | why | instead |
|:-----|:----|:--------|
| `u[3]`, linear indexing of an N-d array | would route generic code through a slow scalar path, and is ill-defined across ranks | `interior_view(u)[3]`, or loop `CartesianIndices(interior_view(u))` (interior) or `eachindex(u)` (global, valid in `u[I]`) |
| `u[:]`, `u[:, 1]`, `u[1:2, :]`, slices | would materialise cross-rank or cross-tile data cell by cell | slice the view, `interior_view(u)[:, 1]`, or [`gather_haloarray`](@ref)`(u)` for the true global array |

Everything performance-relevant, broadcast, reductions, `dot` and `norm`, the
BLAS-1 updates, is specialised and never touches scalar indexing, so these
restrictions only surface in hand-written loops over the array itself.

## Plain arrays in a broadcast

The same global-versus-local rule governs a plain-array operand in an in-place
broadcast, `u .= u .+ g`. On the shared-memory backends a plain operand is
indexed by **global** interior coordinates: a `ThreadedHaloArray` hands each
tile a view of its own global window, with the usual size-1 expansion per
dimension. On a distributed `HaloArray` a plain operand must be **owned-size**,
shaped like `interior_view(u)`: a global-shaped operand would require every
rank to hold the whole array or an implicit scatter. Distribute global data
explicitly, with [`fill_from_global_indices!`](@ref) or a gather in reverse,
and broadcast with local operands.
