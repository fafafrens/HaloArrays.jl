# Cell and face loops

The loop helpers answer one question: *which storage cells does this sweep
visit, in what order, and which of them may be updated at the same time?* They
come in two families, cells and faces, and each family has a CPU form, a
`CartesianIndices` you iterate, and a kernel form, a small window struct that
travels into a GPU kernel together with a launch size. Everything here is in
**storage coordinates** on one block, `parent(u)` or [`tile_parent`](@ref)`(u,
t)`, ghosts included, so every neighbour read is in bounds (see
[Index spaces](indexing.md)).

A 1-D block with halo width 1 and six owned cells, as the helpers see it:

```
storage index    1   2   3   4   5   6   7   8
                 g   ·   ·   ·   ·   ·   ·   g      interior_range(u) = 2:7
global index         1   2   3   4   5   6
cell colour          1   0   1   0   1   0          mod(global index, 2)
faces, by IL       1   2   3   4   5   6   7        interior_faces(fr, 1) = 1:7,  IR = IL + 1
face colour        1   0   1   0   1   0   1        mod(IL, 2)
```

| | all | one colour | kernel window, all | kernel window, one colour |
|:--|:--|:--|:--|:--|
| **cells** | [`interior_cells`](@ref)`(cr)` | `interior_cells(cr, c)` | [`interior_cell_window`](@ref)`(cr)` → [`CellWindow`](@ref) | `interior_cell_window(cr, c)` → [`CellCheckerboard`](@ref) |
| **faces along `d`** | [`interior_faces`](@ref)`(fr, d)` | `interior_faces(fr, d, c)` | [`interior_face_window`](@ref)`(fr, d)` → [`FaceWindow`](@ref) | `interior_face_window(fr, d, c)` → [`FaceCheckerboard`](@ref) |
| **cells, linear index** | [`linear_cell_window`](@ref)`(cr, data)` | `linear_cell_window(cr, data, c)` | | |

with `cr = CellRanges(u[, t])` and `fr = FaceRanges(u)`. Both accept a
collection as well as a single array; the ranges are then spatial, to be used
on each field's storage or through [`siteview`](@ref).

## Cell loops

[`CellRanges`](@ref)`(u)` precomputes the interior-cell ranges of a block;
[`interior_cells`](@ref)`(cr)` is the `CartesianIndices` of every owned cell in
storage coordinates. This is the loop for an out-of-place stencil, a source
term, anything that reads neighbours and writes a different array:

```julia
cr = CellRanges(u)
s, d = parent(u), parent(du)
e1, e2 = unit_vector(u, 1), unit_vector(u, 2)
@inbounds for I in interior_cells(cr)
    d[I] = (s[I + e1] + s[I - e1] + s[I + e2] + s[I - e2] - 4s[I]) * inv_h2
end
```

On a `ThreadedHaloArray` build the ranges **per tile**, `CellRanges(u, t)`,
inside the tile loop. The single-block kinds have one tile, so
`for t in 1:tile_count(u)` with `tile_parent(u, t)` and `CellRanges(u, t)` is
the one loop that runs on every backend.

### Checkerboard colours

An **in-place** nearest-neighbour update, Gauss–Seidel, SOR, a Metropolis
sweep, must not read a neighbour that is being written at the same time. The
checkerboard splits the cells into two colours such that no two cells of one
colour are neighbours, so one colour can be updated in any order, in parallel,
reading only the other.

A cell's colour is the parity of its **global** index, `mod(sum(I), 2)` with
`I` the global index tuple. Global, not storage, so that the colouring is one
continuous checkerboard across rank and tile seams: the ghost cells a block
receives from its neighbour have the colour they have in the neighbour. Each
`CellRanges` carries the offset between its block's storage parity and the
global parity, which is why the ranges are built per tile.

[`interior_cells`](@ref)`(cr, color)` returns the cells of one colour, `0` or
`1`, as a **tuple** of strided `CartesianIndices`: a checkerboard in `N`
dimensions is not one rectangular stride-2 range, so the helper fixes the
parity of the first `N-1` coordinates in every one of the `2^(N-1)` ways and
lets the last coordinate step by two. Iterate the tuple, then each range:

```julia
cr = CellRanges(u)
s, f = parent(u), parent(rhs)   # padded blocks, storage coordinates on both sides
for color in (0, 1)
    @inbounds for sub in interior_cells(cr, color), I in sub
        s[I] = 0.25 * (s[I + e1] + s[I - e1] + s[I + e2] + s[I - e2] - h2 * f[I])
    end
    synchronize_halo!(u)        # the other colour's ghosts changed on the neighbours
end
```

The synchronisation between the colours is part of the method: after colour
0 is updated on every block, colour 1 reads colour-0 neighbours, some of which
sit in the ghost cells and were updated by another rank or tile.

## Face loops

A finite-volume update evaluates one numerical flux per **face** and scatters
it, with opposite signs, onto the two cells the face separates. A face is named
by its **lower** cell `IL`; the upper cell is `IL + `[`unit_vector`](@ref)`(fr,
d)`. [`FaceRanges`](@ref)`(u)` precomputes, per dimension, the faces that touch
the interior: along `d` the lower cells run from `first_interior - 1` to
`last_interior`, `n + 1` faces for `n` cells, and all interior cells in the
transverse directions. The first and last of them are the two boundary faces,
`ghost | interior` and `interior | ghost`. It requires a halo width of at least
1, because of them.

```julia
fr = FaceRanges(u)
e  = unit_vector(fr, dim)
s, d = parent(u), parent(du)
@inbounds for IL in interior_faces(fr, dim)
    IR = IL + e
    F  = numerical_flux(s[IL], s[IR])
    d[IL] -= F * inv_dx
    d[IR] += F * inv_dx
end
```

At the two boundary faces one of the two writes lands on a ghost cell of `du`.
That cell is allocated, nobody reads it, and the next `synchronize_halo!(du)`
overwrites it, so the loop has no branch. [`accumulate_flux_divergence!`](@ref)
is exactly this loop in one call, with `read` and `scatter!` hooks for
multi-field states and a `do`-block form for the flux.

### Face colours

Two faces along `d` that are adjacent share a cell, so a **parallel** scatter
over all faces races on every cell. Faces two apart share nothing.
[`interior_faces`](@ref)`(fr, d, color)` returns the faces whose lower cell has
coordinate parity `color` along `d`, as one `CartesianIndices` with step 2 in
that direction; scatter one colour, then the other. The face colour is the
parity of the **storage** coordinate, not the global one: the race it prevents
is between writes into one block's storage, and nothing has to agree across a
seam. The sum of the two colours is the full face set.

## Several fields at one cell

The ranges of a collection are spatial, so a loop over them visits one cell of
**every** field at once. Two accessors read and write all the fields there,
with the same storage index the ranges produce and the same tile argument as
the ranges were built with, so no coordinate ever changes hands:

- [`sitevector`](@ref)`(state, I[, t])` returns the fields at cell `I` as an
  `SVector{N}`, `N` known from the type of `state`: a copy, for arithmetic.
- [`siteview`](@ref)`(state, I[, t])` returns a lazy, writable array shaped
  like the fields, for writing back.

A cell update of a conserved state, every field read once, every field written
once, is then one `SVector` expression:

```julia
cr = CellRanges(state)                                 # spatial ranges of the collection
e1, e2 = unit_vector(state, 1), unit_vector(state, 2)
for I in interior_cells(cr)
    q  = sitevector(state, I)                          # SVector{4}: ρ, ρu, ρv, E at this cell
    qE = sitevector(state, I + e1)                     # the neighbour's state
    siteview(dstate, I) .= rhs(q, qE, sitevector(state, I - e1),
                               sitevector(state, I + e2), sitevector(state, I - e2))
end
```

The same two calls are the `read` and `scatter!` hooks of
[`accumulate_flux_divergence!`](@ref) for a multi-field flux, where the
numerical flux takes and returns an `SVector`:

```julia
fr = FaceRanges(state)
accumulate_flux_divergence!(field_storages(dstate), field_storages(state), fr, dim, inv_dx, hll_flux;
    read     = (_, I) -> sitevector(state, I),
    scatter! = (_, I, s, F) -> (siteview(dstate, I) .+= s .* F))
```

On a `ThreadedHaloArray` collection pass the tile to both accessors, `sitevector(state,
I, t)` and `siteview(dstate, I, t)`, inside the tile loop. `sitevector` needs a
compile-time field count, so it serves a `MultiHaloArray` (nested or not) and a
single array; an `ArrayOfHaloArray` backed by an `Array` has a runtime field
count and uses `SVector{N}(siteview(state, I))` with `N` written out.

## Kernel windows

A `CartesianIndices` is a poor kernel argument. The windows are the same
regions as isbits structs: `first`, the launch `size` to pass as `ndrange`,
and whatever the mapping needs. Inside the kernel, [`cell_index`](@ref)`(region,
J)` turns the work-item index `J` from `@index(Global, NTuple)` into the storage
cell, for every window type. See `examples/tutorials/gpu.jl`.

[`CellWindow`](@ref), from [`interior_cell_window`](@ref)`(cr)`, covers every
interior cell; the launch is exact.

[`CellCheckerboard`](@ref), from `interior_cell_window(cr, color;
compressed_dim=1)`, covers one colour. The launch is **compressed by two** in
`compressed_dim`: it has `cld(n, 2)` work items along that axis, and
`cell_index` reconstructs the cell by stepping two along it and shifting by 0
or 1 to land on the colour. When the extent is odd the last work item can
reconstruct a cell one past the interior, so a colour kernel always guards with
[`is_cell_index_inbounds`](@ref)`(region, I)`:

```julia
@kernel function red_black!(s, region::CellCheckerboard{2}, h2, f)
    J = @index(Global, NTuple)
    I = cell_index(region, J)
    if is_cell_index_inbounds(region, I)
        @inbounds s[I...] = 0.25 * (s[I[1]+1, I[2]] + s[I[1]-1, I[2]] +
                                    s[I[1], I[2]+1] + s[I[1], I[2]-1] - h2 * f[I...])
    end
end
for color in (0, 1)
    region = interior_cell_window(cr, color)
    red_black!(backend, (16, 16))(parent(u), region, h2, parent(f); ndrange=region.size)
    synchronize_halo!(u)
end
```

The window carries the block's parity offset, so the colour is the global one
here too. `compressed_dim` is a free choice; the default is the first axis.

[`FaceWindow`](@ref), from [`interior_face_window`](@ref)`(fr, d)`, maps `J` to
the lower cell and stores the `offset` to the upper one, `IR = IL +
region.offset`. [`FaceCheckerboard`](@ref), from `interior_face_window(fr, d,
color)`, is one face colour with a `stride` of two along `d`; face launches are
exact, the in-bounds check only guards against launch overshoot.

### Linear-index windows

A kernel that addresses the storage through a Cartesian index redoes the
conversion to a linear address for every neighbour read. A
[`LinearCellWindow`](@ref), from [`linear_cell_window`](@ref)`(cr, data[,
color])`, converts once per cell: [`cell_linear_index`](@ref)`(w, J)` is the
storage linear index of the work item's cell and [`cell_strides`](@ref)`(w)`
are the neighbour offsets, so a nearest-neighbour stencil reads `x[i + s[d]]`
and `x[i - s[d]]` with one add each. The index type is `Int32` unless the
storage exceeds `2^31` elements (`index_type=Int`). Measured on Metal this is
1.4 to 2 times faster than `cell_index` for nearest-neighbour stencils in 4-D.

```julia
w = linear_cell_window(u, color)            # one tile; or linear_cell_window(CellRanges(u, t), tile_parent(u, t), color)
@kernel function update!(x, w, m2)
    J = @index(Global, NTuple)
    if is_cell_linear_index_inbounds(w, J)  # takes the LAUNCH index, unlike is_cell_index_inbounds
        i = cell_linear_index(w, J)
        s = cell_strides(w)
        ϕ₊ = map(sμ -> x[i + sμ], s)
        ϕ₋ = map(sμ -> x[i - sμ], s)
        x[i] = ...
    end
end
update!(backend)(parent(u), w, m2; ndrange=w.size)
```

Note the two in-bounds guards differ: [`is_cell_index_inbounds`](@ref) takes
the reconstructed storage cell, [`is_cell_linear_index_inbounds`](@ref) takes
the launch index, before any conversion. A full linear window needs no guard
and the check returns `true`.

## Which form

| sweep | helper |
|:------|:-------|
| out-of-place stencil, source term, any read-neighbours-write-elsewhere | `interior_cells(cr)`, or `CellWindow` on a GPU |
| in-place nearest-neighbour update (Gauss–Seidel, SOR, Metropolis) | `interior_cells(cr, color)`, or `CellCheckerboard`; `synchronize_halo!` between the colours |
| conservative flux update | `interior_faces(fr, d)` or `accumulate_flux_divergence!`; `FaceWindow` on a GPU |
| the same with a parallel scatter and no atomics | `interior_faces(fr, d, color)`, or `FaceCheckerboard` |
| all fields of a collection at one cell | `sitevector` to read, `siteview` to write, at the loop's storage index |
| nearest-neighbour stencil on a GPU | `LinearCellWindow` with `cell_strides` |
| several tiles | one `CellRanges(u, t)`, and one window, per tile |
