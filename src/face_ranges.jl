@inline function _dim_slab_range(ranges::NTuple{N,Any}, dim::Int, range) where {N}
    return ntuple(d -> d == dim ? range : ranges[d], Val(N))
end

# Spatial dimensionality / interior range come from the shared _spatial_*
# helpers in abstract_haloarray.jl (single arrays, collections, and raw arrays
# of fields).

@inline function _check_face_dim(halo, dim::Int)
    spatial_ndims = _spatial_ndims(halo)
    1 <= dim <= spatial_ndims ||
        throw(ArgumentError("dim must be a spatial dimension in 1:$spatial_ndims, got $dim"))
    return nothing
end

"""
    interior_face_range(halo, dim)

Return every face touching the interior in dimension `dim`, identified by its
lower-index cell: the low-boundary `ghost | interior` face, all internal
`interior | interior` faces, and the high-boundary `interior | ghost` face. This
is one contiguous span — `(first_interior - 1):last_interior` along `dim`, full in
every transverse dimension. Pair each `IL` with `IR = IL + unit_vector(halo, dim)`;
a conservative flux loop scatters each face's flux onto both cells (the harmless
ghost writes at the two boundary faces fall on allocated halo cells).
"""
function interior_face_range(halo, dim::Int)
    _check_face_dim(halo, dim)
    # The two boundary faces scatter into ghost cells; with halo width 0 there
    # are none — the range would start at storage index 0 and the @inbounds
    # flux loop would corrupt memory. The face sweep is undefined without ghosts.
    halo_width(halo) >= 1 || throw(ArgumentError(
        "FaceRanges/interior_face_range require halo width >= 1: the boundary " *
        "faces scatter into the ghost cells, which a halo-0 array does not have."))
    ranges = _spatial_interior_range(halo)
    return ntuple(_spatial_ndims(halo)) do d
        d == dim ? ((first(ranges[d]) - 1):last(ranges[d])) : (first(ranges[d]):last(ranges[d]))
    end
end

interior_face_range(halo, ::Dim{D}) where {D} = interior_face_range(halo, D)

"""
    FaceRanges(halo)

Precompute the face index ranges and offsets for conservative flux-divergence
loops. The stored indices identify the lower-index cell of each face; add
`unit_vector(ranges, dim)` to reach the upper-index cell. Requires halo
width ≥ 1 (throws otherwise): the two boundary faces scatter into ghost cells.

For `MultiHaloArray` and `ArrayOfHaloArray`, the ranges are spatial only. Apply
them after selecting an individual field.

[`interior_faces(ranges, dim)`](@ref interior_faces) gives every face touching the
interior along `dim` (the two boundary faces plus the internal faces) as one
iterable; a conservative update scatters each face's flux onto both cells:

```julia
ranges = FaceRanges(u)
e      = unit_vector(ranges, dim)
udata  = parent(u); dudata = parent(du)

for IL in interior_faces(ranges, dim)
    IR   = IL + e
    flux = numerical_flux(udata[IL], udata[IR])
    dudata[IL] -= flux       # ghost write at the two boundary faces is in-bounds & harmless
    dudata[IR] += flux
end
```
"""
struct FaceRanges{F,D,Halo}
    faces::F          # per-direction: every face touching the interior (lower-index cells)
    unit_vector::D
    halo::Halo
end

function FaceRanges(halo)
    spatial_ndims = _spatial_ndims(halo)
    return FaceRanges(
        ntuple(d -> CartesianIndices(interior_face_range(halo, d)), spatial_ndims),
        ntuple(d -> unit_vector(halo, d), Val(spatial_ndims)),
        halo_width(halo),
    )
end

"""
    interior_faces(ranges, dim)
    interior_faces(ranges, dim, color)

Every face touching the interior along `dim`, identified by its lower-index cell
`IL`: the low-boundary `ghost | interior` face, the internal `interior | interior`
faces, and the high-boundary `interior | ghost` face — one contiguous iterable.
Pair each `IL` with `IL + unit_vector(ranges, dim)`. With a `color` (`0`/`1`)
argument, returns one race-free checkerboard color (for parallel scatter).
"""
interior_faces(ranges::FaceRanges, dim::Int) = ranges.faces[dim]
interior_faces(ranges::FaceRanges, ::Dim{D}) where {D} = interior_faces(ranges, D)
"""
    unit_vector(x[, dim])

The `CartesianIndex` unit step along dimension `dim` (with no `dim`, the tuple
of unit steps for every dimension). `x` can be a `FaceRanges`, a halo array, or
a `Val(N)` dimension count; `dim` may be an `Int` or a `Dim{D}`. Add it to a
lower-face index `IL` to reach the cell across the face,
`IL + unit_vector(x, dim)`, or use it as a stencil offset,
`u[I + unit_vector(u, dim)]`.
"""
unit_vector(ranges::FaceRanges) = ranges.unit_vector
unit_vector(ranges::FaceRanges, dim::Int) = ranges.unit_vector[dim]
unit_vector(ranges::FaceRanges, ::Dim{D}) where {D} = unit_vector(ranges, D)

@inline unit_vector(::Val{N}) where {N} = map(CartesianIndex, versors(Val(N)))
@inline unit_vector(::Val{N}, dim::Int) where {N} = CartesianIndex(versors(Val(N))[dim])

@inline function unit_vector(halo, dim::Int)
    _check_face_dim(halo, dim)
    return unit_vector(Val(_spatial_ndims(halo)), dim)
end
@inline unit_vector(halo, ::Dim{D}) where {D} = unit_vector(halo, D)
@inline unit_vector(halo::AbstractSingleHaloArray) = unit_vector(Val(_spatial_ndims(halo)))

# Default cell accessors for scalar fields.
@inline _scalar_face_read(data, I) = @inbounds data[I]
@inline function _scalar_face_scatter!(data, I, scale, F)
    @inbounds data[I] += scale * F
    return data
end

"""
    accumulate_flux_divergence!(du, u, ranges::FaceRanges, dim, scale, flux,
                                read=…, scatter! =…)

Accumulate the conservative finite-volume flux divergence along `dim` into `du`,
using the precomputed `ranges`. This is the left / internal / right face loop in
a single call.

Each face's numerical `flux` is evaluated once from the two adjacent cell states
and scattered with opposite signs onto the interior cells — `du[IL] -= scale*F` and
`du[IR] += scale*F` — skipping whichever side is a ghost at a physical boundary.

- `flux(UL, UR) -> F`: numerical flux between the lower (`UL`) and upper (`UR`)
  cell states.
- `read(data, I) -> U`: cell state at storage index `I`. Defaults to `data[I]`
  for scalar fields; pass e.g. a `conserved_cell`-style gather to build a
  multi-field `SVector`.
- `scatter!(data, I, s, F)`: add `s*F` into cell `I`. Defaults to
  `data[I] += s*F`; pass a multi-field writer for collections.

`du` and `u` are the raw storage the accessors understand — `parent(field)` for
a single field, or [`field_storages(state)`](@ref field_storages) (a NamedTuple/
array of padded backing arrays) for a multi-field collection — not the halo
arrays themselves. `dim` may be an `Int` or a `Dim{D}`.

```julia
ranges = FaceRanges(u)
accumulate_flux_divergence!(parent(du), parent(u), ranges, 1, inv(dx), numerical_flux)
```
"""
function accumulate_flux_divergence!(du, u, ranges::FaceRanges, dim, scale,
        flux, read, scatter!)
    e = unit_vector(ranges, dim)

    @inbounds for IL in interior_faces(ranges, dim)
        IR = IL + e
        F  = flux(read(u, IL), read(u, IR))
        scatter!(du, IL, -scale, F)
        scatter!(du, IR,  scale, F)
    end

    return du
end

@inline accumulate_flux_divergence!(du, u, ranges::FaceRanges, dim, scale, flux) =
    accumulate_flux_divergence!(du, u, ranges, dim, scale, flux,
        _scalar_face_read, _scalar_face_scatter!)

# Shared by face- and cell-range/region loops (defined here as face_ranges.jl is
# included first). _spatial_ndims / _spatial_interior_range dispatch the spatial
# dimensionality and interior range over single arrays, collections, and raw
# field arrays.
@inline function _check_loop_color(color::Integer)
    (color == 0 || color == 1) ||
        throw(ArgumentError("color must be 0 or 1, got $color"))
    return Int(color)
end

@inline _loop_strided_range(first_index::Int, len::Int, stride::Int) =
    first_index:stride:(first_index + stride * (len - 1))

@inline function _colored_face(indices::CartesianIndices{N}, dim::Int, color::Integer) where {N}
    checked_color = _check_loop_color(color)
    first_tuple = Tuple(first(indices))
    region_size = size(indices)
    delta = mod(checked_color - mod(first_tuple[dim], 2), 2)
    colored_size_dim = region_size[dim] <= delta ? 0 : cld(region_size[dim] - delta, 2)

    return CartesianIndices(ntuple(Val(N)) do d
        len = d == dim ? colored_size_dim : region_size[d]
        stride = d == dim ? 2 : 1
        start = first_tuple[d] + (d == dim ? delta : 0)
        _loop_strided_range(start, len, stride)
    end)
end

interior_faces(ranges::FaceRanges, dim::Int, color::Integer) =
    _colored_face(interior_faces(ranges, dim), dim, color)
interior_faces(ranges::FaceRanges, ::Dim{D}, color::Integer) where {D} =
    interior_faces(ranges, D, color)

