@inline function _dim_slab_range(ranges::NTuple{N,Any}, dim::Int, range) where {N}
    return ntuple(d -> d == dim ? range : ranges[d], Val(N))
end

@inline _loop_ndims(halo::AbstractSingleHaloArray) = ndims(halo)
@inline _loop_ndims(::MultiHaloArray{T,N}) where {T,N} = N
@inline _loop_ndims(::ArrayOfHaloArray{T,N}) where {T,N} = N
@inline _loop_ndims(arr::AbstractArray{<:AbstractSingleHaloArray}) = ndims(first(arr))

@inline _loop_interior_range(halo::AbstractSingleHaloArray) = interior_range(halo)
@inline _loop_interior_range(halo::MultiHaloArray) =
    interior_range(first(values(halo.arrays)))
@inline _loop_interior_range(halo::ArrayOfHaloArray) =
    interior_range(first(parent(halo)))
@inline _loop_interior_range(arr::AbstractArray{<:AbstractSingleHaloArray}) =
    interior_range(first(arr))

@inline function _check_face_dim(halo, dim::Int)
    spatial_ndims = _loop_ndims(halo)
    1 <= dim <= spatial_ndims ||
        throw(ArgumentError("dim must be a spatial dimension in 1:$spatial_ndims, got $dim"))
    return nothing
end

"""
    left_face_range(halo, dim)

Return the lower-side boundary face cells in dimension `dim`.

These are ghost cells. In a face loop, pair each index `IL` with
`IR = IL + face_offset(halo, dim)` to visit the `ghost | owned` face.
"""
function left_face_range(halo, dim::Int)
    _check_face_dim(halo, dim)
    ranges = _loop_interior_range(halo)
    return _dim_slab_range(ranges, dim, (first(ranges[dim]) - 1):(first(ranges[dim]) - 1))
end

left_face_range(halo, ::Dim{D}) where {D} = left_face_range(halo, D)

"""
    internal_face_range(halo, dim)

Return the internal faces for a sweep along `dim`: the owned cells that have an
owned `+dim` neighbour (only `dim` is trimmed by one), while every *transverse*
dimension keeps its full owned extent. This is what a per-direction conservative
flux update needs — it does not drop the last transverse row/column, so the
boundary-face fluxes cancel correctly there. Pair each index `IL` with
`IR = IL + face_offset(halo, dim)`.
"""
function internal_face_range(halo, dim::Int)
    ranges = _loop_interior_range(halo)
    return ntuple(_loop_ndims(halo)) do d
        d == dim ? (first(ranges[d]):(last(ranges[d]) - 1)) : (first(ranges[d]):last(ranges[d]))
    end
end

"""
    right_face_range(halo, dim)

Return the upper-side boundary face cells in dimension `dim`.

These are owned cells adjacent to the upper ghost side. In a face loop, pair
each index `IL` with `IR = IL + face_offset(halo, dim)` to visit the
`owned | ghost` face.
"""
function right_face_range(halo, dim::Int)
    _check_face_dim(halo, dim)
    ranges = _loop_interior_range(halo)
    return _dim_slab_range(ranges, dim, last(ranges[dim]):last(ranges[dim]))
end

right_face_range(halo, ::Dim{D}) where {D} = right_face_range(halo, D)

"""
    FaceRanges(halo)

Precompute Cartesian index ranges and offsets for face loops.

The stored indices always identify the lower-index cell of a face. Add
`get_unit_vector(ranges, dim)` to get the upper-index cell.

For `MultiHaloArray` and `ArrayOfHaloArray`, the ranges are spatial only. Apply
them after selecting an individual field.

- `get_left_face(ranges, dim)`: lower-side ghost cells.
- `get_internal_face(ranges, dim)`: internal owned faces along `dim` (transverse-full).
- `get_right_face(ranges, dim)`: upper-side owned cells.

Minimal owned-cell update:

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

for IL in get_internal_face(ranges,dim)
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
"""
struct FaceRanges{A,Bd,C,D,Halo}
    left_face::A
    internal_face_dirs::Bd    # per-direction internal faces (transverse-full) for flux sweeps
    right_face::C
    unit_vector::D
    halo::Halo
end

function FaceRanges(halo)
    spatial_ndims = _loop_ndims(halo)
    return FaceRanges(
        ntuple(d -> CartesianIndices(left_face_range(halo, d)), spatial_ndims),
        ntuple(d -> CartesianIndices(internal_face_range(halo, d)), spatial_ndims),
        ntuple(d -> CartesianIndices(right_face_range(halo, d)), spatial_ndims),
        ntuple(d -> face_offset(halo, d), Val(spatial_ndims)),
        halo_width(halo),
    )
end

get_left_face(ranges::FaceRanges) = ranges.left_face
get_left_face(ranges::FaceRanges, dim::Int) = ranges.left_face[dim]
get_left_face(ranges::FaceRanges, ::Dim{D}) where {D} = get_left_face(ranges, D)
"""
    get_internal_face(ranges, dim)

Internal faces for a conservative sweep along `dim` (transverse dimensions kept
full): the owned cells with an owned `+dim` neighbour. Pair each `IL` with
`IL + get_unit_vector(ranges, dim)`.
"""
get_internal_face(ranges::FaceRanges, dim::Int) = ranges.internal_face_dirs[dim]
get_internal_face(ranges::FaceRanges, ::Dim{D}) where {D} = get_internal_face(ranges, D)
get_right_face(ranges::FaceRanges) = ranges.right_face
get_right_face(ranges::FaceRanges, dim::Int) = ranges.right_face[dim]
get_right_face(ranges::FaceRanges, ::Dim{D}) where {D} = get_right_face(ranges, D)
get_unit_vector(ranges::FaceRanges) = ranges.unit_vector
get_unit_vector(ranges::FaceRanges, dim::Int) = ranges.unit_vector[dim]
get_unit_vector(ranges::FaceRanges, ::Dim{D}) where {D} = get_unit_vector(ranges, D)

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
and scattered with opposite signs onto the owned cells — `du[IL] -= scale*F` and
`du[IR] += scale*F` — skipping whichever side is a ghost at a physical boundary.

- `flux(UL, UR) -> F`: numerical flux between the lower (`UL`) and upper (`UR`)
  cell states.
- `read(data, I) -> U`: cell state at storage index `I`. Defaults to `data[I]`
  for scalar fields; pass e.g. a `conserved_cell`-style gather to build a
  multi-field `SVector`.
- `scatter!(data, I, s, F)`: add `s*F` into cell `I`. Defaults to
  `data[I] += s*F`; pass a multi-field writer for collections.

`du` and `u` are the raw storage the accessors understand — `parent(field)` for
a single field, or the `parent(state)` NamedTuple for a multi-field collection —
not the halo arrays themselves. `dim` may be an `Int` or a `Dim{D}`.

```julia
ranges = FaceRanges(u)
accumulate_flux_divergence!(parent(du), parent(u), ranges, 1, inv(dx), numerical_flux)
```
"""
function accumulate_flux_divergence!(du, u, ranges::FaceRanges, dim, scale,
        flux, read, scatter!)
    e = get_unit_vector(ranges, dim)

    @inbounds for IL in get_left_face(ranges, dim)
        IR = IL + e
        scatter!(du, IR, scale, flux(read(u, IL), read(u, IR)))
    end

    @inbounds for IL in get_internal_face(ranges, dim)
        IR = IL + e
        F = flux(read(u, IL), read(u, IR))
        scatter!(du, IL, -scale, F)
        scatter!(du, IR,  scale, F)
    end

    @inbounds for IL in get_right_face(ranges, dim)
        IR = IL + e
        scatter!(du, IL, -scale, flux(read(u, IL), read(u, IR)))
    end

    return du
end

@inline accumulate_flux_divergence!(du, u, ranges::FaceRanges, dim, scale, flux) =
    accumulate_flux_divergence!(du, u, ranges, dim, scale, flux,
        _scalar_face_read, _scalar_face_scatter!)

# Shared by face- and cell-range/region loops (defined here as face_ranges.jl is
# included first). _loop_ndims / _loop_interior_range dispatch the spatial
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

"""
    get_colored_left_face(ranges, dim, color)

Return the lower-side face cells of one race-free face color.
"""
get_colored_left_face(ranges::FaceRanges, dim::Int, color::Integer) =
    _colored_face(get_left_face(ranges, dim), dim, color)
get_colored_left_face(ranges::FaceRanges, ::Dim{D}, color::Integer) where {D} =
    get_colored_left_face(ranges, D, color)

"""
    get_colored_internal_face(ranges, dim, color)

Return the internal face cells of one race-free face color.
"""
function get_colored_internal_face(ranges::FaceRanges, dim::Int, color::Integer)
    return _colored_face(get_internal_face(ranges, dim), dim, color)
end
get_colored_internal_face(ranges::FaceRanges, ::Dim{D}, color::Integer) where {D} =
    get_colored_internal_face(ranges, D, color)

"""
    get_colored_right_face(ranges, dim, color)

Return the upper-side face cells of one race-free face color.
"""
get_colored_right_face(ranges::FaceRanges, dim::Int, color::Integer) =
    _colored_face(get_right_face(ranges, dim), dim, color)
get_colored_right_face(ranges::FaceRanges, ::Dim{D}, color::Integer) where {D} =
    get_colored_right_face(ranges, D, color)

"""
    face_offset(halo, dim)

Return the `CartesianIndex` offset from the lower-index cell to the upper-index
cell across a face in dimension `dim`.
"""
@inline function face_offset(halo, dim::Int)
    _check_face_dim(halo, dim)
    return CartesianIndex(versors(Val(_loop_ndims(halo)))[dim])
end

@inline face_offset(halo, ::Dim{D}) where {D} = face_offset(halo, D)
