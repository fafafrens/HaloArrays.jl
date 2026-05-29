@inline function _dim_slab_range(ranges::NTuple{N,Any}, dim::Int, range) where {N}
    return ntuple(d -> d == dim ? range : ranges[d], Val(N))
end

@inline _face_spatial_ndims(halo::AbstractSingleHaloArray) = ndims(halo)
@inline _face_spatial_ndims(::MultiHaloArray{T,N}) where {T,N} = N
@inline _face_spatial_ndims(::ArrayOfHaloArray{T,N}) where {T,N} = N

@inline _face_spatial_interior_range(halo::AbstractSingleHaloArray) = interior_range(halo)
@inline _face_spatial_interior_range(halo::MultiHaloArray) =
    interior_range(first(values(halo.arrays)))
@inline _face_spatial_interior_range(halo::ArrayOfHaloArray) =
    interior_range(first(parent(halo)))

@inline function _check_face_dim(halo, dim::Int)
    spatial_ndims = _face_spatial_ndims(halo)
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
    ranges = _face_spatial_interior_range(halo)
    return _dim_slab_range(ranges, dim, (first(ranges[dim]) - 1):(first(ranges[dim]) - 1))
end

left_face_range(halo, ::Dim{D}) where {D} = left_face_range(halo, D)

"""
    internal_face_range(halo)

Return the dimension-independent owned-cell core used by face loops.

Each returned cell has an owned positive neighbor in every spatial dimension.
For a chosen dimension, pair each index `IL` with
`IR = IL + face_offset(halo, dim)`.
"""
function internal_face_range(halo)
    ranges = _face_spatial_interior_range(halo)
    return ntuple(d -> first(ranges[d]):(last(ranges[d]) - 1), Val(_face_spatial_ndims(halo)))
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
    ranges = _face_spatial_interior_range(halo)
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
- `get_internal_face(ranges)`: dimension-independent owned-cell core.
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
"""
struct FaceRanges{A,B,C,D,Halo}
    left_face::A
    internal_face::B
    right_face::C
    unit_vector::D
    halo::Halo
end

function FaceRanges(halo)
    spatial_ndims = _face_spatial_ndims(halo)
    return FaceRanges(
        ntuple(d -> CartesianIndices(left_face_range(halo, d)), spatial_ndims),
        CartesianIndices(internal_face_range(halo)),
        ntuple(d -> CartesianIndices(right_face_range(halo, d)), spatial_ndims),
        ntuple(d -> face_offset(halo, d), Val(spatial_ndims)),
        halo_width(halo),
    )
end

get_left_face(ranges::FaceRanges) = ranges.left_face
get_left_face(ranges::FaceRanges, dim::Int) = ranges.left_face[dim]
get_left_face(ranges::FaceRanges, ::Dim{D}) where {D} = get_left_face(ranges, D)
get_internal_face(ranges::FaceRanges) = ranges.internal_face
get_right_face(ranges::FaceRanges) = ranges.right_face
get_right_face(ranges::FaceRanges, dim::Int) = ranges.right_face[dim]
get_right_face(ranges::FaceRanges, ::Dim{D}) where {D} = get_right_face(ranges, D)
get_unit_vector(ranges::FaceRanges) = ranges.unit_vector
get_unit_vector(ranges::FaceRanges, dim::Int) = ranges.unit_vector[dim]
get_unit_vector(ranges::FaceRanges, ::Dim{D}) where {D} = get_unit_vector(ranges, D)

@inline function _check_face_color(color::Integer)
    (color == 0 || color == 1) ||
        throw(ArgumentError("face color must be 0 or 1, got $color"))
    return Int(color)
end

@inline _range_from_first_size_stride(first_index::Int, len::Int, stride::Int) =
    first_index:stride:(first_index + stride * (len - 1))

@inline function _colored_face(indices::CartesianIndices{N}, dim::Int, color::Integer) where {N}
    checked_color = _check_face_color(color)
    first_tuple = Tuple(first(indices))
    region_size = size(indices)
    delta = mod(checked_color - mod(first_tuple[dim], 2), 2)
    colored_size_dim = region_size[dim] <= delta ? 0 : cld(region_size[dim] - delta, 2)

    return CartesianIndices(ntuple(Val(N)) do d
        len = d == dim ? colored_size_dim : region_size[d]
        stride = d == dim ? 2 : 1
        start = first_tuple[d] + (d == dim ? delta : 0)
        _range_from_first_size_stride(start, len, stride)
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
    get_unit_vector(ranges, dim)
    return _colored_face(get_internal_face(ranges), dim, color)
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
    return CartesianIndex(versors(Val(_face_spatial_ndims(halo)))[dim])
end

@inline face_offset(halo, ::Dim{D}) where {D} = face_offset(halo, D)
