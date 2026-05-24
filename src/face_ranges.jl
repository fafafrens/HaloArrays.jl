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

Return the lower ghost-cell slab adjacent to owned cells in dimension `dim`.
For a face loop this is the lower-index cell range of the left boundary faces
`ghost | owned`.
"""
function left_face_range(halo, dim::Int)
    _check_face_dim(halo, dim)
    ranges = _face_spatial_interior_range(halo)
    return _dim_slab_range(ranges, dim, (first(ranges[dim]) - 1):(first(ranges[dim]) - 1))
end

left_face_range(halo, ::Dim{D}) where {D} = left_face_range(halo, D)

"""
    internal_face_range(halo)

Return the lower-index owned-cell range for the internal cells that have an
owned neighbor in the positive direction of every dimension.
"""
function internal_face_range(halo)
    ranges = _face_spatial_interior_range(halo)
    return ntuple(d -> first(ranges[d]):(last(ranges[d]) - 1), Val(_face_spatial_ndims(halo)))
end

"""
    right_face_range(halo, dim)

Return the owned-cell slab adjacent to the upper ghost side in dimension `dim`.
For a face loop this is the lower-index cell range of the right boundary faces
`owned | ghost`.
"""
function right_face_range(halo, dim::Int)
    _check_face_dim(halo, dim)
    ranges = _face_spatial_interior_range(halo)
    return _dim_slab_range(ranges, dim, last(ranges[dim]):last(ranges[dim]))
end

right_face_range(halo, ::Dim{D}) where {D} = right_face_range(halo, D)

"""
    FaceRanges(halo)

Return Cartesian index ranges for face loops in all spatial dimensions.

Each left/right range contains the lower/left cell of the face for a given
dimension. The matching upper/right cell is obtained by adding
`get_unit_vector(ranges, dim)`.
For `MultiHaloArray` and `ArrayOfHaloArray`, the ranges and offset describe the
spatial indices of each field; decompose the collection and apply them to the
field arrays.

- `get_left_face(ranges, dim)`: lower ghost cells for `ghost | owned` faces.
- `get_internal_face(ranges)`: lower-index owned cells of internal faces.
- `get_right_face(ranges, dim)`: upper owned cells for `owned | ghost` faces.

Example owned-cell-only face loop:

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


@inline function shift_range(r::AbstractUnitRange, s::Integer)
    return (first(r) + s):(last(r) + s)
end
@inline function shift_ranges(ranges::NTuple{N,Any}, s::Integer) where {N}
    return ntuple(d -> shift_range(ranges[d], s), Val(N))
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
