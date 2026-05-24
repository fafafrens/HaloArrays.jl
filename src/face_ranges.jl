@inline function _dim_slab_range(ranges::NTuple{N,Any}, dim::Int, range) where {N}
    return ntuple(d -> d == dim ? range : ranges[d], Val(N))
end

@inline function _check_face_dim(halo, dim::Int)
    1 <= dim <= ndims(halo) || throw(ArgumentError("dim must be in 1:$(ndims(halo)), got $dim"))
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
    ranges = interior_range(halo)
    return _dim_slab_range(ranges, dim, (first(ranges[dim]) - 1):(first(ranges[dim]) - 1))
end

left_face_range(halo, ::Dim{D}) where {D} = left_face_range(halo, D)

"""
    internal_face_range(halo)

Return the lower-index owned-cell range for the internal cells that have an
owned neighbor in the positive direction of every dimension.
"""
function internal_face_range(halo)
    ranges = interior_range(halo)
    return ntuple(d -> first(ranges[d]):(last(ranges[d]) - 1), Val(ndims(halo)))
end

"""
    right_face_range(halo, dim)

Return the owned-cell slab adjacent to the upper ghost side in dimension `dim`.
For a face loop this is the lower-index cell range of the right boundary faces
`owned | ghost`.
"""
function right_face_range(halo, dim::Int)
    _check_face_dim(halo, dim)
    ranges = interior_range(halo)
    return _dim_slab_range(ranges, dim, last(ranges[dim]):last(ranges[dim]))
end

right_face_range(halo, ::Dim{D}) where {D} = right_face_range(halo, D)

"""
    FaceRanges(halo, dim)

Return Cartesian index ranges for a face loop in dimension `dim`.

Each range contains the lower/left cell of the face. The matching upper/right
cell is obtained by adding `face_offset(halo, dim)`.

- `get_left_face(ranges)`: lower ghost cells for `ghost | owned` faces.
- `get_internal_face(ranges)`: lower-index owned cells of internal faces.
- `get_right_face(ranges)`: upper owned cells for `owned | ghost` faces.

Example owned-cell-only face loop:

```julia
ranges = FaceRanges(u, dim)
e = get_unit_vector(ranges)
udata = parent(u)
dudata = parent(du)

for IL in get_left_face(ranges)
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

for IL in get_right_face(ranges)
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


function FaceRanges(halo, dim::Int)
    return FaceRanges(
        CartesianIndices(left_face_range(halo, dim)),
        CartesianIndices(internal_face_range(halo)),
        CartesianIndices(right_face_range(halo, dim)),
        face_offset(halo, dim),
        halo_width(halo),
    )
end

FaceRanges(halo, ::Dim{D}) where {D} = FaceRanges(halo, D)

get_left_face(ranges::FaceRanges) = ranges.left_face
get_internal_face(ranges::FaceRanges) = ranges.internal_face
get_right_face(ranges::FaceRanges) = ranges.right_face
get_unit_vector(ranges::FaceRanges) = ranges.unit_vector

"""
    face_offset(halo, dim)

Return the `CartesianIndex` offset from the lower-index cell to the upper-index
cell across a face in dimension `dim`.
"""
@inline face_offset(halo, dim::Int) = CartesianIndex(versors(halo)[dim])
@inline face_offset(halo, ::Dim{D}) where {D} = face_offset(halo, D)
