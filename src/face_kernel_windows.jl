"""
    FaceWindow

Compact face-region metadata for launch-style loops. `first` and `size`
describe the lower-index cell region; `offset` maps each lower cell to the upper
cell. The kernel scatters each face's flux onto both cells.
"""
struct FaceWindow{N}
    first::CartesianIndex{N}
    size::NTuple{N,Int}
    offset::CartesianIndex{N}
end

"""
    FaceCheckerboard

Compact metadata for race-free colored face loops. `first`, `size`, and
`stride` describe the lower-index cell region for one face color; `offset` maps
each lower cell to the upper cell.
"""
struct FaceCheckerboard{N}
    first::CartesianIndex{N}
    size::NTuple{N,Int}
    stride::CartesianIndex{N}
    offset::CartesianIndex{N}
end

@inline FaceWindow(indices::CartesianIndices{N}, offset::CartesianIndex{N}) where {N} =
    FaceWindow(first(indices), size(indices), offset)

@inline function FaceCheckerboard(indices::CartesianIndices{N},
                                  offset::CartesianIndex{N}) where {N}
    return FaceCheckerboard(
        first(indices),
        size(indices),
        CartesianIndex(ntuple(d -> step(indices.indices[d]), Val(N))),
        offset,
    )
end

"""
    interior_face_window(ranges, dim)
    interior_face_window(ranges, dim, color)

Launch metadata ([`FaceWindow`](@ref)) for every face touching the interior along
`dim` — the kernel reconstructs each face's lower cell with [`cell_index`](@ref),
the upper with `+ region.offset`, and scatters the flux onto both. With a `color`
(`0`/`1`) argument it returns a [`FaceCheckerboard`](@ref): one race-free color, so
the parallel scatter needs no atomics.
"""
@inline interior_face_window(ranges::FaceRanges, dim::Int) =
    FaceWindow(interior_faces(ranges, dim), unit_vector(ranges, dim))
@inline interior_face_window(ranges::FaceRanges, ::Dim{D}) where {D} =
    interior_face_window(ranges, D)
@inline interior_face_window(ranges::FaceRanges, dim::Int, color::Integer) =
    FaceCheckerboard(interior_faces(ranges, dim, color), unit_vector(ranges, dim))
@inline interior_face_window(ranges::FaceRanges, ::Dim{D}, color::Integer) where {D} =
    interior_face_window(ranges, D, color)

# ------------------------------------------------------------------------------
# Launch-index → storage-index mapping, mirroring the cell-region API so kernels
# use one function for every region type.
# ------------------------------------------------------------------------------

"""
    cell_index(region::FaceWindow, J)
    cell_index(region::FaceCheckerboard, J)

Map a launch-local index `J` to the **lower** cell `IL` of the face (in storage
coordinates); the upper cell is `IL + region.offset`. The colored variant
applies the region's stride.
"""
@inline function cell_index(region::FaceWindow{N},
                            J::NTuple{N,<:Integer}) where {N}
    first_tuple = Tuple(region.first)
    return ntuple(d -> first_tuple[d] + Int(J[d]) - 1, Val(N))
end

@inline function cell_index(region::FaceCheckerboard{N},
                            J::NTuple{N,<:Integer}) where {N}
    first_tuple  = Tuple(region.first)
    stride_tuple = Tuple(region.stride)
    return ntuple(d -> first_tuple[d] + (Int(J[d]) - 1) * stride_tuple[d], Val(N))
end

@inline cell_index(region::Union{FaceWindow{N},FaceCheckerboard{N}},
                   J::CartesianIndex{N}) where {N} =
    CartesianIndex(cell_index(region, Tuple(J)))

@inline _face_region_stride(::FaceWindow{N}) where {N} = ntuple(_ -> 1, Val(N))
@inline _face_region_stride(region::FaceCheckerboard) = Tuple(region.stride)

"""
    is_cell_index_inbounds(region::Union{FaceWindow,FaceCheckerboard}, I)

Whether a storage-space lower cell `I` lies inside the face region (face
launches are exact, so this is only a guard against launch overshoot).
"""
@inline function is_cell_index_inbounds(
        region::Union{FaceWindow{N},FaceCheckerboard{N}},
        I::NTuple{N,<:Integer}) where {N}
    first_tuple  = Tuple(region.first)
    stride_tuple = _face_region_stride(region)
    checks = ntuple(Val(N)) do d
        first_tuple[d] <= Int(I[d]) <= first_tuple[d] + (region.size[d] - 1) * stride_tuple[d]
    end
    return all(checks)
end

@inline is_cell_index_inbounds(
        region::Union{FaceWindow{N},FaceCheckerboard{N}},
        I::CartesianIndex{N}) where {N} =
    is_cell_index_inbounds(region, Tuple(I))
