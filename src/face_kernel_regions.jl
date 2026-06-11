"""
    FaceKernelRegion

Compact face-region metadata for launch-style loops. `first` and `size`
describe the lower-index cell region; `offset` maps each lower cell to the
upper cell. `lower_owned` and `upper_owned` mark which side contributes to the
owned update.
"""
struct FaceKernelRegion{N}
    first::CartesianIndex{N}
    size::NTuple{N,Int}
    offset::CartesianIndex{N}
    lower_owned::Bool
    upper_owned::Bool
end

"""
    ColoredFaceKernelRegion

Compact metadata for race-free colored face loops. `first`, `size`, and
`stride` describe the lower-index cell region for one face color; `offset` maps
each lower cell to the upper cell. `lower_owned` and `upper_owned` mark which
side contributes to the owned update.
"""
struct ColoredFaceKernelRegion{N}
    first::CartesianIndex{N}
    size::NTuple{N,Int}
    stride::CartesianIndex{N}
    offset::CartesianIndex{N}
    lower_owned::Bool
    upper_owned::Bool
end

@inline function FaceKernelRegion(indices::CartesianIndices{N},
                                     offset::CartesianIndex{N},
                                     lower_owned::Bool,
                                     upper_owned::Bool) where {N}
    return FaceKernelRegion(first(indices), size(indices), offset, lower_owned, upper_owned)
end

@inline function ColoredFaceKernelRegion(indices::CartesianIndices{N},
                                         offset::CartesianIndex{N},
                                         lower_owned::Bool,
                                         upper_owned::Bool) where {N}
    return ColoredFaceKernelRegion(
        first(indices),
        size(indices),
        CartesianIndex(ntuple(d -> step(indices.indices[d]), Val(N))),
        offset,
        lower_owned,
        upper_owned,
    )
end

"""
    get_left_face_region(ranges, dim)

Return compact launch metadata for the lower-side `ghost | owned` face.
"""
@inline get_left_face_region(ranges::FaceRanges, dim::Int) =
    FaceKernelRegion(get_left_face(ranges, dim), get_unit_vector(ranges, dim), false, true)
@inline get_left_face_region(ranges::FaceRanges, ::Dim{D}) where {D} =
    get_left_face_region(ranges, D)

"""
    get_internal_face_region(ranges, dim)

Return compact launch metadata for owned-cell internal faces.
"""
@inline get_internal_face_region(ranges::FaceRanges, dim::Int) =
    FaceKernelRegion(get_internal_face(ranges, dim), get_unit_vector(ranges, dim), true, true)
@inline get_internal_face_region(ranges::FaceRanges, ::Dim{D}) where {D} =
    get_internal_face_region(ranges, D)

"""
    get_right_face_region(ranges, dim)

Return compact launch metadata for the upper-side `owned | ghost` face.
"""
@inline get_right_face_region(ranges::FaceRanges, dim::Int) =
    FaceKernelRegion(get_right_face(ranges, dim), get_unit_vector(ranges, dim), true, false)
@inline get_right_face_region(ranges::FaceRanges, ::Dim{D}) where {D} =
    get_right_face_region(ranges, D)

"""
    get_colored_left_face_region(ranges, dim, color)

Return compact launch metadata for one lower-side face color.
"""
@inline get_colored_left_face_region(ranges::FaceRanges, dim::Int, color::Integer) =
    ColoredFaceKernelRegion(
        get_colored_left_face(ranges, dim, color),
        get_unit_vector(ranges, dim),
        false,
        true,
    )
@inline get_colored_left_face_region(ranges::FaceRanges, ::Dim{D}, color::Integer) where {D} =
    get_colored_left_face_region(ranges, D, color)

"""
    get_colored_internal_face_region(ranges, dim, color)

Return compact launch metadata for one internal face color.
"""
@inline get_colored_internal_face_region(ranges::FaceRanges, dim::Int, color::Integer) =
    ColoredFaceKernelRegion(
        get_colored_internal_face(ranges, dim, color),
        get_unit_vector(ranges, dim),
        true,
        true,
    )
@inline get_colored_internal_face_region(ranges::FaceRanges, ::Dim{D}, color::Integer) where {D} =
    get_colored_internal_face_region(ranges, D, color)

"""
    get_colored_right_face_region(ranges, dim, color)

Return compact launch metadata for one upper-side face color.
"""
@inline get_colored_right_face_region(ranges::FaceRanges, dim::Int, color::Integer) =
    ColoredFaceKernelRegion(
        get_colored_right_face(ranges, dim, color),
        get_unit_vector(ranges, dim),
        true,
        false,
    )
@inline get_colored_right_face_region(ranges::FaceRanges, ::Dim{D}, color::Integer) where {D} =
    get_colored_right_face_region(ranges, D, color)

# ------------------------------------------------------------------------------
# Launch-index → storage-index mapping, mirroring the cell-region API so kernels
# use one function for every region type.
# ------------------------------------------------------------------------------

"""
    cell_index(region::FaceKernelRegion, J)
    cell_index(region::ColoredFaceKernelRegion, J)

Map a launch-local index `J` to the **lower** cell `IL` of the face (in storage
coordinates); the upper cell is `IL + region.offset`. The colored variant
applies the region's stride.
"""
@inline function cell_index(region::FaceKernelRegion{N},
                            J::NTuple{N,<:Integer}) where {N}
    first_tuple = Tuple(region.first)
    return ntuple(d -> first_tuple[d] + Int(J[d]) - 1, Val(N))
end

@inline function cell_index(region::ColoredFaceKernelRegion{N},
                            J::NTuple{N,<:Integer}) where {N}
    first_tuple  = Tuple(region.first)
    stride_tuple = Tuple(region.stride)
    return ntuple(d -> first_tuple[d] + (Int(J[d]) - 1) * stride_tuple[d], Val(N))
end

@inline cell_index(region::Union{FaceKernelRegion{N},ColoredFaceKernelRegion{N}},
                   J::CartesianIndex{N}) where {N} =
    CartesianIndex(cell_index(region, Tuple(J)))

@inline _face_region_stride(::FaceKernelRegion{N}) where {N} = ntuple(_ -> 1, Val(N))
@inline _face_region_stride(region::ColoredFaceKernelRegion) = Tuple(region.stride)

"""
    is_cell_index_inbounds(region::Union{FaceKernelRegion,ColoredFaceKernelRegion}, I)

Whether a storage-space lower cell `I` lies inside the face region (face
launches are exact, so this is only a guard against launch overshoot).
"""
@inline function is_cell_index_inbounds(
        region::Union{FaceKernelRegion{N},ColoredFaceKernelRegion{N}},
        I::NTuple{N,<:Integer}) where {N}
    first_tuple  = Tuple(region.first)
    stride_tuple = _face_region_stride(region)
    checks = ntuple(Val(N)) do d
        first_tuple[d] <= Int(I[d]) <= first_tuple[d] + (region.size[d] - 1) * stride_tuple[d]
    end
    return all(checks)
end

@inline is_cell_index_inbounds(
        region::Union{FaceKernelRegion{N},ColoredFaceKernelRegion{N}},
        I::CartesianIndex{N}) where {N} =
    is_cell_index_inbounds(region, Tuple(I))
