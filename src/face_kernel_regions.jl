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

@inline function FaceKernelRegion(indices::CartesianIndices{N},
                                     offset::CartesianIndex{N},
                                     lower_owned::Bool,
                                     upper_owned::Bool) where {N}
    return FaceKernelRegion(first(indices), size(indices), offset, lower_owned, upper_owned)
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
    FaceKernelRegion(get_internal_face(ranges), get_unit_vector(ranges, dim), true, true)
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
