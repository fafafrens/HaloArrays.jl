@inline function _dim_slab_range(ranges::NTuple{N,Any}, dim::Int, range) where {N}
    return ntuple(d -> d == dim ? range : ranges[d], Val(N))
end

@inline function _check_face_dim(halo, dim::Int)
    1 <= dim <= ndims(halo) || throw(ArgumentError("dim must be in 1:$(ndims(halo)), got $dim"))
    return nothing
end

function _check_owned_face_update_compatible(du, u, dim::Int)
    ndims(du) == ndims(u) || throw(DimensionMismatch("du and u must have the same number of dimensions"))
    local_size(du) == local_size(u) || throw(DimensionMismatch("du and u must have the same local interior size"))
    halo_width(du) == halo_width(u) || throw(DimensionMismatch("du and u must have the same halo width"))
    _check_face_dim(u, dim)
    return nothing
end

"""
    lower_owned_face_range(halo, dim)

Return the owned-cell slab adjacent to the lower ghost side in dimension `dim`.
For a face loop this is the right/upper owned cell range of `ghost | owned`
faces.
"""
function lower_owned_face_range(halo, dim::Int)
    _check_face_dim(halo, dim)
    ranges = interior_range(halo)
    return _dim_slab_range(ranges, dim, first(ranges[dim]):first(ranges[dim]))
end

lower_owned_face_range(halo, ::Dim{D}) where {D} = lower_owned_face_range(halo, D)

"""
    internal_owned_face_left_range(halo, dim)

Return the lower-index owned-cell range for all internal `owned | owned` faces
in dimension `dim`. The matching right/upper cells are obtained by adding
`face_offset(halo, dim)`.
"""
function internal_owned_face_left_range(halo, dim::Int)
    _check_face_dim(halo, dim)
    ranges = interior_range(halo)
    return _dim_slab_range(ranges, dim, first(ranges[dim]):(last(ranges[dim]) - 1))
end

internal_owned_face_left_range(halo, ::Dim{D}) where {D} = internal_owned_face_left_range(halo, D)

"""
    upper_owned_face_range(halo, dim)

Return the owned-cell slab adjacent to the upper ghost side in dimension `dim`.
For a face loop this is the left/lower owned cell range of `owned | ghost`
faces.
"""
function upper_owned_face_range(halo, dim::Int)
    _check_face_dim(halo, dim)
    ranges = interior_range(halo)
    return _dim_slab_range(ranges, dim, last(ranges[dim]):last(ranges[dim]))
end

upper_owned_face_range(halo, ::Dim{D}) where {D} = upper_owned_face_range(halo, D)

"""
    owned_face_ranges(halo, dim)

Return named ranges for an owned-cell-only face update in dimension `dim`:

- `lower_owned`: owned cells adjacent to lower ghosts.
- `internal_left`: lower-index owned cells of internal faces.
- `upper_owned`: owned cells adjacent to upper ghosts.

Example owned-cell-only face loop:

```julia
ranges = owned_face_ranges(u, dim)
e = face_offset(u, dim)
udata = parent(u)
dudata = parent(du)

for IR in CartesianIndices(ranges.lower_owned)
    IL = IR - e
    flux = numerical_flux(udata[IL], udata[IR])
    dudata[IR] += flux
end

for IL in CartesianIndices(ranges.internal_left)
    IR = IL + e
    flux = numerical_flux(udata[IL], udata[IR])
    dudata[IL] -= flux
    dudata[IR] += flux
end

for IL in CartesianIndices(ranges.upper_owned)
    IR = IL + e
    flux = numerical_flux(udata[IL], udata[IR])
    dudata[IL] -= flux
end
```
"""
function owned_face_ranges(halo, dim::Int)
    return (
        lower_owned=lower_owned_face_range(halo, dim),
        internal_left=internal_owned_face_left_range(halo, dim),
        upper_owned=upper_owned_face_range(halo, dim),
    )
end

owned_face_ranges(halo, ::Dim{D}) where {D} = owned_face_ranges(halo, D)

"""
    face_offset(halo, dim)

Return the `CartesianIndex` offset from the lower-index cell to the upper-index
cell across a face in dimension `dim`.
"""
@inline face_offset(halo, dim::Int) = CartesianIndex(versors(halo)[dim])
@inline face_offset(halo, ::Dim{D}) where {D} = face_offset(halo, D)

"""
    foreach_owned_face!(f, du, u, dim)

Apply a conservative owned-cell-only face update in dimension `dim`.

`f(u_left, u_right)` must return the oriented flux from the lower-index cell to
the upper-index cell. The routine updates only cells owned by the local array:
lower boundary faces update the upper owned cell, internal faces update both
owned cells, and upper boundary faces update the lower owned cell.

This function does not clear `du` and does not synchronize `u`; do those steps
outside the face loop.
"""
function foreach_owned_face!(f, du, u, dim::Int)
    _check_owned_face_update_compatible(du, u, dim)

    ranges = owned_face_ranges(u, dim)
    offset = face_offset(u, dim)
    udata = parent(u)
    dudata = parent(du)

    @inbounds for IR in CartesianIndices(ranges.lower_owned)
        IL = IR - offset
        dudata[IR] += f(udata[IL], udata[IR])
    end

    @inbounds for IL in CartesianIndices(ranges.internal_left)
        IR = IL + offset
        flux = f(udata[IL], udata[IR])
        dudata[IL] -= flux
        dudata[IR] += flux
    end

    @inbounds for IL in CartesianIndices(ranges.upper_owned)
        IR = IL + offset
        dudata[IL] -= f(udata[IL], udata[IR])
    end

    return du
end

foreach_owned_face!(f, du, u, ::Dim{D}) where {D} = foreach_owned_face!(f, du, u, D)
