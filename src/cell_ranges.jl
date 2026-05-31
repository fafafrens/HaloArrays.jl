@inline _cell_spatial_interior_range(halo::AbstractSingleHaloArray) = interior_range(halo)
@inline _cell_spatial_interior_range(halo::MultiHaloArray) =
    interior_range(first(values(halo.arrays)))
@inline _cell_spatial_interior_range(halo::ArrayOfHaloArray) =
    interior_range(first(parent(halo)))
@inline _cell_spatial_interior_range(arr::AbstractArray{<:AbstractSingleHaloArray}) =
    interior_range(first(arr))

@inline function _check_cell_color(color::Integer)
    (color == 0 || color == 1) ||
        throw(ArgumentError("cell color must be 0 or 1, got $color"))
    return Int(color)
end

@inline _cell_range_from_first_size_stride(first_index::Int, len::Int, stride::Int) =
    first_index:stride:(first_index + stride * (len - 1))

"""
    CellRanges(halo)

Precompute Cartesian index ranges for owned-cell loops.

For `MultiHaloArray` and `ArrayOfHaloArray`, the ranges are spatial only. Apply
them after selecting an individual field.

- `get_owned_cells(ranges)`: all owned cells.
- `get_colored_owned_cell_ranges(ranges, color)`: branch-free checkerboard
  subranges for nearest-neighbor in-place updates.
"""
struct CellRanges{A,Halo}
    owned_cells::A
    halo::Halo
end

function CellRanges(halo)
    return CellRanges(
        CartesianIndices(_cell_spatial_interior_range(halo)),
        halo_width(halo),
    )
end

"""
    get_owned_cells(ranges)

Return the owned-cell Cartesian index range.
"""
get_owned_cells(ranges::CellRanges) = ranges.owned_cells

@inline function _cell_color_mask(::Val{N}, color::Int, subrange_id::Int) where {N}
    prefix_bits = ntuple(d -> d == N ? 0 : ((subrange_id - 1) >>> (d - 1)) & 1, Val(N))
    last_bit = mod(color - sum(prefix_bits), 2)
    return ntuple(d -> d == N ? last_bit : prefix_bits[d], Val(N))
end

@inline function _colored_cell_subrange(indices::CartesianIndices{N},
                                        mask::NTuple{N,Int}) where {N}
    first_tuple = Tuple(first(indices))
    region_size = size(indices)

    return CartesianIndices(ntuple(Val(N)) do d
        delta = mod(mask[d] - mod(first_tuple[d], 2), 2)
        len = region_size[d] <= delta ? 0 : cld(region_size[d] - delta, 2)
        start = first_tuple[d] + delta
        _cell_range_from_first_size_stride(start, len, 2)
    end)
end

@inline function _colored_cell_ranges(indices::CartesianIndices{N},
                                      color::Integer) where {N}
    checked_color = _check_cell_color(color)
    return ntuple(Val(2^(N - 1))) do subrange_id
        mask = _cell_color_mask(Val(N), checked_color, subrange_id)
        _colored_cell_subrange(indices, mask)
    end
end

"""
    get_colored_owned_cell_ranges(ranges, color)

Return a tuple of strided `CartesianIndices` that cover one checkerboard color
of the owned cells. Colors are `0` and `1`, chosen by
`mod(sum(Tuple(I)), 2)`.
"""
get_colored_owned_cell_ranges(ranges::CellRanges, color::Integer) =
    _colored_cell_ranges(get_owned_cells(ranges), color)
