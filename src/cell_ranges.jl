# _spatial_* geometry helpers live in abstract_haloarray.jl;
# _loop_strided_range / _check_loop_color are shared with face_ranges.jl
# (defined there, included first).

"""
    CellRanges(halo)

Precompute Cartesian index ranges for interior-cell loops.

For `MultiHaloArray` and `ArrayOfHaloArray`, the ranges are spatial only. Apply
them after selecting an individual field.

- `get_interior_cells(ranges)`: all interior cells.
- `get_colored_interior_cell_ranges(ranges, color)`: branch-free checkerboard
  subranges for nearest-neighbor in-place updates.
"""
struct CellRanges{A,Halo}
    owned_cells::A
    halo::Halo
end

function CellRanges(halo)
    return CellRanges(
        CartesianIndices(_spatial_interior_range(halo)),
        halo_width(halo),
    )
end

"""
    get_interior_cells(ranges)

Return the interior-cell Cartesian index range.
"""
get_interior_cells(ranges::CellRanges) = ranges.owned_cells

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
        _loop_strided_range(start, len, 2)
    end)
end

@inline function _colored_cell_ranges(indices::CartesianIndices{N},
                                      color::Integer) where {N}
    checked_color = _check_loop_color(color)
    return ntuple(Val(2^(N - 1))) do subrange_id
        mask = _cell_color_mask(Val(N), checked_color, subrange_id)
        _colored_cell_subrange(indices, mask)
    end
end

"""
    get_colored_interior_cell_ranges(ranges, color)

Return a tuple of strided `CartesianIndices` that cover one checkerboard color
of the interior cells. Colors are `0` and `1`, chosen by
`mod(sum(Tuple(I)), 2)`.
"""
get_colored_interior_cell_ranges(ranges::CellRanges, color::Integer) =
    _colored_cell_ranges(get_interior_cells(ranges), color)
