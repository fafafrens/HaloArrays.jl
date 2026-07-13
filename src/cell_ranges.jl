# _spatial_* geometry helpers live in abstract_haloarray.jl;
# _loop_strided_range / _check_loop_color are shared with face_ranges.jl
# (defined there, included first).

"""
    CellRanges(halo)

Precompute Cartesian index ranges for interior-cell loops.

For `MultiHaloArray` and `ArrayOfHaloArray`, the ranges are spatial only. Apply
them after selecting an individual field.

- `interior_cells(ranges)`: all interior cells.
- `interior_cells(ranges, color)`: branch-free checkerboard
  subranges for nearest-neighbor in-place updates.
"""
struct CellRanges{A,Halo}
    owned_cells::A
    halo::Halo
    parity_offset::Int   # storage↔global sum-parity offset for this tile/rank
end

# A cell's color is its GLOBAL parity `mod(sum(global_index), 2)`, kept consistent
# across tile/rank seams by `parity_offset` — the one bit by which this tile's
# storage parity differs from global. Pass the tile id on a `ThreadedHaloArray`;
# single-block backends have one tile.
CellRanges(halo) = CellRanges(halo, 1)

function CellRanges(halo, tile_id::Integer)
    sr = _spatial_interior_range(halo)
    indices = CartesianIndices(sr)
    # `_geometry_field` picks the reference field (a collection's fields share
    # geometry), so the origin works for single arrays and collections.
    origin = interior_to_global_index(_geometry_field(halo), tile_id, map(_ -> 1, sr))
    parity_offset = mod(sum(origin) - sum(Tuple(first(indices))), 2)
    return CellRanges(indices, halo_width(halo), parity_offset)
end

"""
    interior_cells(ranges)

Return the interior-cell Cartesian index range.
"""
interior_cells(ranges::CellRanges) = ranges.owned_cells

@inline function _cell_color_mask(::Val{N}, color::Int, subrange_id::Int) where {N}
    prefix_bits = ntuple(d -> d == N ? 0 : ((subrange_id - 1) >>> (d - 1)) & 1, Val(N))
    last_bit = mod(color - sum(prefix_bits), 2)
    return ntuple(d -> d == N ? last_bit : prefix_bits[d], Val(N))
end

# Branch-free checkerboard in STORAGE coordinates; the caller shifts the color by
# `parity_offset` (see `interior_cells`) so the result lands on global parity.
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
    interior_cells(ranges, color)

Return a tuple of strided `CartesianIndices` that cover one checkerboard color
of the interior cells. Colors are `0` and `1`, chosen by the cell's **global**
parity `mod(sum(global_index), 2)`, so the coloring is consistent across
tile/rank boundaries. On a [`ThreadedHaloArray`](@ref) build the ranges per
tile — `CellRanges(u, tile_id)` — so each tile carries its own parity offset.
"""
function interior_cells(ranges::CellRanges, color::Integer)
    # shift the requested color by this tile's storage↔global parity offset
    storage_color = mod(_check_loop_color(color) - ranges.parity_offset, 2)
    return _colored_cell_ranges(interior_cells(ranges), storage_color)
end
