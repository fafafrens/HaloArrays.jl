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
struct CellRanges{A,Halo,N}
    owned_cells::A
    halo::Halo
    global_origin::NTuple{N,Int}   # global index of the first interior cell (this tile/rank)
end

# `global_origin` anchors the checkerboard parity to the GLOBAL cell index, so
# the colors stay consistent across tile/rank seams (a storage-local anchor
# double-colors adjacent cells at a boundary with an odd local extent). On a
# `ThreadedHaloArray` pass the tile id; single-block backends have one tile.
CellRanges(halo) = CellRanges(halo, 1)

function CellRanges(halo, tile_id::Integer)
    sr = _spatial_interior_range(halo)
    # `_geometry_field` picks the reference field (a collection's fields share
    # geometry), so the global origin works for single arrays and collections.
    origin = interior_to_global_index(_geometry_field(halo), tile_id, map(_ -> 1, sr))
    return CellRanges(CartesianIndices(sr), halo_width(halo), origin)
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

# `origin` is the GLOBAL index of the first interior cell (`indices` are the
# tile/rank-LOCAL storage indices). The color of a cell is its *global* parity,
# so the first colored storage cell is offset by the global-origin parity, not
# the storage parity — this keeps the checkerboard continuous across seams.
@inline function _colored_cell_subrange(indices::CartesianIndices{N},
                                        origin::NTuple{N,Int}, mask::NTuple{N,Int}) where {N}
    first_tuple = Tuple(first(indices))
    region_size = size(indices)

    return CartesianIndices(ntuple(Val(N)) do d
        delta = mod(mask[d] - mod(origin[d], 2), 2)
        len = region_size[d] <= delta ? 0 : cld(region_size[d] - delta, 2)
        start = first_tuple[d] + delta
        _loop_strided_range(start, len, 2)
    end)
end

@inline function _colored_cell_ranges(indices::CartesianIndices{N},
                                      origin::NTuple{N,Int}, color::Integer) where {N}
    checked_color = _check_loop_color(color)
    return ntuple(Val(2^(N - 1))) do subrange_id
        mask = _cell_color_mask(Val(N), checked_color, subrange_id)
        _colored_cell_subrange(indices, origin, mask)
    end
end

"""
    interior_cells(ranges, color)

Return a tuple of strided `CartesianIndices` that cover one checkerboard color
of the interior cells. Colors are `0` and `1`, chosen by the cell's **global**
parity `mod(sum(global_index), 2)`, so the coloring is consistent across
tile/rank boundaries. On a [`ThreadedHaloArray`](@ref) build the ranges per
tile — `CellRanges(u, tile_id)` — so each tile carries its own global origin.
"""
interior_cells(ranges::CellRanges, color::Integer) =
    _colored_cell_ranges(interior_cells(ranges), ranges.global_origin, color)
