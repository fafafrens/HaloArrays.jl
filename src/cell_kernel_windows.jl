"""
    CellWindow

Compact interior-cell metadata for launch-style loops. `first` is the first
interior cell in storage coordinates and `size` is the interior-cell extent.
"""
struct CellWindow{N}
    first::CartesianIndex{N}
    size::NTuple{N,Int}
end

"""
    CellCheckerboard

Compact metadata for race-free colored cell kernels.

`size` is the launch size, compressed by two in `compressed_dim`.
`full_size` is the uncompressed interior-cell extent. GPU kernels reconstruct the
physical cell from the launch index and do a final bound check in the
compressed dimension.
"""
struct CellCheckerboard{N,C}
    first::CartesianIndex{N}
    size::NTuple{N,Int}
    full_size::NTuple{N,Int}
    color::Int
    compressed_dim::Int
    parity::Int   # mod(sum(global_origin) - sum(first), 2): anchors `color` to global parity
end

@inline CellWindow(indices::CartesianIndices{N}) where {N} =
    CellWindow(first(indices), size(indices))

@inline function _check_cell_compressed_dim(::Val{N}, compressed_dim::Integer) where {N}
    dim = Int(compressed_dim)
    1 <= dim <= N ||
        throw(ArgumentError("compressed_dim must be a spatial dimension in 1:$N, got $compressed_dim"))
    return dim
end

@inline function CellCheckerboard(first::CartesianIndex{N},
                                         size::NTuple{N,Int},
                                         full_size::NTuple{N,Int},
                                         color::Integer,
                                         compressed_dim::Integer,
                                         parity::Integer=0) where {N}
    checked_color = _check_loop_color(color)
    compressed = _check_cell_compressed_dim(Val(N), compressed_dim)
    return CellCheckerboard{N,compressed}(
        first,
        size,
        full_size,
        checked_color,
        compressed,
        Int(parity),
    )
end

"""
    cell_index(region, J)

Map a launch-local index `J` to the corresponding storage-space cell index.

For `CellCheckerboard`, `J` lives in the compressed launch region and
the returned cell is adjusted so `mod(sum(I), 2) == region.color`.
"""
@inline function cell_index(region::CellWindow{N},
                            J::NTuple{N,<:Integer}) where {N}
    first_tuple = Tuple(region.first)
    return ntuple(d -> first_tuple[d] + Int(J[d]) - 1, Val(N))
end

@inline cell_index(region::CellWindow{N}, J::CartesianIndex{N}) where {N} =
    CartesianIndex(cell_index(region, Tuple(J)))

@inline function cell_index(region::CellCheckerboard{N,C},
                            J::NTuple{N,<:Integer}) where {N,C}
    first_tuple = Tuple(region.first)

    base_tuple = ntuple(d ->
        first_tuple[d] + (d == C ? 2 * (Int(J[d]) - 1) : Int(J[d]) - 1),
        Val(N)
    )

    # shift the compressed coordinate so the GLOBAL parity matches region.color:
    # mod(sum(I) + parity, 2) == color  (parity carries the global-origin offset).
    delta = mod(region.color - region.parity - sum(base_tuple), 2)

    return ntuple(d -> base_tuple[d] + (d == C ? delta : 0), Val(N))
end

@inline cell_index(region::CellCheckerboard{N},
                   J::CartesianIndex{N}) where {N} =
    CartesianIndex(cell_index(region, Tuple(J)))

@inline _cell_region_full_size(region::CellWindow) = region.size
@inline _cell_region_full_size(region::CellCheckerboard) = region.full_size

"""
    is_cell_index_inbounds(region, I)

Return whether a storage-space cell index `I` lies inside the interior-cell extent
described by `region`. This is mainly needed by compressed colored GPU kernels,
where the last launch index may reconstruct one cell past the interior boundary.
"""
@inline function is_cell_index_inbounds(region::Union{CellWindow{N},CellCheckerboard{N}},
                                        I::NTuple{N,<:Integer}) where {N}
    first_tuple = Tuple(region.first)
    full_size = _cell_region_full_size(region)
    checks = ntuple(Val(N)) do d
        first_tuple[d] <= Int(I[d]) <= first_tuple[d] + full_size[d] - 1
    end
    return all(checks)
end

@inline is_cell_index_inbounds(region::Union{CellWindow{N},CellCheckerboard{N}},
                               I::CartesianIndex{N}) where {N} =
    is_cell_index_inbounds(region, Tuple(I))

@inline function CellCheckerboard(indices::CartesianIndices{N},
                                         color::Integer,
                                         compressed_dim::Integer=1;
                                         global_origin::NTuple{N,Int}=Tuple(first(indices))) where {N}
    checked_color = _check_loop_color(color)
    compressed = _check_cell_compressed_dim(Val(N), compressed_dim)
    full_size = size(indices)
    launch_size = ntuple(d -> d == compressed ? cld(full_size[d], 2) : full_size[d], Val(N))
    parity = mod(sum(global_origin) - sum(Tuple(first(indices))), 2)

    return CellCheckerboard(
        first(indices),
        launch_size,
        full_size,
        checked_color,
        compressed,
        parity,
    )
end

"""
    interior_cell_window(ranges)

Return compact launch metadata for the interior-cell region.
"""
@inline interior_cell_window(ranges::CellRanges) = CellWindow(interior_cells(ranges))

"""
    interior_cell_window(ranges, color; compressed_dim=1)
    interior_cell_window(ranges, color, compressed_dim)

Return compact launch metadata for one checkerboard cell color. The launch
region is compressed in `compressed_dim`; kernels should reconstruct the
physical cell index and check the compressed dimension upper bound.
"""
@inline interior_cell_window(ranges::CellRanges,
                                      color::Integer;
                                      compressed_dim::Integer=1) =
    interior_cell_window(ranges, color, compressed_dim)
@inline interior_cell_window(ranges::CellRanges,
                                      color::Integer,
                                      compressed_dim::Integer) =
    CellCheckerboard(interior_cells(ranges), color, compressed_dim;
                     global_origin=ranges.global_origin)
@inline interior_cell_window(ranges::CellRanges,
                                      color::Integer,
                                      ::Dim{D}) where {D} =
    interior_cell_window(ranges, color, D)
