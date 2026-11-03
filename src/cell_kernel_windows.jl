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
    parity::Int   # storage↔global sum-parity offset (CellRanges.parity_offset): anchors `color` to global parity
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
    # `& 1` is `mod(·, 2)` for every signed Int (negatives included) without the
    # sign fix-up `mod` carries: 1.45× on a Metal colored kernel, which runs this
    # per thread in emulated 64-bit arithmetic.
    delta = (region.color - region.parity - sum(base_tuple)) & 1

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
                                         parity::Integer=0) where {N}
    checked_color = _check_loop_color(color)
    compressed = _check_cell_compressed_dim(Val(N), compressed_dim)
    full_size = size(indices)
    launch_size = ntuple(d -> d == compressed ? cld(full_size[d], 2) : full_size[d], Val(N))

    return CellCheckerboard(
        first(indices),
        launch_size,
        full_size,
        checked_color,
        compressed,
        Int(parity),
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
                     parity=ranges.parity_offset)
@inline interior_cell_window(ranges::CellRanges,
                                      color::Integer,
                                      ::Dim{D}) where {D} =
    interior_cell_window(ranges, color, D)

# ============================================================
# Linear-index cell windows
#
# A stencil kernel that addresses the storage through a CartesianIndex redoes
# the full Cartesian→linear conversion for every neighbour read (`x[I+eμ]`):
# N multiplies in 64-bit integer arithmetic, which GPUs emulate (Apple) or run
# at reduced rate. A LinearCellWindow converts once per cell and reads the
# neighbours at `i ± sμ` with the storage strides, in the index type the window
# was built with (Int32 by default). Measured on Metal, 48⁴, nearest-neighbour
# stencils: 1.4× (force kernel) to 2.1× (colored Metropolis) over `cell_index`.
# ============================================================

"""
    LinearCellWindow{N,C,T}

Launch metadata for GPU / KernelAbstractions cell kernels that address a halo
array's storage by **linear** index: [`cell_linear_index`](@ref) maps a launch
index to the storage linear index of its cell and [`cell_strides`](@ref) gives
the offsets of the nearest neighbours, so a stencil reads `x[i + sμ]`,
`x[i - sμ]` with one add per neighbour. `C` is the checkerboard compressed
dimension (`0`: all interior cells) and `T` the index type (`Int32` by default,
native on GPUs). Build it with [`linear_cell_window`](@ref).
"""
struct LinearCellWindow{N,C,T<:Integer}
    first::T                   # storage linear index of the first interior cell
    strides::NTuple{N,T}       # storage strides (column-major)
    size::NTuple{N,Int}        # launch size (compressed by two in C for a checkerboard)
    first_coords::NTuple{N,T}  # storage coordinates of `first` (checkerboard parity)
    extent::T                  # interior extent along C (checkerboard bound check)
    color::T                   # storage parity of the requested color (global color, parity offset applied)
end

@inline function _column_major_strides(::Type{T}, sz::NTuple{N,Int}) where {T,N}
    ntuple(Val(N)) do d
        s = 1
        for k in 1:d-1
            s *= sz[k]
        end
        T(s)
    end
end

function _check_linear_index_type(::Type{T}, data::AbstractArray) where {T}
    length(data) <= typemax(T) || throw(ArgumentError(
        "a storage of $(length(data)) elements does not fit index_type=$T; pass index_type=Int"))
    return nothing
end

"""
    linear_cell_window(ranges, data; index_type=Int32)
    linear_cell_window(ranges, data, color; compressed_dim=1, index_type=Int32)
    linear_cell_window(ranges, data, color, Dim(d); index_type=Int32)
    linear_cell_window(u[, color]; kwargs...)

Return a [`LinearCellWindow`](@ref) over the interior cells of `ranges` (a
[`CellRanges`](@ref)) in the storage array `data` (the `parent` of the halo
array, or [`tile_parent`](@ref) of a tile), for all cells or for one checkerboard
`color` — the launch is then compressed by two in `compressed_dim`, like
[`interior_cell_window`](@ref), and kernels check
[`is_cell_linear_index_inbounds`](@ref). `linear_cell_window(u, ...)` is the
one-tile shorthand for `linear_cell_window(CellRanges(u), tile_parent(u, 1), ...)`
(a `LocalHaloArray`, an MPI `HaloArray`, or a `ThreadedHaloArray` with one
tile); with more tiles build one window per tile. `index_type` must hold
`length(data)`.

```julia
w = linear_cell_window(u, color)              # host, once per color
@kernel function update!(x, w, m2)
    J = @index(Global, NTuple)
    if is_cell_linear_index_inbounds(w, J)
        i = cell_linear_index(w, J)
        s = cell_strides(w)
        ϕ₊ = map(sμ -> x[i + sμ], s)          # neighbours: one add each
        ϕ₋ = map(sμ -> x[i - sμ], s)
        x[i] = ...
    end
end
update!(backend)(parent(u), w, m2; ndrange=w.size)
```
"""
function linear_cell_window(ranges::CellRanges, data::AbstractArray{<:Any,N};
                            index_type::Type{T}=Int32) where {N,T<:Integer}
    _check_linear_index_type(T, data)
    region = interior_cell_window(ranges)
    return LinearCellWindow{N,0,T}(T(LinearIndices(size(data))[region.first]),
                                   _column_major_strides(T, size(data)),
                                   region.size,
                                   map(T, Tuple(region.first)),
                                   zero(T), zero(T))
end

function linear_cell_window(ranges::CellRanges, data::AbstractArray{<:Any,N}, color::Integer;
                            compressed_dim::Integer=1, index_type::Type{T}=Int32) where {N,T<:Integer}
    _check_linear_index_type(T, data)
    # validates color and compressed_dim, carries the tile's parity offset
    return _linear_cell_window(interior_cell_window(ranges, color, compressed_dim), data, T)
end

@inline linear_cell_window(ranges::CellRanges, data::AbstractArray, color::Integer, ::Dim{D};
                           kwargs...) where {D} =
    linear_cell_window(ranges, data, color; compressed_dim=D, kwargs...)

# The shorthand addresses the one tile's padded storage: `parent` of a
# ThreadedHaloArray is the vector of tiles, `tile_parent(u, 1)` is the storage on
# every backend. More tiles need one window per tile (each has its own storage
# and parity offset).
function _single_tile_storage(u::AbstractSingleHaloArray)
    tile_count(u) == 1 || throw(ArgumentError(
        "linear_cell_window(u, ...) addresses one tile's storage, but u has $(tile_count(u)) tiles; " *
        "build one window per tile: linear_cell_window(CellRanges(u, tile_id), tile_parent(u, tile_id), ...)"))
    return tile_parent(u, 1)
end

@inline linear_cell_window(u::AbstractSingleHaloArray; kwargs...) =
    linear_cell_window(CellRanges(u), _single_tile_storage(u); kwargs...)
@inline linear_cell_window(u::AbstractSingleHaloArray, color::Integer; kwargs...) =
    linear_cell_window(CellRanges(u), _single_tile_storage(u), color; kwargs...)

function _linear_cell_window(region::CellCheckerboard{N,C}, data::AbstractArray{<:Any,N},
                             ::Type{T}) where {N,C,T}
    storage_color = (region.color - region.parity) & 1
    return LinearCellWindow{N,C,T}(T(LinearIndices(size(data))[region.first]),
                                   _column_major_strides(T, size(data)),
                                   region.size,
                                   map(T, Tuple(region.first)),
                                   T(region.full_size[C]),
                                   T(storage_color))
end

"""
    cell_strides(w)

Storage strides of a [`LinearCellWindow`](@ref): the linear offsets of the
nearest neighbours, `x[i + sμ]` / `x[i - sμ]` (the linear-index counterpart of
[`unit_vector`](@ref)).
"""
@inline cell_strides(w::LinearCellWindow) = w.strides

# `foldl`, not `sum`: `sum` promotes Int32 to Int.
@inline _dot_tuple(a::NTuple{N,<:Integer}, b::NTuple{N,<:Integer}) where {N} =
    foldl(+, ntuple(d -> a[d] * b[d], Val(N)))

# launch offsets from the first cell (step two along the compressed dimension) and
# the parity shift of the compressed coordinate, both in the window's index type
@inline function _linear_offset(w::LinearCellWindow{N,C,T}, J::NTuple{N,<:Integer}) where {N,C,T}
    off = ntuple(d -> (T(J[d]) - one(T)) * (d == C ? T(2) : one(T)), Val(N))
    delta = (w.color - foldl(+, w.first_coords) - foldl(+, off)) & one(T)
    return off, delta
end

"""
    cell_linear_index(w, J)

Map a launch index `J` (tuple or `CartesianIndex`) of a [`LinearCellWindow`](@ref)
to the storage linear index of its cell, in the window's index type. For a
checkerboard window the cell is the one of the window's color; check
[`is_cell_linear_index_inbounds`](@ref) first.
"""
@inline function cell_linear_index(w::LinearCellWindow{N,0,T}, J::NTuple{N,<:Integer}) where {N,T}
    off = ntuple(d -> T(J[d]) - one(T), Val(N))
    return w.first + _dot_tuple(off, w.strides)
end

@inline function cell_linear_index(w::LinearCellWindow{N,C,T}, J::NTuple{N,<:Integer}) where {N,C,T}
    off, delta = _linear_offset(w, J)
    return w.first + _dot_tuple(off, w.strides) + delta * w.strides[C]
end

@inline cell_linear_index(w::LinearCellWindow{N}, J::CartesianIndex{N}) where {N} =
    cell_linear_index(w, Tuple(J))

"""
    is_cell_linear_index_inbounds(w, J)

Whether the launch index `J` of a checkerboard [`LinearCellWindow`](@ref) maps to
a cell inside the interior: the compressed launch dimension may reconstruct one
cell past the interior extent. Always `true` for a full window.
"""
@inline is_cell_linear_index_inbounds(::LinearCellWindow{N,0}, ::NTuple{N,<:Integer}) where {N} = true

@inline function is_cell_linear_index_inbounds(w::LinearCellWindow{N,C}, J::NTuple{N,<:Integer}) where {N,C}
    off, delta = _linear_offset(w, J)
    return off[C] + delta < w.extent
end

@inline is_cell_linear_index_inbounds(w::LinearCellWindow{N}, J::CartesianIndex{N}) where {N} =
    is_cell_linear_index_inbounds(w, Tuple(J))
