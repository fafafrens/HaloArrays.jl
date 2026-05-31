struct ThreadedCartesianTopology{N} <: AbstractCartesianTopology{N}
    dims::NTuple{N,Int}
    tile_coords::Vector{NTuple{N,Int}}
    neighbors::Vector{NTuple{N,NTuple{2,Int}}}
    periodic_boundary_condition::NTuple{N,Bool}
end

function _threaded_neighbor_id(coord::NTuple{N,Int}, dims::NTuple{N,Int}, periodic::NTuple{N,Bool},
        linear_indices::LinearIndices, dim::Int, step::Int) where {N}
    next_dim_coord = coord[dim] + step

    if 1 <= next_dim_coord <= dims[dim]
        next_coord = ntuple(d -> d == dim ? next_dim_coord : coord[d], Val(N))
        return linear_indices[CartesianIndex(next_coord)]
    elseif periodic[dim]
        wrapped_dim_coord = next_dim_coord < 1 ? dims[dim] : 1
        next_coord = ntuple(d -> d == dim ? wrapped_dim_coord : coord[d], Val(N))
        return linear_indices[CartesianIndex(next_coord)]
    else
        return 0
    end
end

"""
    ThreadedCartesianTopology(dims; periodic)

Local Cartesian tile topology for `ThreadedHaloArray`.

The neighbor map is precomputed at construction. A neighbor id of `0` means
that side is a physical boundary.
"""
function ThreadedCartesianTopology(dims::NTuple{N,<:Integer}; periodic=ntuple(_ -> false, Val(N))) where {N}
    tile_dims = ntuple(d -> Int(dims[d]), Val(N))
    all(d -> d > 0, tile_dims) || throw(ArgumentError("threaded topology dims must be positive"))

    periodic_tuple = ntuple(d -> Bool(periodic[d]), Val(N))
    cartesian_indices = CartesianIndices(tile_dims)
    linear_indices = LinearIndices(tile_dims)
    tile_coords = Vector{NTuple{N,Int}}(undef, length(cartesian_indices))
    neighbors = Vector{NTuple{N,NTuple{2,Int}}}(undef, length(cartesian_indices))

    for tile_id in 1:length(cartesian_indices)
        coord = Tuple(cartesian_indices[tile_id])
        tile_coords[tile_id] = coord
        neighbors[tile_id] = ntuple(Val(N)) do dim
            lower = _threaded_neighbor_id(coord, tile_dims, periodic_tuple, linear_indices, dim, -1)
            upper = _threaded_neighbor_id(coord, tile_dims, periodic_tuple, linear_indices, dim, 1)
            return (lower, upper)
        end
    end

    return ThreadedCartesianTopology{N}(tile_dims, tile_coords, neighbors, periodic_tuple)
end

@inline Base.ndims(::ThreadedCartesianTopology{N}) where {N} = N
@inline isactive(::ThreadedCartesianTopology) = true
@inline is_root(::ThreadedCartesianTopology; root::Integer=0) = true
@inline tile_count(topology::ThreadedCartesianTopology) = prod(topology.dims)
@inline tile_coordinates(topology::ThreadedCartesianTopology, tile_id::Integer) = topology.tile_coords[tile_id]
@inline neighbor_tile_id(topology::ThreadedCartesianTopology, tile_id::Integer, dim::Integer, side::Integer) =
    topology.neighbors[tile_id][dim][side]

# validate_boundary_condition is inherited from AbstractCartesianTopology (abstract_haloarray.jl)

struct ThreadedHaloArray{T,N,A,Halo,Topo,BCondition} <: AbstractSerialHaloArray{T,N}
    data::Vector{A}
    tile_size::NTuple{N,Int}
    topology::Topo
    boundary_condition::BCondition
end

@inline halo_backend(::Type{<:ThreadedHaloArray}) = ThreadedHaloBackend()

function ThreadedHaloArray(::Type{T}, tile_size::NTuple{N,<:Integer}, halo::Integer;
        dims::NTuple{N,<:Integer},
        boundary_condition=:repeating) where {T,N}
    halo_int = Int(halo)
    halo_int >= 0 || throw(ArgumentError("halo width must be non-negative"))

    owned_tile_size = ntuple(d -> Int(tile_size[d]), Val(N))
    all(d -> d > 0, owned_tile_size) || throw(ArgumentError("tile_size entries must be positive"))
    all(d -> owned_tile_size[d] >= halo_int, 1:N) ||
        throw(ArgumentError("each tile_size entry must be at least the halo width"))

    bc = normalize_boundary_condition(boundary_condition, N)
    topology = ThreadedCartesianTopology(dims; periodic=infer_periodicity(bc))
    validate_boundary_condition(topology, bc)
    full_tile_size = ntuple(d -> owned_tile_size[d] + 2 * halo_int, Val(N))
    data = [zeros(T, full_tile_size...) for _ in 1:tile_count(topology)]

    return ThreadedHaloArray{T,N,typeof(data[1]),halo_int,typeof(topology),typeof(bc)}(
        data, owned_tile_size, topology, bc,
    )
end

ThreadedHaloArray(tile_size::NTuple{N,<:Integer}, halo::Integer; kwargs...) where {N} =
    ThreadedHaloArray(Float64, tile_size, halo; kwargs...)

@inline Base.eltype(::ThreadedHaloArray{T}) where {T} = T
@inline Base.eltype(::Type{<:ThreadedHaloArray{T}}) where {T} = T
@inline Base.ndims(::ThreadedHaloArray{T,N}) where {T,N} = N
@inline Base.ndims(::Type{<:ThreadedHaloArray{T,N}}) where {T,N} = N
@inline Base.parent(halo::ThreadedHaloArray) = halo.data
@inline Base.length(halo::ThreadedHaloArray) = prod(size(halo))
@inline owned_size(halo::ThreadedHaloArray) = ntuple(d -> halo.tile_size[d] * halo.topology.dims[d], Val(ndims(halo)))
@inline Base.size(halo::ThreadedHaloArray) = global_size(halo)
@inline Base.size(halo::ThreadedHaloArray, d::Int) = size(halo)[d]
@inline Base.axes(halo::ThreadedHaloArray) = map(Base.OneTo, size(halo))
@inline Base.axes(halo::ThreadedHaloArray, d::Int) = Base.OneTo(size(halo, d))
@inline owned_axes(halo::ThreadedHaloArray) = map(Base.OneTo, owned_size(halo))
@inline owned_axes(halo::ThreadedHaloArray, d::Int) = Base.OneTo(owned_size(halo, d))
@inline interior_size(halo::ThreadedHaloArray) = owned_size(halo)
@inline halo_width(::Type{<:ThreadedHaloArray{T,N,A,Halo}}) where {T,N,A,Halo} = Halo
@inline halo_width(::ThreadedHaloArray{T,N,A,Halo}) where {T,N,A,Halo} = Halo
@inline tile_size(halo::ThreadedHaloArray) = halo.tile_size
@inline tile_count(halo::ThreadedHaloArray) = length(parent(halo))
@inline tile_parent(halo::ThreadedHaloArray, tile_id::Integer) = parent(halo)[tile_id]
@inline tile_coordinates(halo::ThreadedHaloArray, tile_id::Integer) = tile_coordinates(halo.topology, tile_id)
@inline global_size(halo::ThreadedHaloArray) = owned_size(halo)
@inline isactive(::ThreadedHaloArray) = true
@inline is_root(halo::ThreadedHaloArray; root::Integer=0) = is_root(halo.topology; root=root)
@inline get_comm(::ThreadedHaloArray) = nothing

@inline function _threaded_global_to_tile_index(halo::ThreadedHaloArray{T,N}, I) where {T,N}
    idx = _check_global_scalar_indices(halo, I)
    owned_tile_size = tile_size(halo)
    tile_coord = ntuple(d -> ((idx[d] - 1) ÷ owned_tile_size[d]) + 1, Val(N))
    tile_local_idx = ntuple(d -> ((idx[d] - 1) % owned_tile_size[d]) + 1 + halo_width(halo), Val(N))
    tile_id = LinearIndices(halo.topology.dims)[CartesianIndex(tile_coord)]
    return tile_id, tile_local_idx
end

function Base.getindex(halo::ThreadedHaloArray, I::Vararg{Integer})
    tile_id, tile_local_idx = _threaded_global_to_tile_index(halo, I)
    @inbounds return tile_parent(halo, tile_id)[tile_local_idx...]
end

function Base.setindex!(halo::ThreadedHaloArray, value, I::Vararg{Integer})
    tile_id, tile_local_idx = _threaded_global_to_tile_index(halo, I)
    @inbounds tile_parent(halo, tile_id)[tile_local_idx...] = value
    return halo
end

@inline function storage_size(halo::ThreadedHaloArray)
    return ntuple(d -> tile_size(halo)[d] + 2 * halo_width(halo), Val(ndims(halo)))
end

@inline storage_size(halo::ThreadedHaloArray, d::Int) = storage_size(halo)[d]
@inline full_tile_size(halo::ThreadedHaloArray, tile_id::Integer) = size(tile_parent(halo, tile_id))
@inline full_tile_size(halo::ThreadedHaloArray, tile_id::Integer, d::Int) = size(tile_parent(halo, tile_id), d)

@inline function interior_range(halo::ThreadedHaloArray)
    h = halo_width(halo)
    return ntuple(d -> (h + 1):(h + tile_size(halo)[d]), Val(ndims(halo)))
end

@inline interior_range(halo::ThreadedHaloArray, tile_id::Integer) = interior_range(halo)

@inline function full_range(halo::ThreadedHaloArray)
    return ntuple(d -> 1:storage_size(halo, d), Val(ndims(halo)))
end

@inline full_range(halo::ThreadedHaloArray, tile_id::Integer) =
    ntuple(d -> 1:full_tile_size(halo, tile_id, d), Val(ndims(halo)))

@inline function interior_view(halo::ThreadedHaloArray, tile_id::Integer)
    ranges = interior_range(halo, tile_id)
    @views return tile_parent(halo, tile_id)[ranges...]
end

function interior_view(halo::ThreadedHaloArray)
    return [interior_view(halo, tile_id) for tile_id in eachindex(parent(halo))]
end

@inline function full_view(halo::ThreadedHaloArray, tile_id::Integer)
    ranges = full_range(halo, tile_id)
    @views return tile_parent(halo, tile_id)[ranges...]
end

@inline function get_send_view(::Side{1}, ::Dim{D}, halo::ThreadedHaloArray, tile_id::Int) where {D}
    return get_send_view(Side(1), Dim(D), tile_parent(halo, tile_id), halo_width(halo))
end

@inline function get_send_view(::Side{2}, ::Dim{D}, halo::ThreadedHaloArray, tile_id::Int) where {D}
    return get_send_view(Side(2), Dim(D), tile_parent(halo, tile_id), halo_width(halo))
end

@inline get_send_view(s::Side, dim::Int, halo::ThreadedHaloArray, tile_id::Integer) =
    get_send_view(s, Dim(dim), halo, tile_id)

@inline function get_recv_view(::Side{1}, ::Dim{D}, halo::ThreadedHaloArray, tile_id::Int) where {D}
    return get_recv_view(Side(1), Dim(D), tile_parent(halo, tile_id), halo_width(halo))
end

@inline function get_recv_view(::Side{2}, ::Dim{D}, halo::ThreadedHaloArray, tile_id::Int) where {D}
    return get_recv_view(Side(2), Dim(D), tile_parent(halo, tile_id), halo_width(halo))
end

@inline get_recv_view(s::Side, dim::Int, halo::ThreadedHaloArray, tile_id::Integer) =
    get_recv_view(s, Dim(dim), halo, tile_id)

@inline function neighbor_tile_id(halo::ThreadedHaloArray, tile_id::Integer, dim::Integer, side::Integer)
    return neighbor_tile_id(halo.topology, tile_id, dim, side)
end

@inline function _threaded_exchange_side!(halo::ThreadedHaloArray, tile_id::Integer,
        dim::Dim{D}, side::Side{S}) where {D,S}
    neighbor_id = neighbor_tile_id(halo, tile_id, D, S)
    if neighbor_id != 0
        _threaded_copy_side!(halo, tile_id, neighbor_id, dim, side)
    end
    return halo
end

@inline function _threaded_copy_side!(halo::ThreadedHaloArray, tile_id::Integer, neighbor_id::Integer,
        dim::Dim{D}, side::Side{S}) where {D,S}
    recv_view = get_recv_view(side, dim, halo, tile_id)
    send_view = get_send_view(Side(3 - S), dim, halo, neighbor_id)
    copyto!(recv_view, send_view)
    return halo
end

@inline function _threaded_boundary_side!(halo::ThreadedHaloArray, tile_id::Integer,
        dim::Dim{D}, side::Side{S}) where {D,S}
    if neighbor_tile_id(halo, tile_id, D, S) == 0
        mode = halo.boundary_condition[D][S]
        boundary_condition!(halo, tile_id, side, dim, mode)
    end
    return halo
end

@inline function _threaded_synchronize_side!(halo::ThreadedHaloArray, tile_id::Integer,
        dim::Dim{D}, side::Side{S}) where {D,S}
    neighbor_id = neighbor_tile_id(halo, tile_id, D, S)
    if neighbor_id != 0
        _threaded_copy_side!(halo, tile_id, neighbor_id, dim, side)
    else
        mode = halo.boundary_condition[D][S]
        boundary_condition!(halo, tile_id, side, dim, mode)
    end
    return halo
end

@inline _threaded_exchange_dims!(halo::ThreadedHaloArray, tile_id::Integer, ::Val{0}) = halo
@inline _threaded_boundary_dims!(halo::ThreadedHaloArray, tile_id::Integer, ::Val{0}) = halo
@inline _threaded_synchronize_dims!(halo::ThreadedHaloArray, tile_id::Integer, ::Val{0}) = halo

@inline function _threaded_exchange_dims!(halo::ThreadedHaloArray, tile_id::Integer, ::Val{D}) where {D}
    _threaded_exchange_dims!(halo, tile_id, Val(D - 1))
    _threaded_exchange_side!(halo, tile_id, Dim(D), Side(1))
    _threaded_exchange_side!(halo, tile_id, Dim(D), Side(2))
    return halo
end

@inline function _threaded_boundary_dims!(halo::ThreadedHaloArray, tile_id::Integer, ::Val{D}) where {D}
    _threaded_boundary_dims!(halo, tile_id, Val(D - 1))
    _threaded_boundary_side!(halo, tile_id, Dim(D), Side(1))
    _threaded_boundary_side!(halo, tile_id, Dim(D), Side(2))
    return halo
end

@inline function _threaded_synchronize_dims!(halo::ThreadedHaloArray, tile_id::Integer, ::Val{D}) where {D}
    _threaded_synchronize_dims!(halo, tile_id, Val(D - 1))
    _threaded_synchronize_side!(halo, tile_id, Dim(D), Side(1))
    _threaded_synchronize_side!(halo, tile_id, Dim(D), Side(2))
    return halo
end

@inline _threaded_exchange_tile!(halo::ThreadedHaloArray{T,N}, tile_id::Integer) where {T,N} =
    _threaded_exchange_dims!(halo, tile_id, Val(N))

@inline _threaded_boundary_tile!(halo::ThreadedHaloArray{T,N}, tile_id::Integer) where {T,N} =
    _threaded_boundary_dims!(halo, tile_id, Val(N))

@inline _threaded_synchronize_tile!(halo::ThreadedHaloArray{T,N}, tile_id::Integer) where {T,N} =
    _threaded_synchronize_dims!(halo, tile_id, Val(N))

function halo_exchange!(halo::ThreadedHaloArray)
    @inbounds for tile_id in eachindex(parent(halo))
        _threaded_exchange_tile!(halo, tile_id)
    end
    return halo
end

function halo_exchange_threads!(halo::ThreadedHaloArray)
    tforeach(tile_id -> _threaded_exchange_tile!(halo, tile_id),
        eachindex(parent(halo)); scheduler=:static)
    return halo
end

function boundary_condition!(halo::ThreadedHaloArray, tile_id::Integer, side::Side{S}, dim::Dim{D}) where {S,D}
    _threaded_boundary_side!(halo, tile_id, dim, side)
    return nothing
end

function boundary_condition!(halo::ThreadedHaloArray{T,N,A,H,Topo,BCondition}, tile_id::Integer,
        s::Side{1}, d::Dim{dim}, mode::Reflecting) where {T,N,A,H,Topo,BCondition,dim}
    h = halo_width(halo)
    interior_region = interior_view(halo, tile_id)
    halo_region = get_recv_view(s, d, halo, tile_id)

    for i in 1:size(halo_region, dim)
        src_i = h - i + 1
        mirror_idx = _slice_index(Val(N), dim, src_i)
        dst_idx = _slice_index(Val(N), dim, i)
        @views halo_region[dst_idx...] .= interior_region[mirror_idx...]
    end
    return nothing
end

function boundary_condition!(halo::ThreadedHaloArray{T,N,A,H,Topo,BCondition}, tile_id::Integer,
        s::Side{2}, d::Dim{dim}, mode::Reflecting) where {T,N,A,H,Topo,BCondition,dim}
    interior_region = interior_view(halo, tile_id)
    halo_region = get_recv_view(s, d, halo, tile_id)
    n = size(interior_region, dim)

    for i in 1:size(halo_region, dim)
        src_i = n - (i - 1)
        mirror_idx = _slice_index(Val(N), dim, src_i)
        dst_idx = _slice_index(Val(N), dim, i)
        @views halo_region[dst_idx...] .= interior_region[mirror_idx...]
    end
    return nothing
end

function boundary_condition!(halo::ThreadedHaloArray{T,N,A,H,Topo,BCondition}, tile_id::Integer,
        s::Side{1}, d::Dim{dim}, mode::Antireflecting) where {T,N,A,H,Topo,BCondition,dim}
    h = halo_width(halo)
    interior_region = interior_view(halo, tile_id)
    halo_region = get_recv_view(s, d, halo, tile_id)

    for i in 1:size(halo_region, dim)
        src_i = h - i + 1
        mirror_idx = _slice_index(Val(N), dim, src_i)
        dst_idx = _slice_index(Val(N), dim, i)
        @views halo_region[dst_idx...] .= .-interior_region[mirror_idx...]
    end
    return nothing
end

function boundary_condition!(halo::ThreadedHaloArray{T,N,A,H,Topo,BCondition}, tile_id::Integer,
        s::Side{2}, d::Dim{dim}, mode::Antireflecting) where {T,N,A,H,Topo,BCondition,dim}
    interior_region = interior_view(halo, tile_id)
    halo_region = get_recv_view(s, d, halo, tile_id)
    n = size(interior_region, dim)

    for i in 1:size(halo_region, dim)
        src_i = n - (i - 1)
        mirror_idx = _slice_index(Val(N), dim, src_i)
        dst_idx = _slice_index(Val(N), dim, i)
        @views halo_region[dst_idx...] .= .-interior_region[mirror_idx...]
    end
    return nothing
end

function boundary_condition!(halo::ThreadedHaloArray{T,N,A,H,Topo,BCondition}, tile_id::Integer,
        s::Side{1}, d::Dim{dim}, mode::Repeating) where {T,N,A,H,Topo,BCondition,dim}
    interior_region = interior_view(halo, tile_id)
    halo_region = get_recv_view(s, d, halo, tile_id)
    edge_idx = _slice_index(Val(N), dim, 1)
    boundary_slice = @view interior_region[edge_idx...]

    for i in 1:size(halo_region, dim)
        halo_idx = _slice_index(Val(N), dim, i)
        @views halo_region[halo_idx...] .= boundary_slice
    end
    return nothing
end

function boundary_condition!(halo::ThreadedHaloArray{T,N,A,H,Topo,BCondition}, tile_id::Integer,
        s::Side{2}, d::Dim{dim}, mode::Repeating) where {T,N,A,H,Topo,BCondition,dim}
    interior_region = interior_view(halo, tile_id)
    halo_region = get_recv_view(s, d, halo, tile_id)
    edge_idx = _slice_index(Val(N), dim, size(interior_region, dim))
    boundary_slice = @view interior_region[edge_idx...]

    for i in 1:size(halo_region, dim)
        halo_idx = _slice_index(Val(N), dim, i)
        @views halo_region[halo_idx...] .= boundary_slice
    end
    return nothing
end

function boundary_condition!(halo::ThreadedHaloArray, tile_id::Integer, s::Side, d::Dim, mode::Periodic)
    return nothing
end

function boundary_condition!(halo::ThreadedHaloArray)
    @inbounds for tile_id in eachindex(parent(halo))
        _threaded_boundary_tile!(halo, tile_id)
    end
    return nothing
end

function boundary_condition_threads!(halo::ThreadedHaloArray)
    tforeach(tile_id -> _threaded_boundary_tile!(halo, tile_id),
        eachindex(parent(halo)); scheduler=:static)
    return nothing
end

function synchronize_halo!(halo::ThreadedHaloArray)
    @inbounds for tile_id in eachindex(parent(halo))
        _threaded_synchronize_tile!(halo, tile_id)
    end
    return halo
end

function synchronize_halo_threads!(halo::ThreadedHaloArray)
    tforeach(tile_id -> _threaded_synchronize_tile!(halo, tile_id),
        eachindex(parent(halo)); scheduler=:static)
    return halo
end

start_halo_exchange!(halo::ThreadedHaloArray) = halo_exchange!(halo)
finish_halo_exchange!(halo::ThreadedHaloArray) = halo

function fill_interior(halo::ThreadedHaloArray, value)
    tforeach(tile_id -> _fill_threaded_interior_tile!(halo, tile_id, value),
        eachindex(parent(halo)); scheduler=:static)
    return halo
end

function Base.fill!(halo::ThreadedHaloArray, value)
    tforeach(tile -> fill!(tile, value), parent(halo); scheduler=:static)
    return halo
end

@inline function _fill_threaded_interior_tile!(halo::ThreadedHaloArray, tile_id::Integer, value)
    fill!(interior_view(halo, tile_id), value)
    return nothing
end

function _global_to_tile_dims(halo::ThreadedHaloArray{T,N}, dims::NTuple{M,<:Integer}) where {T,N,M}
    M == N || throw(DimensionMismatch("ThreadedHaloArray similar dims must have $N dimensions"))
    topo_dims = halo.topology.dims
    all(d -> Int(dims[d]) % topo_dims[d] == 0, 1:N) ||
        throw(DimensionMismatch("ThreadedHaloArray global similar dims $dims are not divisible by topology dims $topo_dims"))
    return ntuple(d -> Int(dims[d]) ÷ topo_dims[d], Val(N))
end

function Base.similar(halo::ThreadedHaloArray{T,N,A,Halo,B,BCondition}, ::Type{AA},
        dims::Dims{M}) where {T,N,A,Halo,B,BCondition,AA,M}
    tile_dims = _global_to_tile_dims(halo, dims)
    full_tile_size = ntuple(d -> tile_dims[d] + 2 * halo_width(halo), Val(N))
    data = [similar(tile_parent(halo, 1), AA, full_tile_size) for _ in 1:tile_count(halo)]
    return ThreadedHaloArray{AA,N,typeof(data[1]),Halo,typeof(halo.topology),typeof(halo.boundary_condition)}(
        data, tile_dims, halo.topology, halo.boundary_condition,
    )
end

Base.similar(halo::ThreadedHaloArray{T,N,A,Halo,B,BCondition}, ::Type{AA},
    dims::NTuple{M,<:Integer}) where {T,N,A,Halo,B,BCondition,AA,M} =
    similar(halo, AA, ntuple(d -> Int(dims[d]), Val(M)))

Base.similar(halo::ThreadedHaloArray) = similar(halo, eltype(halo), size(halo))
Base.similar(halo::ThreadedHaloArray, ::Type{AA}) where {AA} = similar(halo, AA, size(halo))
Base.similar(halo::ThreadedHaloArray, dims::Dims{M}) where {M} = similar(halo, eltype(halo), dims)
Base.similar(halo::ThreadedHaloArray, dims::NTuple{M,<:Integer}) where {M} =
    similar(halo, eltype(halo), dims)

# Base.zero inherited from AbstractSingleHaloArray

function Base.copyto!(dest::ThreadedHaloArray, src::ThreadedHaloArray)
    size(dest) == size(src) || throw(DimensionMismatch("ThreadedHaloArray copyto! requires matching global sizes"))
    tile_size(dest) == tile_size(src) || throw(DimensionMismatch("ThreadedHaloArray copyto! requires matching tile sizes"))
    tile_count(dest) == tile_count(src) || throw(DimensionMismatch("ThreadedHaloArray copyto! requires matching tile counts"))

    tforeach(tile_id -> _copy_threaded_tile_storage!(dest, src, tile_id),
        eachindex(parent(dest)); scheduler=:static)
    return dest
end

@inline function _copy_threaded_tile_storage!(dest::ThreadedHaloArray, src::ThreadedHaloArray, tile_id::Integer)
    copyto!(tile_parent(dest, tile_id), tile_parent(src, tile_id))
    return nothing
end

# Base.copy inherited from AbstractSingleHaloArray
