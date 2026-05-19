struct ThreadedCartesianTopology{N}
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
@inline tile_count(topology::ThreadedCartesianTopology) = prod(topology.dims)
@inline tile_coordinates(topology::ThreadedCartesianTopology, tile_id::Integer) = topology.tile_coords[tile_id]
@inline neighbor_tile_id(topology::ThreadedCartesianTopology, tile_id::Integer, dim::Integer, side::Integer) =
    topology.neighbors[tile_id][dim][side]

function validate_boundary_condition(topology::ThreadedCartesianTopology{N}, boundary_condition) where {N}
    for d in 1:N
        left, right = boundary_condition[d]

        if !(left isa AbstractBoundaryCondition) || !(right isa AbstractBoundaryCondition)
            error("boundary_condition[$d] must be a tuple of AbstractBoundaryCondition (got $(left), $(right))")
        end

        topo_is_periodic = topology.periodic_boundary_condition[d]
        both_periodic = (left isa Periodic) && (right isa Periodic)
        any_periodic = (left isa Periodic) || (right isa Periodic)

        if topo_is_periodic && !both_periodic
            error("Threaded topology is periodic in dimension $d but boundary_condition[$d] is not (both sides must be Periodic).")
        elseif !topo_is_periodic && any_periodic
            error("Boundary condition in dimension $d uses Periodic but threaded topology is not periodic.")
        end
    end

    return true
end

struct ThreadedHaloArray{T,N,A,Halo,Topo,BCondition}
    data::Vector{A}
    tile_size::NTuple{N,Int}
    topology::Topo
    boundary_condition::BCondition
end

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
@inline Base.ndims(::ThreadedHaloArray{T,N}) where {T,N} = N
@inline Base.ndims(::Type{<:ThreadedHaloArray{T,N}}) where {T,N} = N
@inline Base.parent(halo::ThreadedHaloArray) = halo.data
@inline Base.length(halo::ThreadedHaloArray) = prod(size(halo))
@inline Base.size(halo::ThreadedHaloArray) = ntuple(d -> halo.tile_size[d] * halo.topology.dims[d], Val(ndims(halo)))
@inline Base.size(halo::ThreadedHaloArray, d::Int) = size(halo)[d]
@inline Base.axes(halo::ThreadedHaloArray) = map(Base.OneTo, size(halo))
@inline Base.axes(halo::ThreadedHaloArray, d::Int) = Base.OneTo(size(halo, d))
@inline interior_size(halo::ThreadedHaloArray) = size(halo)
@inline halo_width(::Type{<:ThreadedHaloArray{T,N,A,Halo}}) where {T,N,A,Halo} = Halo
@inline halo_width(::ThreadedHaloArray{T,N,A,Halo}) where {T,N,A,Halo} = Halo
@inline tile_size(halo::ThreadedHaloArray) = halo.tile_size
@inline tile_count(halo::ThreadedHaloArray) = length(parent(halo))
@inline tile_parent(halo::ThreadedHaloArray, tile_id::Integer) = parent(halo)[tile_id]
@inline tile_coordinates(halo::ThreadedHaloArray, tile_id::Integer) = tile_coordinates(halo.topology, tile_id)
@inline global_size(halo::ThreadedHaloArray) = size(halo)
@inline isactive(::ThreadedHaloArray) = true
@inline get_comm(::ThreadedHaloArray) = nothing

@inline function full_size(halo::ThreadedHaloArray)
    return ntuple(d -> tile_size(halo)[d] + 2 * halo_width(halo), Val(ndims(halo)))
end

@inline full_size(halo::ThreadedHaloArray, d::Int) = full_size(halo)[d]
@inline full_tile_size(halo::ThreadedHaloArray, tile_id::Integer) = size(tile_parent(halo, tile_id))
@inline full_tile_size(halo::ThreadedHaloArray, tile_id::Integer, d::Int) = size(tile_parent(halo, tile_id), d)

@inline function interior_range(halo::ThreadedHaloArray)
    h = halo_width(halo)
    return ntuple(d -> (h + 1):(h + tile_size(halo)[d]), Val(ndims(halo)))
end

@inline interior_range(halo::ThreadedHaloArray, tile_id::Integer) = interior_range(halo)

@inline function full_range(halo::ThreadedHaloArray)
    return ntuple(d -> 1:full_size(halo, d), Val(ndims(halo)))
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

@inline function get_send_view(::Side{1}, ::Dim{D}, halo::ThreadedHaloArray, tile_id::Integer) where {D}
    return get_send_view(Side(1), Dim(D), tile_parent(halo, tile_id), halo_width(halo))
end

@inline function get_send_view(::Side{2}, ::Dim{D}, halo::ThreadedHaloArray, tile_id::Integer) where {D}
    return get_send_view(Side(2), Dim(D), tile_parent(halo, tile_id), halo_width(halo))
end

@inline get_send_view(s::Side, dim::Int, halo::ThreadedHaloArray, tile_id::Integer) =
    get_send_view(s, Dim(dim), halo, tile_id)

@inline function get_recv_view(::Side{1}, ::Dim{D}, halo::ThreadedHaloArray, tile_id::Integer) where {D}
    return get_recv_view(Side(1), Dim(D), tile_parent(halo, tile_id), halo_width(halo))
end

@inline function get_recv_view(::Side{2}, ::Dim{D}, halo::ThreadedHaloArray, tile_id::Integer) where {D}
    return get_recv_view(Side(2), Dim(D), tile_parent(halo, tile_id), halo_width(halo))
end

@inline get_recv_view(s::Side, dim::Int, halo::ThreadedHaloArray, tile_id::Integer) =
    get_recv_view(s, Dim(dim), halo, tile_id)

@inline function neighbor_tile_id(halo::ThreadedHaloArray, tile_id::Integer, dim::Integer, side::Integer)
    return neighbor_tile_id(halo.topology, tile_id, dim, side)
end

function halo_exchange!(halo::ThreadedHaloArray)
    @tasks for tile_id in eachindex(parent(halo))
        @inbounds for dim in 1:ndims(halo), side in 1:2
            neighbor_id = neighbor_tile_id(halo, tile_id, dim, side)
            if neighbor_id != 0
                recv_view = get_recv_view(Side(side), dim, halo, tile_id)
                send_view = get_send_view(Side(3 - side), dim, halo, neighbor_id)
                copyto!(recv_view, send_view)
            end
        end
    end
    return halo
end

function boundary_condition!(halo::ThreadedHaloArray, tile_id::Integer, side::Side{S}, dim::Dim{D}) where {S,D}
    if neighbor_tile_id(halo, tile_id, D, S) == 0
        mode = halo.boundary_condition[D][S]
        boundary_condition!(halo, tile_id, side, dim, mode)
    end
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
    @tasks for tile_id in eachindex(parent(halo))
        ntuple(Val(ndims(halo))) do D
            ntuple(Val(2)) do S
                boundary_condition!(halo, tile_id, Side(S), Dim(D))
            end
        end
    end
    return nothing
end

function synchronize_halo!(halo::ThreadedHaloArray)
    halo_exchange!(halo)
    boundary_condition!(halo)
    return halo
end

start_halo_exchange!(halo::ThreadedHaloArray) = halo_exchange!(halo)
finish_halo_exchange!(halo::ThreadedHaloArray) = halo

function fill_interior(halo::ThreadedHaloArray, value)
    @tasks for tile_id in eachindex(parent(halo))
        fill!(interior_view(halo, tile_id), value)
    end
    return halo
end

function Base.fill!(halo::ThreadedHaloArray, value)
    @tasks for tile in parent(halo)
        fill!(tile, value)
    end
    return halo
end

function Base.similar(halo::ThreadedHaloArray{T,N,A,Halo,B,BCondition}, element_type=eltype(halo),
        tile_dims::NTuple{M,Int}=tile_size(halo)) where {T,N,A,Halo,B,BCondition,M}
    M == N || throw(DimensionMismatch("ThreadedHaloArray similar tile dims must have $N dimensions"))
    full_tile_size = ntuple(d -> tile_dims[d] + 2 * halo_width(halo), Val(N))
    data = [similar(tile_parent(halo, 1), element_type, full_tile_size) for _ in 1:tile_count(halo)]
    return ThreadedHaloArray{element_type,N,typeof(data[1]),Halo,typeof(halo.topology),typeof(halo.boundary_condition)}(
        data, tile_dims, halo.topology, halo.boundary_condition,
    )
end

function Base.zero(halo::ThreadedHaloArray)
    z = similar(halo)
    fill!(z, zero(eltype(halo)))
    return z
end

function Base.copy(halo::ThreadedHaloArray)
    copied = similar(halo)
    @tasks for tile_id in eachindex(parent(halo))
        copyto!(tile_parent(copied, tile_id), tile_parent(halo, tile_id))
    end
    return copied
end
