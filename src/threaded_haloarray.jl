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

@inline is_active(::ThreadedCartesianTopology) = true
# Base.ndims inherited from AbstractCartesianTopology{N}
@inline is_root(::ThreadedCartesianTopology; root::Integer=0) = true
@inline tile_count(topology::ThreadedCartesianTopology) = prod(topology.dims)
@inline tile_coordinates(topology::ThreadedCartesianTopology, tile_id::Integer) = topology.tile_coords[tile_id]
"""
    neighbor_tile_id(x, tile_id, dim, side) -> Int

The id of the tile adjacent to tile `tile_id` along dimension `dim` on the given
`side` (`1` = lower, `2` = upper), or `0` when there is no neighbor (a
non-periodic domain edge). Used to route shared-memory halo copies in a
[`ThreadedHaloArray`](@ref); `x` may be the threaded topology, the threaded halo
array, or an array of fields sharing one topology.
"""
@inline neighbor_tile_id(topology::ThreadedCartesianTopology, tile_id::Integer, dim::Integer, side::Integer) =
    topology.neighbors[tile_id][dim][side]

# validate_boundary_condition is inherited from AbstractCartesianTopology (abstract_haloarray.jl)

"""
    ThreadedHaloArray(T, tile_size, halo; dims, boundary_condition=:repeating,
                      thread_backend=OhMyThreadsBackend())

A shared-memory halo array: the global grid is split into a `dims` layout of
rectangular tiles, each stored as its own padded array with `halo` ghost cells.
Tiles exchange halos in memory (no MPI), so after [`synchronize_halo!`](@ref)
each tile can be updated independently on its own thread.

# Arguments
- `T`: element type (defaults to `Float64` if omitted).
- `tile_size::NTuple{N,Int}`: interior cells per tile in each dimension.
- `halo::Int`: ghost-cell width on each side.
- `dims::NTuple{N,Int}`: number of tiles in each dimension (the tile layout).
  Defaults to `(1, …, Threads.nthreads())` — one tile per thread along the last
  dimension. The global interior size is `tile_size .* dims`.
- `boundary_condition`: applied at the physical domain edges (see
  [`LocalHaloArray`](@ref) for the accepted forms).
- `thread_backend::`[`ThreadBackend`](@ref): how per-tile work is dispatched
  ([`OhMyThreadsBackend`](@ref) default, [`SerialBackend`](@ref),
  [`PolyesterBackend`](@ref)). Retrieve it with [`thread_backend`](@ref).

Work on a tile with [`tile_parent`](@ref)`(u, tile_id)` (a plain padded array)
over the shared [`interior_range`](@ref)`(u)`; iterate `1:`[`tile_count`](@ref)`(u)`.

# Examples
```julia
u = ThreadedHaloArray(Float64, (32, 32), 1; dims=(2, 2), boundary_condition=:periodic)
synchronize_halo!(u)
```
"""
struct ThreadedHaloArray{T,N,A,Halo,Topo,BCondition,TB<:ThreadBackend} <: AbstractSerialHaloArray{T,N}
    data::Vector{A}
    tile_size::NTuple{N,Int}
    topology::Topo
    boundary_condition::BCondition
    backend::TB
end

@inline halo_backend(::Type{<:ThreadedHaloArray}) = ThreadedHaloBackend()

"""
    thread_backend(u::ThreadedHaloArray) -> ThreadBackend

The thread-execution backend that dispatches this array's per-tile work.
"""
@inline thread_backend(halo::ThreadedHaloArray) = halo.backend

function ThreadedHaloArray(::Type{T}, tile_size::NTuple{N,<:Integer}, halo::Integer;
        dims::NTuple{N,<:Integer} = ntuple(d -> d == N ? Threads.nthreads() : 1, Val(N)),
        boundary_condition=:repeating,
        thread_backend::ThreadBackend=OhMyThreadsBackend()) where {T,N}
    halo_int = Int(halo)
    halo_int >= 0 || throw(ArgumentError("halo width must be non-negative"))

    owned_tile_size = ntuple(d -> Int(tile_size[d]), Val(N))
    all(d -> d > 0, owned_tile_size) || throw(ArgumentError("tile_size entries must be positive"))
    all(d -> owned_tile_size[d] >= halo_int, 1:N) ||
        throw(ArgumentError("each tile_size entry must be at least the halo width"))

    _require_thread_backend(thread_backend)
    bc = normalize_boundary_condition(boundary_condition, N)
    topology = ThreadedCartesianTopology(dims; periodic=infer_periodicity(bc))
    validate_boundary_condition(topology, bc)
    full_tile_size = ntuple(d -> owned_tile_size[d] + 2 * halo_int, Val(N))
    data = [zeros(T, full_tile_size...) for _ in 1:tile_count(topology)]

    return ThreadedHaloArray{T,N,typeof(data[1]),halo_int,typeof(topology),typeof(bc),typeof(thread_backend)}(
        data, owned_tile_size, topology, bc, thread_backend,
    )
end

ThreadedHaloArray(tile_size::NTuple{N,<:Integer}, halo::Integer; kwargs...) where {N} =
    ThreadedHaloArray(Float64, tile_size, halo; kwargs...)

# eltype/ndims come from AbstractArray{T,N}; parent from AbstractSingleHaloArray.
@inline interior_size(halo::ThreadedHaloArray) = ntuple(d -> halo.tile_size[d] * halo.topology.dims[d], Val(ndims(halo)))
# size, axes, length inherited from AbstractSingleHaloArray
# interior_axes uses interior_size (tiles have no single interior_view without tile_id)
@inline interior_axes(halo::ThreadedHaloArray)         = map(Base.OneTo, interior_size(halo))
@inline interior_axes(halo::ThreadedHaloArray, d::Int) = Base.OneTo(interior_size(halo, d))
# eachindex/iterate use global CartesianIndices (interior_view without tile_id was removed).
# iterate must yield the *values* at those indices, not the indices — the
# AbstractSingleHaloArray fallback (interior_view-based) doesn't apply to tiles.
@inline Base.eachindex(halo::ThreadedHaloArray) = CartesianIndices(axes(halo))
@inline function Base.iterate(halo::ThreadedHaloArray,
        state = (eachindex(halo),))
    it = length(state) == 1 ? iterate(state[1]) : iterate(state[1], state[2])
    it === nothing && return nothing
    I, s = it
    return (@inbounds halo[I]), (state[1], s)
end
@inline halo_width(::Type{<:ThreadedHaloArray{T,N,A,Halo}}) where {T,N,A,Halo} = Halo
@inline halo_width(::ThreadedHaloArray{T,N,A,Halo}) where {T,N,A,Halo} = Halo
"""    tile_size(u) -> dims

Interior (ghost-free) cells per tile, in each dimension. The global interior size is
`tile_size(u) .* dims` (the tile layout)."""
@inline tile_size(halo::ThreadedHaloArray) = halo.tile_size

"""    tile_count(u) -> Int

Number of tiles (`prod(dims)`). Loop a parallel sweep over `1:tile_count(u)`."""
@inline tile_count(halo::ThreadedHaloArray) = length(parent(halo))

"""    tile_parent(u, tile_id) -> Array

The raw, ghost-padded storage array for tile `tile_id`. This is what you read
and write inside a tile loop, indexed over the shared [`interior_range`](@ref)`(u)`
(with ghost-safe stencil offsets)."""
@inline tile_parent(halo::ThreadedHaloArray, tile_id::Integer) = parent(halo)[tile_id]

"""    tile_coordinates(u, tile_id) -> NTuple

The Cartesian position of tile `tile_id` within the tile layout (`dims`)."""
@inline tile_coordinates(halo::ThreadedHaloArray, tile_id::Integer) = tile_coordinates(halo.topology, tile_id)

"""
    interior_to_global_index(u::ThreadedHaloArray, tile_id, interior_idx) -> NTuple

Map a tile-local interior index `interior_idx` — 1-based, excluding the
halo — in tile `tile_id` to its global index in the full domain. Mirrors
`interior_to_global_index` on [`HaloArray`](@ref) and [`LocalHaloArray`](@ref),
with the extra `tile_id` since the interior region is per tile.
"""
@inline function interior_to_global_index(halo::ThreadedHaloArray{T,N},
        tile_id::Integer, owned_idx::NTuple{N,<:Integer}) where {T,N}
    coord = tile_coordinates(halo, tile_id)
    ts    = tile_size(halo)
    all(d -> 1 <= owned_idx[d] <= ts[d], 1:N) ||
        throw(BoundsError(halo, owned_idx))
    return ntuple(d -> (coord[d] - 1) * ts[d] + owned_idx[d], Val(N))
end
# Global CartesianIndex of the first ghost cell of tile `tile_id` on the (side,dim)
# face — ghost analog of `interior_to_global_index`, per tile (see the HaloArray
# version for the index convention).
@inline function ghost_origin(halo::ThreadedHaloArray{T,N}, ::Side{S}, ::Dim{D},
        tile_id::Integer) where {T,N,S,D}
    coord = tile_coordinates(halo, tile_id)
    ts    = tile_size(halo)
    hw    = halo_width(halo)
    CartesianIndex(ntuple(i -> (coord[i] - 1) * ts[i] +
        (i == D ? (S == 1 ? 1 - hw : ts[i] + 1) : 1), Val(N)))
end
@inline global_size(halo::ThreadedHaloArray) = interior_size(halo)
@inline is_root(halo::ThreadedHaloArray; root::Integer=0) = is_root(halo.topology; root=root)
# is_active, communicator inherited from AbstractSerialHaloArray

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

@inline function full_view(halo::ThreadedHaloArray, tile_id::Integer)
    ranges = full_range(halo, tile_id)
    @views return tile_parent(halo, tile_id)[ranges...]
end

@inline function edge_view(halo::ThreadedHaloArray, s::Side, ::Dim{D}, tile_id::Int) where {D}
    return edge_view(tile_parent(halo, tile_id), s, Dim(D), halo_width(halo))
end

@inline function ghost_view(halo::ThreadedHaloArray, s::Side, ::Dim{D}, tile_id::Int) where {D}
    return ghost_view(tile_parent(halo, tile_id), s, Dim(D), halo_width(halo))
end

@inline function neighbor_tile_id(halo::ThreadedHaloArray, tile_id::Integer, dim::Integer, side::Integer)
    return neighbor_tile_id(halo.topology, tile_id, dim, side)
end

@inline function _threaded_exchange_side!(halo::ThreadedHaloArray, tile_id::Integer,
        side::Side{S}, dim::Dim{D}) where {D,S}
    neighbor_id = neighbor_tile_id(halo, tile_id, D, S)
    if neighbor_id != 0
        _threaded_copy_side!(halo, tile_id, neighbor_id, side, dim)
    end
    return halo
end

@inline function _threaded_copy_side!(halo::ThreadedHaloArray, tile_id::Integer, neighbor_id::Integer,
        side::Side{S}, dim::Dim{D}) where {D,S}
    recv_view = ghost_view(halo, side, dim, tile_id)
    send_view = edge_view(halo, Side(3 - S), dim, neighbor_id)
    copyto!(recv_view, send_view)
    return halo
end

@inline function _threaded_boundary_side!(halo::ThreadedHaloArray, tile_id::Integer,
        side::Side{S}, dim::Dim{D}) where {D,S}
    if neighbor_tile_id(halo, tile_id, D, S) == 0
        mode = halo.boundary_condition[D][S]
        boundary_condition!(halo, tile_id, side, dim, mode)
    end
    return halo
end

@inline function _threaded_synchronize_side!(halo::ThreadedHaloArray, tile_id::Integer,
        side::Side{S}, dim::Dim{D}) where {D,S}
    neighbor_id = neighbor_tile_id(halo, tile_id, D, S)
    if neighbor_id != 0
        _threaded_copy_side!(halo, tile_id, neighbor_id, side, dim)
    else
        mode = halo.boundary_condition[D][S]
        boundary_condition!(halo, tile_id, side, dim, mode)
    end
    return halo
end

# Per-tile work: iterate every (Side, Dim) face via the shared `_foreach_face`
# primitive (closure-free, allocation-safe), passing the tile id through.
@inline _threaded_exchange_tile!(halo::ThreadedHaloArray{T,N}, tile_id::Integer) where {T,N} =
    _foreach_face(_threaded_exchange_side!, halo, tile_id, Val(N))

@inline _threaded_boundary_tile!(halo::ThreadedHaloArray{T,N}, tile_id::Integer) where {T,N} =
    _foreach_face(_threaded_boundary_side!, halo, tile_id, Val(N))

@inline _threaded_synchronize_tile!(halo::ThreadedHaloArray{T,N}, tile_id::Integer) where {T,N} =
    _foreach_face(_threaded_synchronize_side!, halo, tile_id, Val(N))

function halo_exchange!(halo::ThreadedHaloArray)
    @inbounds for tile_id in eachindex(parent(halo))
        _threaded_exchange_tile!(halo, tile_id)
    end
    return halo
end

"""
    halo_exchange_threads!(u)
    boundary_condition_threads!(u)
    synchronize_halo_threads!(u)

Threaded variants of [`halo_exchange!`](@ref) / [`boundary_condition!`](@ref) /
[`synchronize_halo!`](@ref) for a [`ThreadedHaloArray`](@ref): the per-tile work
runs in parallel through the array's [`thread_backend`](@ref). The default
(non-`_threads!`) versions are a serial tile loop, which is allocation-free and
usually faster for small halo surfaces — reach for these only when benchmarking
shows the parallel exchange wins (large surfaces, wide halos, many tiles).
"""
function halo_exchange_threads!(halo::ThreadedHaloArray)
    tile_foreach(thread_backend(halo), tile_id -> _threaded_exchange_tile!(halo, tile_id),
        eachindex(parent(halo)); scheduler=:static)
    return halo
end

function boundary_condition!(halo::ThreadedHaloArray, tile_id::Integer, side::Side{S}, dim::Dim{D}) where {S,D}
    _threaded_boundary_side!(halo, tile_id, side, dim)
    return halo
end

# Per-tile BC: delegate to the shared ghost-fill kernels (boundary.jl) with the
# tile-aware views — the same code the single-array backends use, only the views
# differ. Periodic is a no-op (the inter-tile exchange already wraps the edges).
@inline boundary_condition!(h::ThreadedHaloArray, t::Integer, s::Side, d::Dim, ::Reflecting) =
    _reflect_into!(ghost_view(h, s, d, t), edge_view(h, s, d, t), d, 1)
@inline boundary_condition!(h::ThreadedHaloArray, t::Integer, s::Side, d::Dim, ::Antireflecting) =
    _reflect_into!(ghost_view(h, s, d, t), edge_view(h, s, d, t), d, -1)
@inline boundary_condition!(h::ThreadedHaloArray, t::Integer, s::Side, d::Dim, ::Repeating) =
    _repeating_into!(ghost_view(h, s, d, t), edge_view(h, s, d, t), s, d)
boundary_condition!(::ThreadedHaloArray, ::Integer, ::Side, ::Dim, ::Periodic) = nothing
# FunctionBC: tile-local views + per-tile global origin, same closure as the single
# backends (see haloarray.jl `FunctionBC`).
@inline boundary_condition!(h::ThreadedHaloArray, t::Integer, s::Side, d::Dim, bc::FunctionBC) =
    bc.f(ghost_view(h, s, d, t), edge_view(h, s, d, t), s, d, halo_width(h), ghost_origin(h, s, d, t))

function boundary_condition!(halo::ThreadedHaloArray)
    @inbounds for tile_id in eachindex(parent(halo))
        _threaded_boundary_tile!(halo, tile_id)
    end
    return halo
end

function boundary_condition_threads!(halo::ThreadedHaloArray)
    tile_foreach(thread_backend(halo), tile_id -> _threaded_boundary_tile!(halo, tile_id),
        eachindex(parent(halo)); scheduler=:static)
    return halo
end

function synchronize_halo!(halo::ThreadedHaloArray)
    @inbounds for tile_id in eachindex(parent(halo))
        _threaded_synchronize_tile!(halo, tile_id)
    end
    return halo
end

function synchronize_halo_threads!(halo::ThreadedHaloArray)
    tile_foreach(thread_backend(halo), tile_id -> _threaded_synchronize_tile!(halo, tile_id),
        eachindex(parent(halo)); scheduler=:static)
    return halo
end

@doc (@doc halo_exchange_threads!) boundary_condition_threads!
@doc (@doc halo_exchange_threads!) synchronize_halo_threads!

start_halo_exchange!(halo::ThreadedHaloArray) = halo_exchange!(halo)
finish_halo_exchange!(halo::ThreadedHaloArray) = halo

# interior-only, like the generic fill!; per-tile, in parallel
function Base.fill!(halo::ThreadedHaloArray, value)
    tile_foreach(thread_backend(halo), tile_id -> _fill_threaded_interior_tile!(halo, tile_id, value),
        eachindex(parent(halo)); scheduler=:static)
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

function Base.similar(halo::ThreadedHaloArray{T,N,A,Halo,Topo,BCondition}, ::Type{AA},
        dims::Dims{M}) where {T,N,A,Halo,Topo,BCondition,AA,M}
    tile_dims = _global_to_tile_dims(halo, dims)
    full_tile_size = ntuple(d -> tile_dims[d] + 2 * halo_width(halo), Val(N))
    data = [similar(tile_parent(halo, 1), AA, full_tile_size) for _ in 1:tile_count(halo)]
    return ThreadedHaloArray{AA,N,typeof(data[1]),Halo,typeof(halo.topology),typeof(halo.boundary_condition),typeof(halo.backend)}(
        data, tile_dims, halo.topology, halo.boundary_condition, halo.backend,
    )
end

# Non-Int dims are normalized to Dims by Base's generic similar fallbacks; a
# direct NTuple{M,<:Integer} method here would be ambiguous against them.

# Base.similar dispatchers and Base.zero inherited from AbstractSingleHaloArray

function Base.foreach(f, halo::ThreadedHaloArray)
    for tile_id in 1:tile_count(halo)
        foreach(f, interior_view(halo, tile_id))
    end
    return nothing
end

function Base.copyto!(dest::ThreadedHaloArray, src::ThreadedHaloArray)
    size(dest) == size(src) || throw(DimensionMismatch("ThreadedHaloArray copyto! requires matching global sizes"))
    tile_size(dest) == tile_size(src) || throw(DimensionMismatch("ThreadedHaloArray copyto! requires matching tile sizes"))
    tile_count(dest) == tile_count(src) || throw(DimensionMismatch("ThreadedHaloArray copyto! requires matching tile counts"))

    tile_foreach(thread_backend(dest), tile_id -> _copy_threaded_tile_storage!(dest, src, tile_id),
        eachindex(parent(dest)); scheduler=:static)
    return dest
end

@inline function _copy_threaded_tile_storage!(dest::ThreadedHaloArray, src::ThreadedHaloArray, tile_id::Integer)
    copyto!(tile_parent(dest, tile_id), tile_parent(src, tile_id))
    return nothing
end

# ---- show -------------------------------------------------------------
#
# ThreadedHaloArray is an AbstractArray, but its storage is a vector of
# per-tile arrays rather than one flat global buffer. The generic
# AbstractArray printer assumes flat global indexing and errors, so we
# provide our own show (mirroring HaloArray / LocalHaloArray).

function Base.show(io::IO, obj::ThreadedHaloArray)
    print(io, "ThreadedHaloArray{", eltype(obj), ",", ndims(obj), "}(global ",
          global_size(obj), ", ", tile_count(obj), " tiles of ", tile_size(obj),
          ", halo=", halo_width(obj), ")")
end

function Base.show(io::IO, mime::MIME"text/plain", obj::ThreadedHaloArray)
    nt = tile_count(obj)
    println(io, "ThreadedHaloArray{", eltype(obj), ", ", ndims(obj), "}")
    println(io, "  global size : ", global_size(obj))
    println(io, "  tiles       : ", nt, " (layout ", obj.topology.dims,
            "), tile_size=", tile_size(obj))
    println(io, "  storage/tile: ", storage_size(obj), " (halo=", halo_width(obj), ")")
    println(io, "  boundary    : ", obj.boundary_condition)
    maxtiles = 4
    println(io, "  interior data by tile:")
    for tile_id in 1:min(nt, maxtiles)
        print(io, "    tile ", tile_id, " @ ", tile_coordinates(obj, tile_id), " => ")
        show(io, interior_view(obj, tile_id))
        println(io)
    end
    if nt > maxtiles
        print(io, "    ⋮ (", nt - maxtiles, " more tile(s))")
    end
end

# Base.copy inherited from AbstractSingleHaloArray

function fill_from_global_indices!(f, halo::ThreadedHaloArray{T,N,A,Halo}) where {T,N,A,Halo}
    range = interior_range(halo)
    ts    = tile_size(halo)
    for tile_id in 1:tile_count(halo)
        coords = tile_coordinates(halo, tile_id)
        data   = tile_parent(halo, tile_id)
        for storage_I in CartesianIndices(range)
            local_I  = ntuple(d -> storage_I[d] - Halo, Val(N))
            global_I = ntuple(d -> (coords[d] - 1) * ts[d] + local_I[d], Val(N))
            data[storage_I] = f(global_I)
        end
    end
    return halo
end
