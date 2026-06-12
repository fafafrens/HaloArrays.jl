using Base.Threads: @threads

_as_tuple(x::Number, n::Int) = ntuple(_ -> x, n)
_as_tuple(x, ::Int) = Tuple(x)

function stable_heat_dt(alpha, cfl, dx)
    dxs = dx isa Number ? (dx,) : Tuple(dx)
    return cfl / (alpha * sum(inv(abs2(d)) for d in dxs))
end

function problem_size_from_topology(interior_size, topology_dims)
    return ntuple(d -> interior_size[d] * topology_dims[d], length(interior_size))
end

function heat_initial_value(I, problem_size)
    center = ntuple(d -> (problem_size[d] + 1) / 2, length(problem_size))
    widths = ntuple(d -> problem_size[d] / 10, length(problem_size))
    exponent = sum(((I[d] - center[d]) / widths[d])^2 for d in eachindex(I))
    return exp(-exponent)
end

@inline function _heat_step_data!(dest_data, src_data, alpha, dt, dxs, offsets, range, ::Val{N}) where {N}
    @inbounds for I in CartesianIndices(range)
        laplacian = zero(eltype(src_data))
        for dim in 1:N
            offset = offsets[dim]
            laplacian += (src_data[I + offset] - 2 * src_data[I] + src_data[I - offset]) / dxs[dim]^2
        end
        dest_data[I] = src_data[I] + alpha * dt * laplacian
    end
    return dest_data
end

function fill_heat_initial!(halo::Union{HaloArray,LocalHaloArray}, problem_size)
    fill!(parent(halo), 0.0)
    fill_from_global_indices!(halo) do I
        heat_initial_value(I, problem_size)
    end
    synchronize_halo!(halo)
    return halo
end

function fill_heat_initial!(halo::ThreadedHaloArray{T,N}, problem_size) where {T,N}
    fill!(halo, 0.0)
    h = halo_width(halo)
    tile_dims = tile_size(halo)

    @threads :static for tile_id in 1:tile_count(halo)
        coords = tile_coordinates(halo, tile_id)
        tile_offset = ntuple(d -> (coords[d] - 1) * tile_dims[d], Val(N))
        data = tile_parent(halo, tile_id)

        @inbounds for I in CartesianIndices(interior_range(halo, tile_id))
            global_I = ntuple(d -> tile_offset[d] + I[d] - h, Val(N))
            data[I] = heat_initial_value(global_I, problem_size)
        end
    end

    synchronize_halo!(halo)
    return halo
end

function heat_step!(dest::Union{HaloArray{T,N},LocalHaloArray{T,N}},
        src::Union{HaloArray{S,N},LocalHaloArray{S,N}}, alpha, dt, dx) where {T,S,N}
    dxs = _as_tuple(dx, N)
    offsets = CartesianIndex.(versors(Val(N)))
    src_data = parent(src)
    dest_data = parent(dest)
    _heat_step_data!(dest_data, src_data, alpha, dt, dxs, offsets, interior_range(src), Val(N))

    return dest
end

function heat_step!(dest::ThreadedHaloArray{T,N}, src::ThreadedHaloArray{S,N}, alpha, dt, dx) where {T,S,N}
    dxs = _as_tuple(dx, N)
    offsets = CartesianIndex.(versors(Val(N)))
    range = interior_range(src)
    src_tiles = parent(src)
    dest_tiles = parent(dest)

    @threads :static for tile_id in eachindex(src_tiles)
        src_data = src_tiles[tile_id]
        dest_data = dest_tiles[tile_id]
        _heat_step_data!(dest_data, src_data, alpha, dt, dxs, offsets, range, Val(N))
    end

    return dest
end

function copy_halo_storage!(dest::Union{HaloArray,LocalHaloArray}, src::Union{HaloArray,LocalHaloArray})
    copyto!(parent(dest), parent(src))
    return dest
end

function copy_halo_storage!(dest::ThreadedHaloArray, src::ThreadedHaloArray)
    @threads :static for tile_id in 1:tile_count(dest)
        copyto!(tile_parent(dest, tile_id), tile_parent(src, tile_id))
    end
    return dest
end

function solve_heat_steps!(u, scratch; steps, alpha, dt, dx)
    current = u
    next = scratch

    for _ in 1:steps
        synchronize_halo!(current)
        heat_step!(next, current, alpha, dt, dx)
        current, next = next, current
    end

    synchronize_halo!(current)
    if current !== u
        copy_halo_storage!(u, current)
        synchronize_halo!(u)
    end
    return u
end

function heat_single_step!(scratch, u; alpha, dt, dx)
    synchronize_halo!(u)
    heat_step!(scratch, u, alpha, dt, dx)
    return scratch
end

function snapshot_interior(halo::LocalHaloArray)
    return Array(interior_view(halo))
end

function snapshot_interior(halo::ThreadedHaloArray{T,N}) where {T,N}
    data = Array{T}(undef, size(halo))
    tile_dims = tile_size(halo)

    for tile_id in 1:tile_count(halo)
        coords = tile_coordinates(halo, tile_id)
        inds = ntuple(Val(N)) do d
            first_owned = (coords[d] - 1) * tile_dims[d] + 1
            last_owned = coords[d] * tile_dims[d]
            first_owned:last_owned
        end
        data[inds...] .= interior_view(halo, tile_id)
    end

    return data
end

function benchmark_case!(rows, benchmark, name, f, samples, warmups, metadata; comm=nothing, rank=0)
    times = benchmark_times!(f, samples, warmups; comm=comm)
    if rank == 0
        print_summary(name, times)
        push!(rows, benchmark_record(benchmark, name, times; metadata=copy(metadata)))
    end
    return times
end

function benchmark_case!(rows, benchmark, name, f, samples, warmups, metadata, timer::Symbol)
    times = benchmark_times!(f, samples, warmups, timer)
    print_summary(name, times)
    push!(rows, benchmark_record(benchmark, name, times; metadata=copy(metadata)))
    return times
end
