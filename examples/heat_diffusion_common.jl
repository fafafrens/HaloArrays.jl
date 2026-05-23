using HaloArrays
using OhMyThreads: @tasks

_as_tuple(x::Number, ::Val{N}) where {N} = ntuple(_ -> x, Val(N))
_as_tuple(x, ::Val{N}) where {N} = Tuple(x)

function stable_heat_dt(alpha, cfl, dx)
    dxs = dx isa Number ? (dx,) : Tuple(dx)
    return cfl / (alpha * sum(inv(abs2(d)) for d in dxs))
end

function fill_centered_gaussian!(u; baseline=1.0, amplitude=1.0, widths=nothing)
    N = ndims(u)
    gsize = global_size(u)
    center = ntuple(i -> (gsize[i] + 1) / 2, Val(N))
    widths_tuple = widths === nothing ? ntuple(i -> gsize[i] / 10, Val(N)) : Tuple(widths)

    fill_from_global_indices!(u) do I
        exponent = sum(((I[d] - center[d]) / widths_tuple[d])^2 for d in 1:N)
        return baseline + amplitude * exp(-exponent)
    end
    boundary_condition!(u)
    return u
end

function fill_centered_gaussian!(u::ThreadedHaloArray; baseline=1.0, amplitude=1.0, widths=nothing)
    N = ndims(u)
    gsize = global_size(u)
    center = ntuple(i -> (gsize[i] + 1) / 2, Val(N))
    widths_tuple = widths === nothing ? ntuple(i -> gsize[i] / 10, Val(N)) : Tuple(widths)

    for I in CartesianIndices(axes(u))
        global_index = Tuple(I)
        exponent = sum(((global_index[d] - center[d]) / widths_tuple[d])^2 for d in 1:N)
        u[global_index...] = baseline + amplitude * exp(-exponent)
    end
    synchronize_halo!(u)
    return u
end

function heat_step!(u_next, u_old, alpha, dt, dx)
    N = ndims(u_old)
    dxs = _as_tuple(dx, Val(N))
    data_old = parent(u_old)
    data_next = parent(u_next)
    offsets = CartesianIndex.(versors(u_old))

    @inbounds for I in CartesianIndices(interior_range(u_old))
        laplacian = zero(eltype(data_old))
        for dim in 1:N
            offset = offsets[dim]
            laplacian += (data_old[I + offset] - 2 * data_old[I] + data_old[I - offset]) / dxs[dim]^2
        end
        data_next[I] = data_old[I] + alpha * dt * laplacian
    end
    return u_next
end

function _heat_step_tile!(data_next, data_old, alpha, dt, dxs, offsets, range, ::Val{N}) where {N}
    @inbounds for I in CartesianIndices(range)
        laplacian = zero(eltype(data_old))
        for dim in 1:N
            offset = offsets[dim]
            laplacian += (data_old[I + offset] - 2 * data_old[I] + data_old[I - offset]) / dxs[dim]^2
        end
        data_next[I] = data_old[I] + alpha * dt * laplacian
    end
    return data_next
end

function heat_step!(u_next::ThreadedHaloArray, u_old::ThreadedHaloArray, alpha, dt, dx)
    N = ndims(u_old)
    dxs = _as_tuple(dx, Val(N))
    offsets = CartesianIndex.(versors(Val(N)))
    range = interior_range(u_old)

    @tasks for tile_id in 1:tile_count(u_old)
        data_old = tile_parent(u_old, tile_id)
        data_next = tile_parent(u_next, tile_id)
        _heat_step_tile!(data_next, data_old, alpha, dt, dxs, offsets, range, Val(N))
    end
    return u_next
end

function _copy_heat_solution!(dest, src)
    copyto!(dest, src)
    return dest
end

function solve_heat!(u; alpha, dt, dx, nt)
    current = u
    next = similar(u)

    for _ in 1:nt
        synchronize_halo!(current)
        heat_step!(next, current, alpha, dt, dx)
        current, next = next, current
    end

    synchronize_halo!(current)
    if current !== u
        _copy_heat_solution!(u, current)
        synchronize_halo!(u)
    end
    return u
end

function heat_rhs!(du, u, p, t)
    alpha, dx = p
    synchronize_halo!(u)
    heat_step!(du, u, alpha, one(eltype(u)), dx)

    if u isa ThreadedHaloArray
        @tasks for tile_id in 1:tile_count(u)
            interior_view(du, tile_id) .-= interior_view(u, tile_id)
        end
    else
        interior_view(du) .-= interior_view(u)
    end

    return du
end

function interior_mean(u)
    return sum(interior_view(u)) / length(u)
end

function interior_mean(u::ThreadedHaloArray)
    total = sum(tile_id -> sum(interior_view(u, tile_id)), 1:tile_count(u))
    return total / length(u)
end
