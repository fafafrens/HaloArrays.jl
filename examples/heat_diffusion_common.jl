using HaloArrays

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
        copyto!(u, current)
        synchronize_halo!(u)
    end
    return u
end
