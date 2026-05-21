using Printf
using DifferentialEquations
using HaloArrays
using OhMyThreads: @tasks

const ALPHA = 1.0
const CFL = 0.2

function stable_heat_dt(alpha, cfl, dx)
    return cfl / (alpha * sum(inv(abs2(d)) for d in dx))
end

function initial_condition(I, global_dims)
    center = ntuple(d -> (global_dims[d] + 1) / 2, Val(length(global_dims)))
    widths = ntuple(d -> global_dims[d] / 10, Val(length(global_dims)))
    exponent = sum(((I[d] - center[d]) / widths[d])^2 for d in eachindex(I))
    return 1.0 + exp(-exponent)
end

function fill_initial_condition!(u)
    gsize = global_size(u)
    for I in CartesianIndices(axes(u))
        u[Tuple(I)...] = initial_condition(Tuple(I), gsize)
    end
    synchronize_halo!(u)
    return u
end

function heat_rhs!(du::LocalHaloArray, u::LocalHaloArray, p, t)
    alpha, dx = p
    synchronize_halo!(u)

    u_data = parent(u)
    du_data = parent(du)
    offsets = CartesianIndex.(versors(Val(ndims(u))))

    @inbounds for I in CartesianIndices(interior_range(u))
        laplacian = zero(eltype(u))
        for dim in 1:ndims(u)
            offset = offsets[dim]
            laplacian += (u_data[I + offset] - 2 * u_data[I] + u_data[I - offset]) / dx[dim]^2
        end
        du_data[I] = alpha * laplacian
    end
    return du
end

function heat_rhs!(du::ThreadedHaloArray, u::ThreadedHaloArray, p, t)
    alpha, dx = p
    synchronize_halo!(u)
    offsets = CartesianIndex.(versors(Val(ndims(u))))

    @tasks for tile_id in 1:tile_count(u)
        u_data = tile_parent(u, tile_id)
        du_data = tile_parent(du, tile_id)

        @inbounds for I in CartesianIndices(interior_range(u, tile_id))
            laplacian = zero(eltype(u))
            for dim in 1:ndims(u)
                offset = offsets[dim]
                laplacian += (u_data[I + offset] - 2 * u_data[I] + u_data[I - offset]) / dx[dim]^2
            end
            du_data[I] = alpha * laplacian
        end
    end
    return du
end

function solve_heat_diffeq(u0; alpha=ALPHA, cfl=CFL, domain_length=(1.0, 1.0), nt=100)
    dx = ntuple(d -> domain_length[d] / global_size(u0)[d], Val(ndims(u0)))
    dt = stable_heat_dt(alpha, cfl, dx)
    tspan = (0.0, nt * dt)
    prob = ODEProblem(heat_rhs!, u0, tspan, (alpha, dx))
    sol = solve(prob, Tsit5(); dt, adaptive=false, save_everystep=false)
    u = sol.u[end]
    synchronize_halo!(u)
    return u, dt
end

function interior_mean(u::LocalHaloArray)
    return sum(interior_view(u)) / length(u)
end

function interior_mean(u::ThreadedHaloArray)
    total = sum(tile_id -> sum(interior_view(u, tile_id)), 1:tile_count(u))
    return total / length(u)
end

function run_local_example()
    u0 = LocalHaloArray(Float64, (64, 64), 1; boundary_condition=:periodic)
    fill_initial_condition!(u0)
    return solve_heat_diffeq(u0)
end

function run_threaded_example()
    u0 = ThreadedHaloArray(Float64, (32, 32), 1; dims=(2, 2), boundary_condition=:periodic)
    fill_initial_condition!(u0)
    return solve_heat_diffeq(u0)
end

function main()
    local_u, local_dt = run_local_example()
    threaded_u, threaded_dt = run_threaded_example()

    @printf("DifferentialEquations LocalHaloArray:    size=%s, tiles=1,    dt=%.3e, final mean=%.12f\n",
        string(size(local_u)), local_dt, interior_mean(local_u))
    @printf("DifferentialEquations ThreadedHaloArray: size=%s, tiles=%d, dt=%.3e, final mean=%.12f\n",
        string(size(threaded_u)), tile_count(threaded_u), threaded_dt, interior_mean(threaded_u))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
