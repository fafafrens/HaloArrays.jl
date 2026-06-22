using HaloArrays
using Printf

burgers_flux(u) = 0.5 * u^2

function rusanov_flux(ul, ur)
    wavespeed = max(abs(ul), abs(ur))
    return 0.5 * (burgers_flux(ul) + burgers_flux(ur)) - 0.5 * wavespeed * (ur - ul)
end

function _accumulate_burgers_flux!(du_data, u_data, ranges::FaceRanges, dx)
    invdx = inv(dx)
    offset = unit_vector(ranges, 1)

    @inbounds for IL in left_face(ranges, 1)
        IR = IL + offset
        du_data[IR] += rusanov_flux(u_data[IL], u_data[IR]) * invdx
    end

    @inbounds for IL in internal_face(ranges, 1)
        IR = IL + offset
        flux = rusanov_flux(u_data[IL], u_data[IR]) * invdx
        du_data[IL] -= flux
        du_data[IR] += flux
    end

    @inbounds for IL in right_face(ranges, 1)
        IR = IL + offset
        du_data[IL] -= rusanov_flux(u_data[IL], u_data[IR]) * invdx
    end

    return du_data
end

function burgers_rhs!(du::LocalHaloArray, u::LocalHaloArray, dx)
    fill!(du, 0)
    synchronize_halo!(u)
    _accumulate_burgers_flux!(parent(du), parent(u), FaceRanges(u), dx)
    return du
end

function burgers_rhs!(du::ThreadedHaloArray, u::ThreadedHaloArray, dx)
    fill!(du, 0)
    synchronize_halo!(u)
    ranges = FaceRanges(u)

    for tile_id in 1:tile_count(u)
        _accumulate_burgers_flux!(tile_parent(du, tile_id), tile_parent(u, tile_id), ranges, dx)
    end

    return du
end

function _euler_update!(u_next, u, du, dt)
    data_next = parent(u_next)
    data = parent(u)
    ddata = parent(du)

    @inbounds for I in CartesianIndices(interior_range(u))
        data_next[I] = data[I] + dt * ddata[I]
    end

    return u_next
end

function _euler_update!(u_next::ThreadedHaloArray, u::ThreadedHaloArray, du::ThreadedHaloArray, dt)
    range = interior_range(u)

    for tile_id in 1:tile_count(u)
        data_next = tile_parent(u_next, tile_id)
        data = tile_parent(u, tile_id)
        ddata = tile_parent(du, tile_id)

        @inbounds for I in CartesianIndices(range)
            data_next[I] = data[I] + dt * ddata[I]
        end
    end

    return u_next
end

function finite_volume_step!(u_next, u, du, dt, dx)
    burgers_rhs!(du, u, dx)
    _euler_update!(u_next, u, du, dt)
    return u_next
end

function fill_burgers_initial_condition!(u)
    nx = global_size(u)[1]

    for I in CartesianIndices(axes(u))
        x = (I[1] - 0.5) / nx
        u[Tuple(I)...] = 0.5 + exp(-100 * (x - 0.35)^2)
    end

    synchronize_halo!(u)
    return u
end

function solve_burgers!(u; steps=300, cfl=0.4)
    dx = 1 / global_size(u)[1]
    dt = cfl * dx / 1.5
    u_next = similar(u)
    du = similar(u)

    current = u
    next = u_next

    for _ in 1:steps
        finite_volume_step!(next, current, du, dt, dx)
        current, next = next, current
    end

    if current !== u
        copyto!(u, current)
    end
    synchronize_halo!(u)

    return (; dx, dt, time=steps * dt)
end

function run_local_burgers(; nx=200, steps=300, cfl=0.4)
    u = LocalHaloArray(Float64, (nx,), 1; boundary_condition=:periodic)
    fill_burgers_initial_condition!(u)
    initial_mass = sum(u)
    info = solve_burgers!(u; steps, cfl)
    final_mass = sum(u)
    return u, info, initial_mass, final_mass
end

function run_threaded_burgers(; nx=200, tile_dims=(4,), steps=300, cfl=0.4)
    nx % tile_dims[1] == 0 ||
        throw(ArgumentError("nx=$nx must be divisible by tile_dims[1]=$(tile_dims[1])"))

    tile_size = (nx ÷ tile_dims[1],)
    u = ThreadedHaloArray(Float64, tile_size, 1; dims=tile_dims, boundary_condition=:periodic)
    fill_burgers_initial_condition!(u)
    initial_mass = sum(u)
    info = solve_burgers!(u; steps, cfl)
    final_mass = sum(u)
    return u, info, initial_mass, final_mass
end

function print_summary(label, u, info, initial_mass, final_mass)
    @printf("%-22s nx=%d dt=%.3e time=%.3f max=%.6f mass_error=%.3e\n",
        label,
        global_size(u)[1],
        info.dt,
        info.time,
        maximum(u),
        final_mass - initial_mass)
end

function main()
    nx = 200
    steps = 300
    cfl = 0.4

    local_u, local_info, local_m0, local_m1 = run_local_burgers(; nx, steps, cfl)
    print_summary("LocalHaloArray", local_u, local_info, local_m0, local_m1)

    threaded_u, threaded_info, threaded_m0, threaded_m1 =
        run_threaded_burgers(; nx, tile_dims=(4,), steps, cfl)
    print_summary("ThreadedHaloArray", threaded_u, threaded_info, threaded_m0, threaded_m1)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
