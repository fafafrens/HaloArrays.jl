using HaloArrays
using DiffEqBase: ODEProblem
using MPI
using OhMyThreads: tforeach, tmapreduce
using OrdinaryDiffEq
using Printf
using StaticArrays

if !MPI.Initialized()
    MPI.Init()
end

ideal_hydro_boundary_conditions(boundary_condition) = (;
    rho=boundary_condition,
    mx=boundary_condition,
    my=boundary_condition,
    energy=boundary_condition,
)

conserved_cell(data, I) = SVector(data.rho[I], data.mx[I], data.my[I], data.energy[I])

function primitive(U, gamma)
    rho = max(U[1], eps(eltype(U)))
    vx = U[2] / rho
    vy = U[3] / rho
    kinetic = 0.5 * rho * (vx^2 + vy^2)
    pressure = max((gamma - 1) * (U[4] - kinetic), eps(eltype(U)))
    sound_speed = sqrt(gamma * pressure / rho)
    return rho, vx, vy, pressure, sound_speed
end

function euler_flux(U, dim, gamma)
    _, vx, vy, pressure, _ = primitive(U, gamma)
    rho, mx, my, energy = U

    if dim == 1
        return SVector(mx, mx * vx + pressure, my * vx, (energy + pressure) * vx)
    else
        return SVector(my, mx * vy, my * vy + pressure, (energy + pressure) * vy)
    end
end

function max_wave_speed(U, dim, gamma)
    _, vx, vy, _, sound_speed = primitive(U, gamma)
    normal_velocity = dim == 1 ? vx : vy
    return abs(normal_velocity) + sound_speed
end

function rusanov_flux(UL, UR, dim, gamma)
    speed = max(max_wave_speed(UL, dim, gamma), max_wave_speed(UR, dim, gamma))
    return 0.5 * (euler_flux(UL, dim, gamma) + euler_flux(UR, dim, gamma)) -
           0.5 * speed * (UR - UL)
end

function add_conserved!(data, I, scale, U)
    data.rho[I] += scale * U[1]
    data.mx[I] += scale * U[2]
    data.my[I] += scale * U[3]
    data.energy[I] += scale * U[4]
    return data
end

function apply_hydro_fluxes!(du_data, u_data, ranges::FaceRanges, dim, scale, gamma)
    offset = get_unit_vector(ranges, dim)

    @inbounds for IL in get_left_face(ranges, dim)
        IR = IL + offset
        flux = rusanov_flux(conserved_cell(u_data, IL), conserved_cell(u_data, IR), dim, gamma)
        add_conserved!(du_data, IR, scale, flux)
    end

    @inbounds for IL in get_internal_face(ranges)
        IR = IL + offset
        flux = rusanov_flux(conserved_cell(u_data, IL), conserved_cell(u_data, IR), dim, gamma)
        add_conserved!(du_data, IL, -scale, flux)
        add_conserved!(du_data, IR, scale, flux)
    end

    @inbounds for IL in get_right_face(ranges, dim)
        IR = IL + offset
        flux = rusanov_flux(conserved_cell(u_data, IL), conserved_cell(u_data, IR), dim, gamma)
        add_conserved!(du_data, IL, -scale, flux)
    end

    return du_data
end

function _ideal_hydro_rhs_serial!(du, u, p)
    ranges = FaceRanges(u)
    u_data = parent(u)
    du_data = parent(du)

    apply_hydro_fluxes!(du_data, u_data, ranges, 1, inv(p.dx), p.gamma)
    apply_hydro_fluxes!(du_data, u_data, ranges, 2, inv(p.dy), p.gamma)

    return du
end

function _ideal_hydro_rhs_threaded!(du, u, p)
    ranges = FaceRanges(u)

    tforeach(1:tile_count(u); scheduler=:static) do tile_id
        u_data = tile_parent(u, tile_id)
        du_data = tile_parent(du, tile_id)
        apply_hydro_fluxes!(du_data, u_data, ranges, 1, inv(p.dx), p.gamma)
        apply_hydro_fluxes!(du_data, u_data, ranges, 2, inv(p.dy), p.gamma)
    end

    return du
end

function ideal_hydro_rhs!(du, u, p, t)
    fill!(du, zero(eltype(du)))
    synchronize_halo!(u)

    if u[:rho] isa ThreadedHaloArray
        return _ideal_hydro_rhs_threaded!(du, u, p)
    else
        return _ideal_hydro_rhs_serial!(du, u, p)
    end
end

function initial_hydro_state(global_i, global_j, nx, ny, gamma)
    x = (global_i - 0.5) / nx
    y = (global_j - 0.5) / ny
    r2 = (x - 0.5)^2 + (y - 0.5)^2

    rho = 1.0 + 0.2 * exp(-80 * r2)
    vx = 0.0
    vy = 0.0
    pressure = 1.0 + exp(-80 * r2)
    energy = pressure / (gamma - 1) + 0.5 * rho * (vx^2 + vy^2)

    return rho, rho * vx, rho * vy, energy
end

function fill_pressure_bump!(u; gamma=1.4)
    if u[:rho] isa ThreadedHaloArray
        return fill_pressure_bump_threaded!(u; gamma)
    else
        return fill_pressure_bump_serial!(u; gamma)
    end
end

function fill_pressure_bump_serial!(u; gamma=1.4)
    nx, ny = global_size(u[:rho])
    h = halo_width(u[:rho])
    data = parent(u)

    @inbounds for I in CartesianIndices(interior_range(u[:rho]))
        storage_i, storage_j = Tuple(I)
        global_i, global_j = owned_to_global_index(u[:rho], (storage_i - h, storage_j - h))
        rho, mx, my, energy = initial_hydro_state(global_i, global_j, nx, ny, gamma)

        data.rho[I] = rho
        data.mx[I] = mx
        data.my[I] = my
        data.energy[I] = energy
    end

    synchronize_halo!(u)
    return u
end

function fill_pressure_bump_threaded!(u; gamma=1.4)
    nx, ny = global_size(u[:rho])
    h = halo_width(u[:rho])
    tile_cells = tile_size(u[:rho])

    tforeach(1:tile_count(u); scheduler=:static) do tile_id
        data = tile_parent(u, tile_id)
        coord = tile_coordinates(u, tile_id)

        @inbounds for I in CartesianIndices(interior_range(u[:rho]))
            storage_i, storage_j = Tuple(I)
            owned_i = storage_i - h
            owned_j = storage_j - h
            global_i = (coord[1] - 1) * tile_cells[1] + owned_i
            global_j = (coord[2] - 1) * tile_cells[2] + owned_j
            rho, mx, my, energy = initial_hydro_state(global_i, global_j, nx, ny, gamma)

            data.rho[I] = rho
            data.mx[I] = mx
            data.my[I] = my
            data.energy[I] = energy
        end
    end

    synchronize_halo!(u)
    return u
end

function _local_max_signal_speed(data, range, gamma)
    speed = 0.0

    @inbounds for I in CartesianIndices(range)
        U = conserved_cell(data, I)
        speed = max(speed, max_wave_speed(U, 1, gamma), max_wave_speed(U, 2, gamma))
    end

    return speed
end

function max_signal_speed(u, gamma)
    if u[:rho] isa ThreadedHaloArray
        range = interior_range(u[:rho])
        local_speed = tmapreduce(tile_id -> _local_max_signal_speed(tile_parent(u, tile_id), range, gamma),
            max, 1:tile_count(u); scheduler=:static)
    else
        local_speed = _local_max_signal_speed(parent(u), interior_range(u[:rho]), gamma)
    end

    return u[:rho] isa HaloArray ? MPI.Allreduce(local_speed, max, get_comm(u[:rho])) : local_speed
end

function _local_min_pressure(data, range, gamma)
    pressure_min = Inf

    @inbounds for I in CartesianIndices(range)
        _, _, _, pressure, _ = primitive(conserved_cell(data, I), gamma)
        pressure_min = min(pressure_min, pressure)
    end

    return pressure_min
end

function min_pressure(u, gamma)
    if u[:rho] isa ThreadedHaloArray
        range = interior_range(u[:rho])
        local_min = tmapreduce(tile_id -> _local_min_pressure(tile_parent(u, tile_id), range, gamma),
            min, 1:tile_count(u); scheduler=:static)
    else
        local_min = _local_min_pressure(parent(u), interior_range(u[:rho]), gamma)
    end

    return u[:rho] isa HaloArray ? MPI.Allreduce(local_min, min, get_comm(u[:rho])) : local_min
end

function hydro_diagnostics(u; gamma, dx, dy)
    cell_volume = dx * dy
    return (;
        mass=cell_volume * sum(u[:rho]),
        energy=cell_volume * sum(u[:energy]),
        min_rho=minimum(u[:rho]),
        min_pressure=min_pressure(u, gamma),
        max_speed=max_signal_speed(u, gamma),
    )
end

function solve_ideal_hydro!(
        u;
        gamma=1.4,
        cfl=0.25,
        steps=80,
        adaptive=true,
        reltol=1e-5,
        abstol=1e-7,
)
    nx, ny = global_size(u[:rho])
    dx = 1 / nx
    dy = 1 / ny
    dt = cfl * min(dx, dy) / max_signal_speed(u, gamma)
    tspan = (0.0, steps * dt)
    p = (; gamma, dx, dy)
    problem = ODEProblem{true}(ideal_hydro_rhs!, u, tspan, p)
    sol = solve(problem, Tsit5(); dt, dtmax=dt, adaptive, reltol, abstol, save_everystep=false)

    copyto!(u, sol.u[end])
    synchronize_halo!(u)

    return (;
        dx,
        dy,
        time=last(tspan),
        dt,
        method=:diffeq,
        adaptive,
        steps,
        saved_steps=length(sol.t) - 1,
    )
end

function run_ideal_hydro_2d!(u; gamma=1.4, cfl=0.25, steps=80, adaptive=true, reltol=1e-5, abstol=1e-7)
    nx, ny = global_size(u[:rho])
    dx = 1 / nx
    dy = 1 / ny

    fill_pressure_bump!(u; gamma)
    initial = hydro_diagnostics(u; gamma, dx, dy)
    info = solve_ideal_hydro!(u; gamma, cfl, steps, adaptive, reltol, abstol)
    final = hydro_diagnostics(u; gamma, dx, dy)

    return u, info, initial, final
end

is_root(u) = !(u[:rho] isa HaloArray) || MPI.Comm_rank(get_comm(u[:rho])) == 0

function print_hydro_summary(label, u, info, initial, final)
    if is_root(u)
        @printf(
            "%-28s method=%s adaptive=%s nx=%d ny=%d time=%.4f dtmax=%.3e steps=%d min_rho=%.6f min_p=%.6f max_speed=%.6f mass_error=%.3e energy_error=%.3e\n",
            label,
            String(info.method),
            string(info.adaptive),
            global_size(u[:rho])[1],
            global_size(u[:rho])[2],
            info.time,
            info.dt,
            info.steps,
            final.min_rho,
            final.min_pressure,
            final.max_speed,
            final.mass - initial.mass,
            final.energy - initial.energy,
        )
    end

    return nothing
end
