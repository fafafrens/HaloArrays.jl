using HaloArrays
using MPI
using DiffEqBase: ODEProblem
using OrdinaryDiffEq
using OhMyThreads: tforeach, tmapreduce
using Printf

if !MPI.Initialized()
    MPI.Init()
end

initial_profile(x) = 1.0 + 0.25 * sin(2 * pi * x) + 0.1 * cos(4 * pi * x)
periodic_coordinate(x) = mod(x, 1.0)
upwind_flux(ul, ur, velocity) = velocity >= 0 ? velocity * ul : velocity * ur

function _accumulate_advection_flux!(du_data, u_data, ranges::FaceRanges, velocity, dx)
    invdx = inv(dx)
    offset = get_unit_vector(ranges, 1)

    @inbounds for IL in get_left_face(ranges, 1)
        IR = IL + offset
        du_data[IR] += upwind_flux(u_data[IL], u_data[IR], velocity) * invdx
    end

    @inbounds for IL in get_internal_face(ranges)
        IR = IL + offset
        flux = upwind_flux(u_data[IL], u_data[IR], velocity) * invdx
        du_data[IL] -= flux
        du_data[IR] += flux
    end

    @inbounds for IL in get_right_face(ranges, 1)
        IR = IL + offset
        du_data[IL] -= upwind_flux(u_data[IL], u_data[IR], velocity) * invdx
    end

    return du_data
end

function _zero_storage!(u::Union{HaloArray,LocalHaloArray})
    fill!(parent(u), zero(eltype(u)))
    return u
end

function _zero_storage!(u::ThreadedHaloArray)
    tforeach(1:tile_count(u); scheduler=:static) do tile_id
        fill!(tile_parent(u, tile_id), zero(eltype(u)))
    end
    return u
end

function _prepare_rhs!(du, u)
    start_halo_exchange!(u)
    _zero_storage!(du)
    finish_halo_exchange!(u)
    boundary_condition!(u)
    return du
end

function advection_rhs!(du::Union{HaloArray,LocalHaloArray}, u::Union{HaloArray,LocalHaloArray}, p, t)
    _prepare_rhs!(du, u)
    _accumulate_advection_flux!(parent(du), parent(u), FaceRanges(u), p.velocity, p.dx)
    return du
end

function advection_rhs!(du::ThreadedHaloArray, u::ThreadedHaloArray, p, t)
    _prepare_rhs!(du, u)
    ranges = FaceRanges(u)

    tforeach(1:tile_count(u); scheduler=:static) do tile_id
        _accumulate_advection_flux!(tile_parent(du, tile_id), tile_parent(u, tile_id), ranges,
            p.velocity, p.dx)
    end

    return du
end

function fill_advection_initial_condition!(u::Union{HaloArray,LocalHaloArray})
    nx = global_size(u)[1]

    fill_from_global_indices!(u) do I
        x = (I[1] - 0.5) / nx
        initial_profile(x)
    end

    synchronize_halo!(u)
    return u
end

function fill_advection_initial_condition!(u::ThreadedHaloArray)
    nx = global_size(u)[1]

    for I in CartesianIndices(axes(u))
        x = (I[1] - 0.5) / nx
        u[Tuple(I)...] = initial_profile(x)
    end

    synchronize_halo!(u)
    return u
end

function solve_advection_diffeq(u0; velocity=1.0, steps=200, cfl=0.4)
    dx = 1 / global_size(u0)[1]
    dt = cfl * dx / abs(velocity)
    tspan = (0.0, steps * dt)
    p = (; velocity, dx)
    prob = ODEProblem{true}(advection_rhs!, u0, tspan, p)
    sol = solve(prob, Tsit5(); dt, adaptive=false, save_everystep=false)
    u = sol.u[end]
    synchronize_halo!(u)
    return u, sol, (; velocity, dx, dt, time=last(tspan))
end

function _exact_value(global_i, nx, velocity, time)
    x = (global_i - 0.5) / nx
    return initial_profile(periodic_coordinate(x - velocity * time))
end

function max_exact_error(u::Union{HaloArray,LocalHaloArray}, velocity, time)
    h = halo_width(u)
    nx = global_size(u)[1]
    local_error = 0.0

    @inbounds for I in CartesianIndices(interior_range(u))
        storage_i = Tuple(I)
        owned_i = ntuple(d -> storage_i[d] - h, Val(ndims(u)))
        global_i = owned_to_global_index(u, owned_i)[1]
        exact = _exact_value(global_i, nx, velocity, time)
        local_error = max(local_error, abs(parent(u)[I] - exact))
    end

    return u isa HaloArray ? MPI.Allreduce(local_error, max, get_comm(u)) : local_error
end

function max_exact_error(u::ThreadedHaloArray, velocity, time)
    h = halo_width(u)
    nx = global_size(u)[1]
    owned_tile_size = tile_size(u)

    return tmapreduce(tile_id -> begin
        coord = tile_coordinates(u, tile_id)
        data = tile_parent(u, tile_id)
        tile_error = 0.0

        @inbounds for I in CartesianIndices(interior_range(u))
            storage_i = Tuple(I)
            owned_i = storage_i[1] - h
            global_i = (coord[1] - 1) * owned_tile_size[1] + owned_i
            exact = _exact_value(global_i, nx, velocity, time)
            tile_error = max(tile_error, abs(data[I] - exact))
        end

        tile_error
    end, max, 1:tile_count(u); scheduler=:static)
end

function run_local_advection_diffeq(; nx=256, velocity=1.0, steps=200, cfl=0.4)
    u0 = LocalHaloArray(Float64, (nx,), 1; boundary_condition=:periodic)
    fill_advection_initial_condition!(u0)
    initial_mass = sum(u0) / global_size(u0)[1]
    u, sol, info = solve_advection_diffeq(u0; velocity, steps, cfl)
    final_mass = info.dx * sum(u)
    exact_error = max_exact_error(u, info.velocity, info.time)
    return u, sol, info, initial_mass, final_mass, exact_error
end

function run_threaded_advection_diffeq(; nx=256, tile_dims=(4,), velocity=1.0, steps=200, cfl=0.4)
    nx % tile_dims[1] == 0 ||
        throw(ArgumentError("nx=$nx must be divisible by tile_dims[1]=$(tile_dims[1])"))

    tile_size_ = (nx ÷ tile_dims[1],)
    u0 = ThreadedHaloArray(Float64, tile_size_, 1; dims=tile_dims, boundary_condition=:periodic)
    fill_advection_initial_condition!(u0)
    initial_mass = sum(u0) / global_size(u0)[1]
    u, sol, info = solve_advection_diffeq(u0; velocity, steps, cfl)
    final_mass = info.dx * sum(u)
    exact_error = max_exact_error(u, info.velocity, info.time)
    return u, sol, info, initial_mass, final_mass, exact_error
end

function run_mpi_advection_diffeq(; nx=256, velocity=1.0, steps=200, cfl=0.4)
    comm = MPI.COMM_WORLD
    nranks = MPI.Comm_size(comm)
    nx % nranks == 0 ||
        throw(ArgumentError("nx=$nx must be divisible by MPI ranks=$nranks"))

    topology = CartesianTopology(comm, (0,); periodic=(true,))
    u0 = HaloArray(Float64, (nx ÷ nranks,), 1, topology; boundary_condition=:periodic)
    fill_advection_initial_condition!(u0)
    initial_mass = sum(u0) / global_size(u0)[1]
    u, sol, info = solve_advection_diffeq(u0; velocity, steps, cfl)
    final_mass = info.dx * sum(u)
    exact_error = max_exact_error(u, info.velocity, info.time)
    return u, sol, info, initial_mass, final_mass, exact_error
end

function print_summary(label, u, sol, info, max_value, mass_error, exact_error)
    @printf("%-32s nx=%d a=%.2f dt=%.3e time=%.3f saved=%d max=%.6f mass_error=%.3e exact_error=%.3e\n",
        label,
        global_size(u)[1],
        info.velocity,
        info.dt,
        info.time,
        length(sol.t) - 1,
        max_value,
        mass_error,
        exact_error)
end

function root_print_summary(label, u, sol, info, initial_mass, final_mass, exact_error)
    max_value = maximum(u)
    mass_error = final_mass - initial_mass
    MPI.Comm_rank(MPI.COMM_WORLD) == 0 &&
        print_summary(label, u, sol, info, max_value, mass_error, exact_error)
    return nothing
end

function main()
    nx = 256
    velocity = 1.0
    steps = 200
    cfl = 0.4

    local_u, local_sol, local_info, local_m0, local_m1, local_err =
        run_local_advection_diffeq(; nx, velocity, steps, cfl)
    root_print_summary("OrdinaryDiffEq LocalHaloArray", local_u, local_sol, local_info,
        local_m0, local_m1, local_err)

    threaded_u, threaded_sol, threaded_info, threaded_m0, threaded_m1, threaded_err =
        run_threaded_advection_diffeq(; nx, tile_dims=(4,), velocity, steps, cfl)
    root_print_summary("OrdinaryDiffEq ThreadedHaloArray", threaded_u, threaded_sol, threaded_info,
        threaded_m0, threaded_m1, threaded_err)

    mpi_u, mpi_sol, mpi_info, mpi_m0, mpi_m1, mpi_err =
        run_mpi_advection_diffeq(; nx, velocity, steps, cfl)
    root_print_summary("OrdinaryDiffEq HaloArray MPI", mpi_u, mpi_sol, mpi_info,
        mpi_m0, mpi_m1, mpi_err)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
