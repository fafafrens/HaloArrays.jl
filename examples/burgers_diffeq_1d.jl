using HaloArrays
using MPI
using DiffEqBase: ODEProblem
using OrdinaryDiffEq
using OhMyThreads: tforeach
using Printf

if !MPI.Initialized()
    MPI.Init()
end

burgers_flux(u) = 0.5 * u^2

function rusanov_flux(ul, ur)
    wavespeed = max(abs(ul), abs(ur))
    return 0.5 * (burgers_flux(ul) + burgers_flux(ur)) - 0.5 * wavespeed * (ur - ul)
end

function _accumulate_burgers_flux!(du_data, u_data, ranges::FaceRanges, dx)
    invdx = inv(dx)
    offset = get_unit_vector(ranges, 1)

    @inbounds for IL in get_left_face(ranges, 1)
        IR = IL + offset
        du_data[IR] += rusanov_flux(u_data[IL], u_data[IR]) * invdx
    end

    @inbounds for IL in get_internal_face(ranges)
        IR = IL + offset
        flux = rusanov_flux(u_data[IL], u_data[IR]) * invdx
        du_data[IL] -= flux
        du_data[IR] += flux
    end

    @inbounds for IL in get_right_face(ranges, 1)
        IR = IL + offset
        du_data[IL] -= rusanov_flux(u_data[IL], u_data[IR]) * invdx
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

function burgers_rhs!(du::Union{HaloArray,LocalHaloArray}, u::Union{HaloArray,LocalHaloArray}, dx, t)
    _zero_storage!(du)
    synchronize_halo!(u)
    _accumulate_burgers_flux!(parent(du), parent(u), FaceRanges(u), dx)
    return du
end

function burgers_rhs!(du::ThreadedHaloArray, u::ThreadedHaloArray, dx, t)
    _zero_storage!(du)
    synchronize_halo!(u)
    ranges = FaceRanges(u)

    tforeach(1:tile_count(u); scheduler=:static) do tile_id
        _accumulate_burgers_flux!(tile_parent(du, tile_id), tile_parent(u, tile_id), ranges, dx)
    end

    return du
end

function fill_burgers_initial_condition!(u::Union{HaloArray,LocalHaloArray})
    nx = global_size(u)[1]

    fill_from_global_indices!(u) do I
        x = (I[1] - 0.5) / nx
        0.5 + exp(-100 * (x - 0.35)^2)
    end

    synchronize_halo!(u)
    return u
end

function fill_burgers_initial_condition!(u::ThreadedHaloArray)
    nx = global_size(u)[1]

    for I in CartesianIndices(axes(u))
        x = (I[1] - 0.5) / nx
        u[Tuple(I)...] = 0.5 + exp(-100 * (x - 0.35)^2)
    end

    synchronize_halo!(u)
    return u
end

function solve_burgers_diffeq(u0; steps=300, cfl=0.4)
    dx = 1 / global_size(u0)[1]
    dt = cfl * dx / 1.5
    tspan = (0.0, steps * dt)
    prob = ODEProblem{true}(burgers_rhs!, u0, tspan, dx)
    sol = solve(prob, Tsit5(); dt, adaptive=false, save_everystep=false)
    u = sol.u[end]
    synchronize_halo!(u)
    return u, sol, (; dx, dt, time=last(tspan))
end

function run_local_burgers_diffeq(; nx=200, steps=300, cfl=0.4)
    u0 = LocalHaloArray(Float64, (nx,), 1; boundary_condition=:periodic)
    fill_burgers_initial_condition!(u0)
    initial_mass = sum(u0) / global_size(u0)[1]
    u, sol, info = solve_burgers_diffeq(u0; steps, cfl)
    final_mass = info.dx * sum(u)
    return u, sol, info, initial_mass, final_mass
end

function run_threaded_burgers_diffeq(; nx=200, tile_dims=(4,), steps=300, cfl=0.4)
    nx % tile_dims[1] == 0 ||
        throw(ArgumentError("nx=$nx must be divisible by tile_dims[1]=$(tile_dims[1])"))

    tile_size = (nx ÷ tile_dims[1],)
    u0 = ThreadedHaloArray(Float64, tile_size, 1; dims=tile_dims, boundary_condition=:periodic)
    fill_burgers_initial_condition!(u0)
    initial_mass = sum(u0) / global_size(u0)[1]
    u, sol, info = solve_burgers_diffeq(u0; steps, cfl)
    final_mass = info.dx * sum(u)
    return u, sol, info, initial_mass, final_mass
end

function run_mpi_burgers_diffeq(; nx=200, steps=300, cfl=0.4)
    comm = MPI.COMM_WORLD
    nranks = MPI.Comm_size(comm)
    nx % nranks == 0 ||
        throw(ArgumentError("nx=$nx must be divisible by MPI ranks=$nranks"))

    topology = CartesianTopology(comm, (0,); periodic=(true,))
    u0 = HaloArray(Float64, (nx ÷ nranks,), 1, topology; boundary_condition=:periodic)
    fill_burgers_initial_condition!(u0)
    initial_mass = sum(u0) / global_size(u0)[1]
    u, sol, info = solve_burgers_diffeq(u0; steps, cfl)
    final_mass = info.dx * sum(u)
    return u, sol, info, initial_mass, final_mass
end

function print_summary(label, u, sol, info, max_value, mass_error)
    @printf("%-28s nx=%d dt=%.3e time=%.3f saved=%d max=%.6f mass_error=%.3e\n",
        label,
        global_size(u)[1],
        info.dt,
        info.time,
        length(sol.t) - 1,
        max_value,
        mass_error)
end

function root_print_summary(label, u, sol, info, initial_mass, final_mass)
    max_value = maximum(u)
    mass_error = final_mass - initial_mass
    MPI.Comm_rank(MPI.COMM_WORLD) == 0 && print_summary(label, u, sol, info, max_value, mass_error)
    return nothing
end

function main()
    nx = 200
    steps = 300
    cfl = 0.4

    local_u, local_sol, local_info, local_m0, local_m1 =
        run_local_burgers_diffeq(; nx, steps, cfl)
    root_print_summary("OrdinaryDiffEq LocalHaloArray", local_u, local_sol, local_info, local_m0, local_m1)

    threaded_u, threaded_sol, threaded_info, threaded_m0, threaded_m1 =
        run_threaded_burgers_diffeq(; nx, tile_dims=(4,), steps, cfl)
    root_print_summary("OrdinaryDiffEq ThreadedHaloArray", threaded_u, threaded_sol, threaded_info,
        threaded_m0, threaded_m1)

    mpi_u, mpi_sol, mpi_info, mpi_m0, mpi_m1 =
        run_mpi_burgers_diffeq(; nx, steps, cfl)
    root_print_summary("OrdinaryDiffEq HaloArray MPI", mpi_u, mpi_sol, mpi_info, mpi_m0, mpi_m1)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
