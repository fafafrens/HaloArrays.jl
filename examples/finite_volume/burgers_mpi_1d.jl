using HaloArrays
using MPI
using Printf

burgers_flux(u) = 0.5 * u^2

function rusanov_flux(ul, ur)
    wavespeed = max(abs(ul), abs(ur))
    return 0.5 * (burgers_flux(ul) + burgers_flux(ur)) - 0.5 * wavespeed * (ur - ul)
end

function _accumulate_burgers_flux!(du_data, u_data, ranges::FaceRanges, dx)
    invdx = inv(dx)
    offset = unit_vector(ranges, 1)

    @inbounds for IL in interior_faces(ranges, 1)
        IR = IL + offset
        flux = rusanov_flux(u_data[IL], u_data[IR]) * invdx
        du_data[IL] -= flux
        du_data[IR] += flux
    end

    return du_data
end

function burgers_rhs!(du::HaloArray, u::HaloArray, dx)
    fill!(parent(du), 0)
    synchronize_halo!(u)
    _accumulate_burgers_flux!(parent(du), parent(u), FaceRanges(u), dx)
    return du
end

function _euler_update!(u_next::HaloArray, u::HaloArray, du::HaloArray, dt)
    data_next = parent(u_next)
    data = parent(u)
    ddata = parent(du)

    @inbounds for I in CartesianIndices(interior_range(u))
        data_next[I] = data[I] + dt * ddata[I]
    end

    return u_next
end

function finite_volume_step!(u_next::HaloArray, u::HaloArray, du::HaloArray, dt, dx)
    burgers_rhs!(du, u, dx)
    _euler_update!(u_next, u, du, dt)
    return u_next
end

function fill_burgers_initial_condition!(u::HaloArray)
    nx = global_size(u)[1]

    fill_from_global_indices!(u) do I
        x = (I[1] - 0.5) / nx
        0.5 + exp(-100 * (x - 0.35)^2)
    end

    synchronize_halo!(u)
    return u
end

function solve_burgers!(u::HaloArray; steps=300, cfl=0.4)
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
        copyto!(parent(u), parent(current))
    end
    synchronize_halo!(u)

    return (; dx, dt, time=steps * dt)
end

function run_mpi_burgers(; owned_cells=100, steps=300, cfl=0.4)
    if !MPI.Initialized()
        MPI.Init()
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    topology = CartesianTopology(comm, (0,); periodic=(true,))
    u = HaloArray(Float64, (owned_cells,), 1, topology; boundary_condition=:periodic)

    fill_burgers_initial_condition!(u)
    info0 = (; dx=1 / global_size(u)[1])
    initial_mass = info0.dx * sum(u)

    info = solve_burgers!(u; steps, cfl)
    final_mass = info.dx * sum(u)
    max_value = maximum(u)

    if rank == 0
        @printf("%-22s ranks=%d owned/rank=%d nx=%d dt=%.3e time=%.3f max=%.6f mass_error=%.3e\n",
            "HaloArray MPI",
            MPI.Comm_size(comm),
            owned_cells,
            global_size(u)[1],
            info.dt,
            info.time,
            max_value,
            final_mass - initial_mass)
    end

    return u, info, initial_mass, final_mass
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_mpi_burgers()
end
