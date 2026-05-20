using MPI
using HaloArrays
using DiffEqBase: ODEProblem
using OrdinaryDiffEq

const ODE_RATE = 0.1

if !MPI.Initialized()
    MPI.Init()
end

function ode_rhs!(du, u, p, t)
    du .= -ODE_RATE .* u
    return du
end

initial_value(I) = 1.0 + sum(I)
root_println(args...) = MPI.Comm_rank(MPI.COMM_WORLD) == 0 && println(args...)

function initialize_example!(u::Union{HaloArray,LocalHaloArray})
    fill_from_global_indices!(u) do I
        initial_value(I)
    end
    return u
end

function initialize_example!(u::ThreadedHaloArray)
    for I in CartesianIndices(axes(u))
        u[Tuple(I)...] = initial_value(Tuple(I))
    end
    return u
end

function solve_decay(u0; tspan=(0.0, 0.5), dt=0.1)
    prob = ODEProblem(ode_rhs!, u0, tspan)
    sol = solve(prob, Tsit5(); dt, adaptive=false, save_everystep=false)
    return sol.u[end]
end

function max_interior_error(u, u0, expected_factor)
    return maximum(abs, interior_view(u) .- expected_factor .* interior_view(u0))
end

function max_interior_error(u::ThreadedHaloArray, u0::ThreadedHaloArray, expected_factor)
    return maximum(1:tile_count(u)) do tile_id
        maximum(abs, interior_view(u, tile_id) .- expected_factor .* interior_view(u0, tile_id))
    end
end

function run_local_example()
    u0 = initialize_example!(LocalHaloArray(Float64, (4, 5), 1; boundary_condition=:repeating))
    u_final = solve_decay(u0)
    expected_factor = exp(-ODE_RATE * 0.5)
    err = max_interior_error(u_final, u0, expected_factor)
    root_println("LocalHaloArray: size=$(size(u_final)), max error=$err")
    return err
end

function run_threaded_example()
    u0 = initialize_example!(ThreadedHaloArray(Float64, (3, 4), 1;
        dims=(2, 1), boundary_condition=:repeating))
    u_final = solve_decay(u0)
    expected_factor = exp(-ODE_RATE * 0.5)
    err = max_interior_error(u_final, u0, expected_factor)
    root_println("ThreadedHaloArray: size=$(size(u_final)), max error=$err")
    return err
end

function run_mpi_example()
    comm = MPI.COMM_WORLD
    nranks = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    topology = CartesianTopology(comm, (nranks,); periodic=(false,))
    u0 = initialize_example!(HaloArray(Float64, (4,), 1, topology; boundary_condition=:repeating))
    u_final = solve_decay(u0)
    expected_factor = exp(-ODE_RATE * 0.5)
    local_err = max_interior_error(u_final, u0, expected_factor)
    err = MPI.Allreduce(local_err, max, comm)
    rank == 0 && root_println("HaloArray MPI: size=$(size(u_final)), max error=$err")
    return err
end

local_err = run_local_example()
threaded_err = run_threaded_example()
mpi_err = run_mpi_example()

tol = 1.0e-10
all(err < tol for err in (local_err, threaded_err, mpi_err)) ||
    error("OrdinaryDiffEq HaloArrays example exceeded tolerance $tol")
