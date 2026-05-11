using Test
using MPI
using DiffEqBase
using HaloArrays

const ODE_RATE = 0.1

function _ode_topology(comm)
    dims = MPI.Dims_create(MPI.Comm_size(comm), (0, 0))
    return CartesianTopology(comm, Tuple(Int.(dims)); periodic=(false, false))
end

function _ode_initial_condition(topology)
    local_size = (4, 5)
    global_dims = local_size .* topology.dims
    u0 = HaloArray(Float64, local_size, 1, topology; boundary_condition=:repeating)

    fill_from_global_indices!(u0) do I
        x = I[1]
        y = I[2]
        return 1 + exp(-(x - global_dims[1] / 2)^2 / 5^2 - (y - global_dims[2] / 2)^2 / 7^2)
    end

    return u0
end

function _rhs!(du, u, p, t)
    interior_view(du) .= -ODE_RATE .* interior_view(u)
    return du
end

function _explicit_euler_solve(prob, dt)
    u = copy(prob.u0)
    du = zero(prob.u0)
    steps = round(Int, (last(prob.tspan) - first(prob.tspan)) / dt)
    t = first(prob.tspan)

    for _ in 1:steps
        prob.f(du, u, prob.p, t)
        interior_view(u) .+= dt .* interior_view(du)
        t += dt
    end

    return u
end

function _solve_halo_ode(comm)
    topology = _ode_topology(comm)
    u0 = _ode_initial_condition(topology)

    tspan = (0.0, 0.5)
    dt = 0.1
    prob = ODEProblem{true}(_rhs!, u0, tspan, (;))
    u_final = _explicit_euler_solve(prob, dt)

    expected_factor = (1 - ODE_RATE * dt)^round(Int, last(tspan) / dt)
    return u0, u_final, expected_factor
end

function _check_solution(u0, u_final, expected_factor)
    @test u_final isa HaloArray
    @test size(u_final) == size(u0)
    @test halo_width(u_final) == halo_width(u0)
    @test interior_view(u_final) ≈ expected_factor .* interior_view(u0)
    return nothing
end

@testset "ODEProblem HaloArray state" begin
    topology = _ode_topology(MPI.COMM_SELF)
    u0 = _ode_initial_condition(topology)

    prob = ODEProblem{true}(_rhs!, u0, (0.0, 1.0), (;))
    du = zero(u0)
    _rhs!(du, u0, prob.p, 0.0)

    @test prob.u0 === u0
    @test du isa HaloArray
    @test size(du) == size(u0)
    @test halo_width(du) == halo_width(u0)
    @test all(isfinite, interior_view(du))
    @test all(x -> x < 0, interior_view(du))
end

@testset "Explicit Euler ODE solve on one rank" begin
    u0, u_final, expected_factor = _solve_halo_ode(MPI.COMM_SELF)
    _check_solution(u0, u_final, expected_factor)
end

@testset "Explicit Euler ODE solve on MPI ranks" begin
    if MPI.Comm_size(MPI.COMM_WORLD) == 1
        @test_skip "MPI multi-rank ODE solve requires mpiexec -n > 1"
    else
        u0, u_final, expected_factor = _solve_halo_ode(MPI.COMM_WORLD)
        _check_solution(u0, u_final, expected_factor)
    end
end

@testset "DiffEq default checks use HaloArray reductions" begin
    topology = _ode_topology(MPI.COMM_WORLD)
    u = _ode_initial_condition(topology)

    unorm = DiffEqBase.ODE_DEFAULT_NORM(u, 0.0)
    unorms = MPI.Allgather(unorm, MPI.COMM_WORLD)
    @test allequal(unorms)

    w = copy(u)
    @test DiffEqBase.ODE_DEFAULT_UNSTABLE_CHECK(nothing, w, nothing, nothing) == false

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        interior_view(w)[1, 1] = NaN
    end
    @test DiffEqBase.ODE_DEFAULT_UNSTABLE_CHECK(nothing, w, nothing, nothing) == true
end
