using Test
using MPI
using DiffEqBase
using OrdinaryDiffEq
using HaloArrays

function _ode_topology()
    comm = MPI.COMM_WORLD
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
    interior_view(du) .= -0.1 .* interior_view(u)
    return du
end

@testset "ODEProblem HaloArray state" begin
    topology = _ode_topology()
    u0 = _ode_initial_condition(topology)

    tspan = (0.0, 1.0)
    prob = ODEProblem{true}(_rhs!, u0, tspan, (;))
    du = zero(u0)
    _rhs!(du, u0, prob.p, first(tspan))

    @test prob.u0 === u0
    @test du isa HaloArray
    @test size(du) == size(u0)
    @test halo_width(du) == halo_width(u0)
    @test all(isfinite, du)
    @test all(x -> x < 0, du)
end

@testset "DiffEq default checks use HaloArray reductions" begin
    topology = _ode_topology()
    u = _ode_initial_condition(topology)

    unorm = DiffEqBase.ODE_DEFAULT_NORM(u, 0.0)
    unorms = MPI.Allgather(unorm, MPI.COMM_WORLD)
    @test allequal(unorms)

    w = copy(u)
    @test DiffEqBase.ODE_DEFAULT_UNSTABLE_CHECK(nothing, w, nothing, nothing) == false

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        w[1, 1] = NaN
    end
    @test DiffEqBase.ODE_DEFAULT_UNSTABLE_CHECK(nothing, w, nothing, nothing) == true
end
