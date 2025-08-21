# Test interaction with DifferentialEquations.jl.
# We solve a trivial decoupled system of ODEs.


using MPI
using OrdinaryDiffEq
using Test
using HDF5


include("cartesian_topology.jl")
include("haloarray.jl")
include("haloarrays.jl")
include("interior_broadcast.jl")
include("interior_broadcast_marray.jl")
include("reduction.jl")
include("halo_exchange.jl")
include("boundary.jl")
include("reduce_dim.jl")
include("gather.jl")
include("save_hdf5.jl")

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nproc = MPI.Comm_size(comm)
rank == 0 || redirect_stdout(devnull)

dims = MPI.Dims_create(nproc, (0, 0))
periods = (true, true)  # periodic boundary in 2D
topology = CartesianTopology(comm, Tuple(dims), periodic=periods)

N_local = (10, 10)  # Local grid size per process
Nglobal = N_local .* dims
h = 1

u0 = HaloArray(Float64, N_local, h, topology; boundary_condition = (:periodic, :periodic))

fill_from_global_indices!(u0) do i
    x = i[1]
    y = i[2]
    return  10 * exp(-(x - (Nglobal[1]/2))^2/5^2 - (y - (Nglobal[2]/2))^2/10^2) + 1
end


function rhs!(du, u, p, t)
    @. du = -0.1 * u
    du
end




@testset "OrdinaryDiffEq" begin
    tspan = (0.0, 1000.0)
    params = (;)
    prob = @inferred ODEProblem{true}(rhs!, u0, tspan, params)

    # This is not fully inferred...
    integrator = init(
        prob, Tsit5();
        adaptive = true, save_everystep = false,
    )

    # Check that all timesteps are the same
    for _ = 1:10
        local dts = MPI.Allgather(integrator.dt, comm)
        @test allequal(dts)
        step!(integrator)
    end

end 


@testset "DiffEqBase" begin
    unorm = DiffEqBase.ODE_DEFAULT_NORM(u0, 0.0)
    unorms = MPI.Allgather(unorm, comm)
    @test allequal(unorms)

    # Note that ODE_DEFAULT_UNSTABLE_CHECK calls NAN_CHECK.
    w = copy(u0)
    wcheck = DiffEqBase.ODE_DEFAULT_UNSTABLE_CHECK(nothing, w, nothing, nothing)
    @test wcheck == false

    # After setting a single value to NaN, all processes should detect it.
    if rank == 0
        w[1] = NaN
    end
    wcheck = DiffEqBase.ODE_DEFAULT_UNSTABLE_CHECK(nothing, w, nothing, nothing)
    @test wcheck == true
end

#=
    @testset "ArrayPartition" begin
        v0 = ArrayPartition(u0)
        prob = @inferred ODEProblem{true}(rhs!, v0, tspan, params)

        # TODO for now this fails when permutations are enabled due to incompatible
        # broadcasting.
        @test_skip integrator = init(
            prob, Tsit5();
            adaptive = true, save_everystep = false,
        )
    end

    # Solve the equation for a 2D vector field represented by a StructArray.
    @testset "StructArray" begin
        v0 = to_structarray((u0, 2u0))
        @assert eltype(v0) <: SVector{2}
        tspan = (0.0, 1.0)
        prob = @inferred ODEProblem{true}(rhs!, v0, tspan, params)
        integrator = init(
            prob, Tsit5();
            adaptive = true, save_everystep = false,
        )
        @test integrator.u == v0
        for _ ∈ 1:10
            step!(integrator)
        end
        @test integrator.u ≠ v0
    end
end
=#