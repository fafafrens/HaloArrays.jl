using Test
using HaloArrays
using MPI
using LinearSolve
using Krylov
using OrdinaryDiffEq
using LinearAlgebra

# Distributed implicit solve regression: the stiff Fisher–KPP reaction–diffusion
# integrated with FBDF + HaloKrylov(:gmres), the state a *distributed* HaloArray.
# Asserts the integrator stays collective (identical accepted-step counts on
# every rank) and that the gathered solution matches a single-process reference.
# Runs only under the MPI CI job (MPI_SIZE > 1).

@testset "MPI implicit (FBDF + HaloKrylov)" begin
    comm = MPI.COMM_WORLD
    nr = MPI.Comm_size(comm)
    GNX = 96
    @test GNX % nr == 0           # the chosen grid divides the CI rank counts (2, 4)

    D, R, E1 = 1.0, 8.0, CartesianIndex(1)
    ic(x) = 0.1 + 0.8 * exp(-50 * (x - 0.5)^2)
    function rhs!(du, u, p, t)
        dx2inv = p
        synchronize_halo!(u)
        for tile in 1:tile_count(u)
            s = tile_parent(u, tile); d = tile_parent(du, tile)
            @inbounds for I in CartesianIndices(interior_range(u))
                d[I] = D * (s[I - E1] - 2s[I] + s[I + E1]) * dx2inv + R * s[I] * (1 - s[I])
            end
        end
        return nothing
    end
    alg = FBDF(linsolve = HaloKrylov(:gmres), concrete_jac = false)
    dx2inv = inv((1.0 / GNX)^2)

    topo = CartesianTopology(comm, (0,); periodic = (true,))
    u0 = HaloArray(Float64, (GNX ÷ nr,), 1, topo; boundary_condition = ((Periodic(), Periodic()),))
    fill_from_global_indices!(I -> ic((I[1] - 0.5) / GNX), u0)
    sol = solve(ODEProblem(rhs!, u0, (0.0, 0.3), dx2inv), alg;
                reltol = 1e-7, abstol = 1e-7, save_everystep = false)
    @test sol.retcode == ReturnCode.Success

    # Newton/linear convergence decisions must be collective → identical step counts.
    steps = MPI.Allgather(sol.stats.naccept, comm)
    @test all(==(steps[1]), steps)

    # Gathered distributed solution matches the serial reference.
    gathered = gather_haloarray(sol.u[end]; root = 0)
    if MPI.Comm_rank(comm) == 0
        uref = LocalHaloArray(Float64, (GNX,), 1; boundary_condition = :periodic)
        fill_from_global_indices!(I -> ic((I[1] - 0.5) / GNX), uref)
        sref = solve(ODEProblem(rhs!, uref, (0.0, 0.3), dx2inv), alg;
                     reltol = 1e-7, abstol = 1e-7, save_everystep = false)
        @test maximum(abs, vec(gathered) .- collect(interior_view(sref.u[end]))) < 1e-6
    end
    MPI.Barrier(comm)
end
