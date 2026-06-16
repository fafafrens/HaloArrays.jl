# Distributed companion to stiff_reaction_diffusion_implicit_1d.jl: the same
# implicit (FBDF) + autodiff + matrix-free Krylov solve, but the ODE state is a
# *distributed* HaloArray over a CartesianTopology. Run with, e.g.
#
#   mpiexec -n 2 julia --project=examples examples/finite_volume/stiff_reaction_diffusion_implicit_mpi_1d.jl
#   mpiexec -n 4 julia --project=examples examples/finite_volume/stiff_reaction_diffusion_implicit_mpi_1d.jl
#
# Nothing about the solve changes from the serial version — only the state
# constructor (HaloArray + topology). It works because the vector-space ops the
# solver uses are collective: dot/norm Allreduce, and the elementwise updates
# (axpy!/broadcast) are correct local-per-rank. The script checks that the
# integrator stays collective (identical accepted-step counts on every rank) and
# that the gathered solution matches a single-process reference.

using MPI
using HaloArrays
using OrdinaryDiffEq
using LinearSolve
using Krylov
using Printf

MPI.Initialized() || MPI.Init()
const COMM = MPI.COMM_WORLD
const NR   = MPI.Comm_size(COMM)

const D = 1.0
const R = 8.0
const E1 = CartesianIndex(1)
const GNX = 96                       # global cells; must divide the rank count
const DX2INV = inv((1.0 / GNX)^2)

ic(x) = 0.1 + 0.8 * exp(-50 * (x - 0.5)^2)

# Identical to the serial example's RHS — `tile_count` is 1 for a distributed
# HaloArray (this rank's local block), so the same per-tile loop applies.
function rhs!(du, u, p, t)
    synchronize_halo!(u)             # the MPI halo exchange (also on Duals)
    for tile in 1:tile_count(u)
        s = tile_parent(u, tile); d = tile_parent(du, tile)
        @inbounds for I in CartesianIndices(interior_range(u))
            d[I] = D * (s[I - E1] - 2s[I] + s[I + E1]) * DX2INV + R * s[I] * (1 - s[I])
        end
    end
    return nothing
end

const ALG = FBDF(linsolve = HaloKrylov(:gmres), concrete_jac = false)

function run_mpi()
    NR == 1 || @assert GNX % NR == 0 "global grid ($GNX) must divide the rank count ($NR)"
    topo = CartesianTopology(COMM, (0,); periodic = (true,))
    u0 = HaloArray(Float64, (GNX ÷ NR,), 1, topo; boundary_condition = ((Periodic(), Periodic()),))
    fill_from_global_indices!(I -> ic((I[1] - 0.5) / GNX), u0)

    sol = solve(ODEProblem(rhs!, u0, (0.0, 0.3)), ALG;
                reltol = 1e-7, abstol = 1e-7, save_everystep = false)

    steps    = MPI.Allgather(sol.stats.naccept, COMM)
    gathered = gather_haloarray(sol.u[end]; root = 0)        # global Array on rank 0

    if MPI.Comm_rank(COMM) == 0
        uref = LocalHaloArray(Float64, (GNX,), 1; boundary_condition = :periodic)
        fill_from_global_indices!(I -> ic((I[1] - 0.5) / GNX), uref)
        sref = solve(ODEProblem(rhs!, uref, (0.0, 0.3)), ALG;
                     reltol = 1e-7, abstol = 1e-7, save_everystep = false)
        err = maximum(abs, vec(gathered) .- collect(interior_view(sref.u[end])))
        @printf("distributed implicit on %d rank(s)\n", NR)
        @printf("  accepted steps / rank : %s  %s\n", steps,
                all(==(steps[1]), steps) ? "(consistent)" : "(STEP COUNTS DIVERGED!)")
        @printf("  max |MPI - serial|    : %.2e  %s\n", err, err < 1e-6 ? "[OK]" : "[MISMATCH]")
    end
    MPI.Barrier(COMM)
    return sol
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_mpi()
end
