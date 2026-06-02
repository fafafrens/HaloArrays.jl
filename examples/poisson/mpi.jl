# ============================================================
# HaloArrays.jl — DISTRIBUTED matrix-free Poisson (MPI)
#
# Run with (e.g. 4 ranks):
#   mpiexec -n 4 julia --project=examples examples/poisson_mpi.jl
#
# This is the MPI counterpart of examples/poisson_operator.jl. It solves
#
#     -∇²u = f   on (0,1)²,   u = 0 on the boundary
#
# on a grid decomposed across ranks, using the SAME coordinate-free solvers
# (CG, BiCGStab, GMRES) from examples/krylov_solvers.jl. Nothing about them or
# the operator
# changes for the distributed case — the only reason it works is that
# HaloArrays.jl defines dot/norm as GLOBAL reductions (MPI Allreduce), so the
# Krylov inner products are correct across ranks. synchronize_halo!(x) inside
# the operator does the MPI halo exchange plus the boundary condition.
#
# The operator is written with separate hx, hy so it is correct for ANY rank
# decomposition (the global grid need not be square). The domain is always the
# unit square (0,1)²; only the cell spacings differ per direction.
# ============================================================

using MPI
using HaloArrays
using SciMLOperators
using LinearAlgebra: mul!
using Printf

include("krylov_solvers.jl")        # cg!, bicgstab!, gmres!

MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NRANKS = MPI.Comm_size(COMM)

# -∇² with separate inv_hx², inv_hy² (anisotropic-safe across decompositions)
function neg_laplacian!(y, x, _u, p, _t)
    synchronize_halo!(x)                       # MPI exchange + boundary condition
    xd = parent(x); yd = parent(y)
    ex = CartesianIndex(1, 0); ey = CartesianIndex(0, 1)
    @inbounds for I in CartesianIndices(interior_range(x))
        yd[I] = -((xd[I+ex] - 2xd[I] + xd[I-ex]) * p.inv_hx2 +
                  (xd[I+ey] - 2xd[I] + xd[I-ey]) * p.inv_hy2)
    end
    return y
end

function run_distributed_poisson(; owned=(32, 32))
    topo = CartesianTopology(COMM, (0, 0); periodic=(false, false))
    bc = ((Antireflecting(), Antireflecting()), (Antireflecting(), Antireflecting()))

    u   = HaloArray(Float64, owned, 1, topo; boundary_condition=bc)
    rhs = HaloArray(Float64, owned, 1, topo; boundary_condition=bc)
    uex = HaloArray(Float64, owned, 1, topo; boundary_condition=bc)

    ng = global_size(u)                        # global cell counts (Nx, Ny)
    hx, hy = 1.0 / ng[1], 1.0 / ng[2]
    cx(i) = (i - 0.5) * hx
    cy(j) = (j - 0.5) * hy

    # manufactured u = x(1-x)·y(1-y) (zero on every wall);  -∇²u = 2(x(1-x)+y(1-y))
    fill_from_global_indices!(uex) do I
        cx(I[1]) * (1 - cx(I[1])) * cy(I[2]) * (1 - cy(I[2]))
    end
    fill_from_global_indices!(rhs) do I
        2 * (cx(I[1]) * (1 - cx(I[1])) + cy(I[2]) * (1 - cy(I[2])))
    end
    fill!(u, 0.0)

    L = FunctionOperator(neg_laplacian!, similar(rhs), similar(rhs);
        islinear=true, isconstant=true, issymmetric=true, isposdef=true,
        p=(inv_hx2=1 / hx^2, inv_hy2=1 / hy^2))

    if RANK == 0
        @printf("  ranks=%d  topology=%s  global=%dx%d\n", NRANKS, string(topo.dims), ng...)
    end

    # Every solver is coordinate-free, so all three run distributed unchanged.
    # GMRES is the real check: its Arnoldi basis is a set of HaloArrays and the
    # Hessenberg is built from dot(Vᵢ, w) — those dots must be the global
    # Allreduce for the basis to be consistent across ranks.
    solvers = (("CG",        (uu, A, b) -> cg!(uu, A, b; tol=1e-10)),
               ("BiCGStab",  (uu, A, b) -> bicgstab!(uu, A, b; tol=1e-10)),
               ("GMRES(50)", (uu, A, b) -> gmres!(uu, A, b; tol=1e-10, restart=50)))
    for (name, solve) in solvers
        uu = similar(rhs); fill!(uu, 0.0)
        _, iters, res = solve(uu, L, rhs)
        local_err  = maximum(abs, interior_view(uu) .- interior_view(uex))
        global_err = MPI.Allreduce(local_err, MPI.MAX, COMM)
        RANK == 0 && @printf("    %-10s iters=%-4d  residual=%.2e  max|u-u_exact|=%.3e\n",
            name, iters, res, global_err)
    end
    return nothing
end

RANK == 0 && println("Distributed matrix-free Poisson (MPI) — CG / BiCGStab / GMRES")
run_distributed_poisson(; owned=(32, 32))
MPI.Barrier(COMM)
RANK == 0 && println("poisson_mpi complete.")
