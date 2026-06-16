# =============================================================================
# Multi-GPU heat diffusion: one MPI rank per GPU, device-resident HaloArray,
# GPU-to-GPU halo exchange.
#
# This is the "one rank ↔ one GPU" pattern: each rank owns a device-resident
# subdomain; `synchronize_halo!` exchanges ghost layers directly between GPUs.
# HaloArrays makes this work by construction — the parent array type is generic,
# and the MPI send/recv buffers are `similar(data, …)`, so a device parent gives
# device buffers that are handed straight to `MPI.Isend`/`Irecv!` (no host
# round-trip). The interior update below is a *broadcast* (no scalar indexing),
# so the same code runs on CPU and GPU.
#
# REQUIREMENTS for the real GPU path:
#   * A GPU-aware MPI (CUDA-aware / ROCm-aware) — the default MPICH/OpenMPI JLL is
#     host-only and will NOT accept device pointers. Configure MPI.jl to use the
#     system GPU-aware MPI (see MPIPreferences.use_system_binary()).
#   * One visible GPU per rank; this script binds each rank to a device by its
#     node-local rank.
#
# LAUNCH
#   # CUDA cluster, 4 GPUs on the node:
#   HALO_BACKEND=cuda mpiexec -n 4 julia --project=examples examples/heat/multigpu_mpi_2d.jl
#   # Local CPU smoke test (no GPU needed — verifies the exchange + reduction logic):
#   mpiexec -n 4 julia --project=examples examples/heat/multigpu_mpi_2d.jl
#
# NOTE: the GPU path is untested in CI (needs GPU + GPU-aware MPI). The CPU path
# is the reference; the GPU path is identical code on a device array type.
# =============================================================================

using HaloArrays
using MPI
using LinearAlgebra: norm
using Printf

MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NR   = MPI.Comm_size(COMM)

# ---- backend selection ------------------------------------------------------
# CPU by default so the example runs anywhere. Set HALO_BACKEND=cuda|amdgpu on a
# node that has the GPU package + a GPU-aware MPI. We load the GPU package lazily
# so the CPU path needs no GPU dependency installed.
const BACKEND = lowercase(get(ENV, "HALO_BACKEND", "cpu"))

function bind_device_and_get_adaptor()
    if BACKEND == "cpu"
        return identity, "CPU Float64"
    end
    # node-local rank → which GPU this rank owns
    nodecomm   = MPI.Comm_split_type(COMM, MPI.COMM_TYPE_SHARED, RANK)
    local_rank = MPI.Comm_rank(nodecomm)
    if BACKEND == "cuda"
        @eval Main using CUDA
        ndev = length(Main.CUDA.devices())
        Main.CUDA.device!(local_rank % ndev)
        return (A -> Main.CuArray(A)), "CUDA (rank $RANK → gpu $(local_rank % ndev))"
    elseif BACKEND == "amdgpu"
        @eval Main using AMDGPU
        ndev = length(Main.AMDGPU.devices())
        Main.AMDGPU.device!(Main.AMDGPU.devices()[local_rank % ndev + 1])
        return (A -> Main.ROCArray(A)), "AMDGPU (rank $RANK → gpu $(local_rank % ndev))"
    else
        error("unknown HALO_BACKEND=$BACKEND (use cpu|cuda|amdgpu)")
    end
end

to_device, backend_label = bind_device_and_get_adaptor()

# ---- build a device-resident, MPI-decomposed HaloArray ----------------------
const T     = Float64
const HALO  = 1
const LOCAL = (256, 256)                 # this rank's owned interior (per GPU)
const FULL  = LOCAL .+ 2HALO             # padded storage (interior + ghosts)

topo = CartesianTopology(COMM, (0, 0); periodic = (true, true))
bc   = ((Periodic(), Periodic()), (Periodic(), Periodic()))

# full padded array on the chosen device, wrapped as an MPI HaloArray
mk() = HaloArray(to_device(zeros(T, FULL...)), HALO, topo, bc)
u    = mk()
unew = mk()

# smooth global initial condition (set on each rank's interior via broadcast)
gx, gy = LOCAL .* topo.dims
fill_from_global_indices!(I -> sinpi(2I[1] / gx) * sinpi(2I[2] / gy), u)

# ---- one diffusion step: exchange ghosts, then a broadcast 5-point Laplacian -
# Broadcasting over shifted parent views is GPU-safe (no scalar getindex). The
# inner range is the interior; ±1 reaches into the freshly-synced ghost layer.
function step!(unew, u; dx2inv = 1.0)
    synchronize_halo!(u)                              # GPU↔GPU (or rank↔rank) exchange
    s  = parent(u); d = parent(unew)
    ix = (HALO + 1):(HALO + LOCAL[1])
    iy = (HALO + 1):(HALO + LOCAL[2])
    @views @. d[ix, iy] = s[ix, iy] + dx2inv * (
        s[ix .- 1, iy] + s[ix .+ 1, iy] + s[ix, iy .- 1] + s[ix, iy .+ 1] - 4 * s[ix, iy])
    return nothing
end

# ---- run a few steps; report the global L2 norm (a cross-rank Allreduce) -----
# Wrapped in a function: a top-level `for` would make the ping-pong swap (`u, unew
# = unew, u`) loop-local and also reduce performance under global scope.
function run!(u, unew, nstep, dt)
    MPI.Barrier(COMM); t0 = time()
    for _ in 1:nstep
        step!(unew, u; dx2inv = dt)
        u, unew = unew, u                             # ping-pong buffers
    end
    MPI.Barrier(COMM)
    return u, MPI.Allreduce(time() - t0, MPI.MAX, COMM)
end

const NSTEP = 20
const DT    = 0.1                                     # CFL-stable for unit spacing
if RANK == 0
    @printf("backend=%s   ranks=%d   topo=%s   global=%dx%d\n",
            backend_label, NR, string(topo.dims), gx, gy)
end
u, el = run!(u, unew, NSTEP, DT)

gnorm = norm(u)                                       # global (Allreduce) L2 norm
if RANK == 0
    @printf("done: %d steps in %.4f s (%.3f ms/step)   global ‖u‖₂ = %.6f\n",
            NSTEP, el, 1e3 * el / NSTEP, gnorm)
    @assert isfinite(gnorm) "diffusion diverged — check CFL / exchange"
    println("OK")
end
MPI.Barrier(COMM)
