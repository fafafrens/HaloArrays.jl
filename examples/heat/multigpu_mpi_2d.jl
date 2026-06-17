# =============================================================================
# Multi-GPU heat diffusion: one MPI rank per GPU, device-resident HaloArray,
# GPU-to-GPU halo exchange.
#
# This is the "one rank ↔ one GPU" pattern: each rank owns a device-resident
# subdomain; `synchronize_halo!` exchanges ghost layers directly between GPUs.
# HaloArrays makes this work by construction — the parent array type is generic,
# and the MPI send/recv buffers are `similar(data, …)`, so a device parent gives
# device buffers that are handed straight to `MPI.Isend`/`Irecv!` (no host
# round-trip). The interior update is a single KernelAbstractions `@kernel` that
# compiles for whatever backend the parent array lives on (CPU / CUDA / Metal /
# ROCm via `get_backend`), so the same code runs everywhere.
#
# NB: `@index(Global, NTuple)` is hoisted to its own line *before* `cell_index`.
# KA's CPU backend rewrites `@index` via an AST transform that only fires when the
# macro is in a standalone/assignment position; nesting it inside a call
# (`cell_index(region, @index(...))`) makes it compile for GPU but FAIL on the CPU
# backend (no method for the 1-arg `__index_Global_Cartesian`). Hoisting keeps the
# kernel runnable on both.
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
using KernelAbstractions
using LinearAlgebra: norm
using Printf

const KA = KernelAbstractions

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

# ---- one diffusion step: exchange ghosts, then a portable KA 5-point Laplacian -
# HaloArrays' CellKernelRegion + cell_index map the launch index to the padded
# parent cell; ±1 reaches into the freshly-synced ghost layer.
@kernel function heat_kernel!(out, s, dx2inv, region::CellKernelRegion{2})
    J = @index(Global, NTuple)                        # hoisted (see header note)
    i, j = cell_index(region, J)                      # (i, j) into the padded parent
    @inbounds out[i, j] = s[i, j] + dx2inv *
        (s[i-1, j] + s[i+1, j] + s[i, j-1] + s[i, j+1] - 4 * s[i, j])
end

function step!(unew, u, kernel!, backend; dx2inv)
    synchronize_halo!(u)                              # GPU↔GPU (or rank↔rank) exchange
    region = get_interior_cell_region(CellRanges(u))
    any(==(0), region.size) && return nothing
    kernel!(parent(unew), parent(u), dx2inv, region; ndrange = region.size)
    KA.synchronize(backend)
    return nothing
end

# ---- run a few steps; report the global L2 norm (a cross-rank Allreduce) -----
# Wrapped in a function: a top-level `for` would make the ping-pong swap (`u, unew
# = unew, u`) loop-local and also reduce performance under global scope.
function run!(u, unew, nstep, dt)
    backend = KA.get_backend(parent(u))               # CPU / CUDA / Metal / ROC
    kernel! = heat_kernel!(backend)
    MPI.Barrier(COMM); t0 = time()
    for _ in 1:nstep
        step!(unew, u, kernel!, backend; dx2inv = dt)
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
