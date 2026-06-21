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
#   # Local CPU smoke test (no GPU needed — verifies the exchange + reduction logic):
#   mpiexec -n 4 julia --project=examples examples/heat/multigpu_mpi_2d.jl
#   # GPU cluster: see RUNNING_ON_LEONARDO.md for a full, tested recipe (system MPI +
#   # system HDF5 + CUDA local toolkit + `srun --mpi=pmix_v3`).
#
# STATUS: the CPU path runs in CI. The CUDA path is VERIFIED on CINECA Leonardo
# (4× A100, one rank per GPU, GPU-to-GPU CUDA-aware-MPI exchange): the global L2
# norm is bit-identical to the CPU result (1 GPU → 127.691943, 4 GPUs → 255.845833),
# so the device path computes the same answer as the reference.
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
# node that has the GPU package + a GPU-aware MPI.
const BACKEND = lowercase(get(ENV, "HALO_BACKEND", "cpu"))

# Load the GPU package at TOP LEVEL (not lazily inside the function): a
# `@eval using CUDA` followed by using `CUDA`/`Main.CUDA` in the *same* call frame
# throws a world-age error ("the binding may be too new"). Doing it here means the
# package is in effect before `bind_device_and_get_adaptor` runs. The conditional
# keeps the CPU path free of any GPU dependency.
BACKEND == "cuda"   && @eval using CUDA
BACKEND == "amdgpu" && @eval using AMDGPU

function bind_device_and_get_adaptor()
    if BACKEND == "cpu"
        return identity, "CPU Float64"
    end
    # node-local rank → which GPU this rank owns
    nodecomm   = MPI.Comm_split_type(COMM, MPI.COMM_TYPE_SHARED, RANK)
    local_rank = MPI.Comm_rank(nodecomm)
    if BACKEND == "cuda"
        ndev = length(Main.CUDA.devices())
        Main.CUDA.device!(local_rank % ndev)
        return (A -> Main.CuArray(A)), "CUDA (rank $RANK → gpu $(local_rank % ndev))"
    elseif BACKEND == "amdgpu"
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

gx, gy = LOCAL .* topo.dims

# Initial condition. `fill_from_global_indices!` is a scalar host loop (it writes
# `parent[I] = f(global_I)` cell by cell), so it is NOT GPU-safe. Build the IC on a
# host HaloArray of the same topology, then move its parent to the device and wrap
# it — the cpu_vs_gpu_2d.jl pattern. Only this one-time setup is staged through the
# host; the MPI exchange and KA stencil below are device-native.
host = HaloArray(T, LOCAL, HALO, topo, bc)
fill_from_global_indices!(I -> sinpi(2I[1] / gx) * sinpi(2I[2] / gy), host)
u    = HaloArray(to_device(parent(host)), HALO, topo, bc)   # device, IC already set
unew = HaloArray(to_device(zeros(T, FULL...)), HALO, topo, bc)

# ---- one diffusion step: exchange ghosts, then a portable KA 5-point Laplacian -
# HaloArrays' CellWindow + cell_index map the launch index to the padded
# parent cell; ±1 reaches into the freshly-synced ghost layer.
@kernel function heat_kernel!(out, s, dx2inv, region::CellWindow{2})
    J = @index(Global, NTuple)                        # hoisted (see header note)
    i, j = cell_index(region, J)                      # (i, j) into the padded parent
    @inbounds out[i, j] = s[i, j] + dx2inv *
        (s[i-1, j] + s[i+1, j] + s[i, j-1] + s[i, j+1] - 4 * s[i, j])
end

function step!(unew, u, kernel!, backend; dx2inv)
    synchronize_halo!(u)                              # GPU↔GPU (or rank↔rank) exchange
    region = get_interior_cell_window(CellRanges(u))
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
