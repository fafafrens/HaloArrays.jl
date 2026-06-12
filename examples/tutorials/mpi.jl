# ============================================================
# HaloArrays.jl — MPI tutorial
#
# Run with:
#   mpiexec -n 4 julia --project=. examples/tutorials/mpi.jl
#
# Sections:
#   1. MPI setup and CartesianTopology
#   2. HaloArray — distributed halo array
#   3. Halo exchange and boundary conditions
#   4. Global reductions and gather
#   5. Multi-field MPI arrays
#   6. Heat equation on a distributed grid (2-D)
# ============================================================

using MPI
using HaloArrays
using Printf

MPI.Init()
const comm  = MPI.COMM_WORLD
const rank  = MPI.Comm_rank(comm)
const nrank = MPI.Comm_size(comm)

# Helper: only rank 0 prints section headers
section(s) = rank == 0 && println("\n" * "=" ^ 60 * "\n" * s * "\n" * "=" ^ 60)

# ============================================================
# 1. MPI SETUP AND CartesianTopology
# ============================================================
#
# CartesianTopology partitions a structured grid across MPI ranks
# arranged in a Cartesian process grid.  Pass 0 in any dimension
# to let MPI choose the decomposition automatically.
#
# Arguments:
#   CartesianTopology(comm, dims; periodic)
#
#   dims     — tuple of integers (0 = auto-select)
#   periodic — which dimensions are periodic

section("Section 1 — CartesianTopology")

# Decompose in both dimensions automatically
topology = CartesianTopology(comm, (0, 0); periodic=(true, true))

if rank == 0
    println("nranks          : ", nrank)
    println("process grid    : ", topology.dims)
    println("my coordinates  : ", topology.cart_coords)
    println("neighbors dim-1 : ", topology.neighbors[1])  # (left, right)
    println("neighbors dim-2 : ", topology.neighbors[2])  # (bottom, top)
end

# ============================================================
# 2. HaloArray — DISTRIBUTED HALO ARRAY
# ============================================================
#
# HaloArray is the MPI-backed counterpart of LocalHaloArray.
# Each rank stores an interior subdomain of size `owned_dims`.  Ghost cells
# at subdomain boundaries are filled from neighbouring ranks
# during halo exchange.
#
# Construction:
#   HaloArray(T, owned_dims, halo_width, topology; boundary_condition)
#
# owned_dims  — size of THIS rank's interior subdomain (not the global size)
# halo_width  — ghost-cell layers on each face
# topology    — CartesianTopology describing the process layout

section("Section 2 — HaloArray")

owned_dims = (8, 8)       # interior cells on this rank
halo_w     = 1

u = HaloArray(Float64, owned_dims, halo_w, topology; boundary_condition=:periodic)

if rank == 0
    println("interior_size: ", interior_size(u))
    println("storage_size : ", storage_size(u))   # includes ghost cells
    println("global_size  : ", global_size(u))    # entire distributed grid
    println("halo_width   : ", halo_width(u))
end

# Logical indexing through u uses global 1-based indices on this rank's
# subdomain.  interior_axes returns the local interior axes.
if rank == 0
    println("interior_axes   : ", interior_axes(u))
end

# Fill each cell with its global linear index as a simple pattern
fill_from_global_indices!(u) do I
    return Float64(I[1] * 100 + I[2])
end

# ============================================================
# 3. HALO EXCHANGE AND BOUNDARY CONDITIONS
# ============================================================
#
# synchronize_halo! performs TWO things in one call:
#   (a) MPI halo exchange — sends/receives subdomain boundary data
#       with neighbouring ranks
#   (b) Boundary conditions — fills ghost cells at domain edges
#       (non-periodic boundaries)
#
# For fine-grained control there is also an async API:
#   start_halo_exchange!(u)   — post MPI Isend / Irecv
#   boundary_condition!(u)    — fill physical-boundary ghosts (can
#                               overlap with computation)
#   finish_halo_exchange!(u)  — wait for MPI transfers to complete

section("Section 3 — Halo exchange")

MPI.Barrier(comm)
synchronize_halo!(u)

# Verify: ghost cell from the left neighbour should hold the last interior
# cell value of that neighbour.
left_ghost  = parent(u)[1,         2]     # left ghost (dim 1, any j)
right_ghost = parent(u)[end,       2]     # right ghost

if rank == 0
    println("left ghost value  : ", left_ghost)
    println("right ghost value : ", right_ghost)
end

# Async example (non-blocking exchange overlapped with a local computation)
start_halo_exchange!(u)
# ... do local, ghost-free work here ...
finish_halo_exchange!(u)
boundary_condition!(u)

# ============================================================
# 4. GLOBAL REDUCTIONS AND GATHER
# ============================================================
#
# Global SCALAR reductions over the interior cells of every
# rank use the ordinary Base functions — they Allreduce internally
# and return the same result on every rank:
#
#   sum(u)   maximum(u)   minimum(u)   mapreduce(f, op, u)
#
# To collapse only SOME axes (and keep a distributed array) use
# mapreduce_haloarray_dims(f, op, u, dims).  Passing `dims=` to the
# scalar functions above is intentionally rejected — a per-slice
# global reduction needs sub-communicators, which that helper builds.
#
# gather_haloarray collects all subdomain data onto rank 0.

section("Section 4 — Reductions and gather")

global_sum = sum(u)
global_max = maximum(u)
global_mr  = mapreduce(identity, +, u)   # collective: EVERY rank must call it

if rank == 0
    println("global sum : ", global_sum)
    println("global max : ", global_max)
    println("mapreduce  : ", global_mr)   # same as sum(u)
end

# gather_haloarray assembles the full grid on rank 0 (returns nothing on others)
global_array = gather_haloarray(u; root=0)
if rank == 0
    println("gathered size : ", size(global_array))
end

# ============================================================
# 5. MULTI-FIELD MPI ARRAYS
# ============================================================
#
# MultiHaloArray groups several HaloArrays on the same topology.
# synchronize_halo! exchanges halos for every field at once.
# Individual fields are accessed by name: state.rho, state.vel, …

section("Section 5 — Multi-field MPI arrays")

bcs_hydro = (
    rho = ((Periodic(), Periodic()), (Periodic(), Periodic())),
    vel = ((Periodic(), Periodic()), (Periodic(), Periodic())),
    pre = ((Periodic(), Periodic()), (Periodic(), Periodic())),
)

state = MultiHaloArray(HaloArray, Float64, owned_dims, halo_w, topology;
    boundary_conditions=bcs_hydro)

fill_from_global_indices!(state.rho) do I; Float64(I[1]); end
fill_from_global_indices!(state.vel) do I; Float64(I[2]); end
fill_from_global_indices!(state.pre) do I; 1.0; end

synchronize_halo!(state)   # exchanges all three fields in one call

rho_max = maximum(state.rho)
MPI.Barrier(comm)
if rank == 0
    println("max(rho) after exchange : ", rho_max)
    println("n_field  : ", n_field(state))
end

# ============================================================
# 6. HEAT EQUATION ON A DISTRIBUTED GRID (2-D)
# ============================================================
#
# We solve  ∂u/∂t = α (∂²u/∂x² + ∂²u/∂y²)
# on a periodic domain decomposed across ranks.
#
# The only difference from the local case is:
#   • HaloArray instead of LocalHaloArray
#   • synchronize_halo! also triggers MPI exchange
#   • global diagnostics use Allreduce through mapreduce

section("Section 6 — Distributed heat equation (2-D)")

function heat_rhs_2d!(du, u, alpha, dx)
    data  = parent(u)
    ddata = parent(du)
    ex = CartesianIndex(1, 0)
    ey = CartesianIndex(0, 1)
    inv_dx2 = inv(dx[1]^2)
    inv_dy2 = inv(dx[2]^2)
    for I in CartesianIndices(interior_range(u))
        lap = (data[I+ex] - 2*data[I] + data[I-ex]) * inv_dx2 +
              (data[I+ey] - 2*data[I] + data[I-ey]) * inv_dy2
        ddata[I] = alpha * lap
    end
    return du
end

function run_distributed_heat_2d(; owned_dims=(16,16), alpha=1.0, nt=50, cfl=0.4)
    topo = CartesianTopology(comm, (0, 0); periodic=(true, true))

    u     = HaloArray(Float64, owned_dims, 1, topo; boundary_condition=:periodic)
    u_nxt = similar(u)
    du    = similar(u)

    n_global = global_size(u)
    dx = (1.0/n_global[1], 1.0/n_global[2])
    dt = cfl / (alpha * (inv(dx[1]^2) + inv(dx[2]^2)))

    # Gaussian initial condition via global index
    fill_from_global_indices!(u) do I
        cx, cy = (n_global[1]+1)/2, (n_global[2]+1)/2
        r2 = ((I[1]-cx)/(n_global[1]/8))^2 + ((I[2]-cy)/(n_global[2]/8))^2
        return 1.0 + exp(-r2)
    end
    synchronize_halo!(u)

    u0_max = maximum(u)

    current, nxt = u, u_nxt
    for _ in 1:nt
        synchronize_halo!(current)
        heat_rhs_2d!(du, current, alpha, dx)
        interior_view(nxt) .= interior_view(current) .+ dt .* interior_view(du)
        current, nxt = nxt, current
    end
    synchronize_halo!(current)

    u_max = maximum(current)

    if rank == 0
        @printf("  global grid  : %d x %d  (nranks=%d)\n", n_global..., nrank)
        @printf("  dt=%.2e  nt=%d  max(u₀)=%.4f  max(u)=%.4f\n",
            dt, nt, u0_max, u_max)
    end
    return current
end

run_distributed_heat_2d()

MPI.Barrier(comm)
rank == 0 && println("\nMPI tutorial complete.")
MPI.Finalize()
