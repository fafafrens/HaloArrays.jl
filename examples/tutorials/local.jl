# ============================================================
# HaloArrays.jl — hands-on tutorial (single process, no MPI)
#
# Run the whole file, or step through sections in the REPL:
#   julia --project=examples -t 4 examples/tutorial_local.jl
#
#   1. What is a halo array?          (LocalHaloArray, storage layout)
#   2. Boundary conditions
#   3. Range helpers                  (CellRanges / FaceRanges)
#   4. Worked example: 1-D heat equation
#   5. Multiple fields                (LocalMultiHaloArray)
#   6. Threaded arrays                (ThreadedHaloArray)
#   7. Indexed field collections      (ArrayOfHaloArray)
# ============================================================

using HaloArrays
using OhMyThreads: @tasks
using Printf

# Section banner helper (keeps the examples below uncluttered).
section(title) = (println(); println("="^60); println(title); println("="^60))

# ============================================================
# 1. WHAT IS A HALO ARRAY?
# ============================================================
# A stencil update needs a ring of "ghost" cells around the data a process
# owns, filled from neighbours or boundary conditions before each sweep.
# LocalHaloArray wraps a plain array and tracks the halo width and boundary
# condition.  Indexing (u[i], axes, eachindex) always refers to the OWNED
# cells — the ghost offset is hidden from you.
#
#   storage:  [ ghost | 1 2 3 4 | ghost ]   (halo_width = 1)
#   indices the user sees:  1..4

section("1 — Storage layout")

u1d = LocalHaloArray(Float64, (4,), 1; boundary_condition=:periodic)
interior_view(u1d) .= [10.0, 20.0, 30.0, 40.0]   # interior_view = owned cells only

println("owned / storage size : ", owned_size(u1d), " / ", storage_size(u1d))  # (4,) / (6,)
println("u1d[2], axes(u1d)    : ", u1d[2], ", ", axes(u1d))                     # 20.0, (1:4,)
println("full storage         : ", collect(parent(u1d)))                        # ghosts still 0

# ============================================================
# 2. BOUNDARY CONDITIONS
# ============================================================
# synchronize_halo! fills the ghost cells from the chosen condition:
#   :periodic        ghosts wrap from the far side
#   :repeating       ghosts copy the nearest owned value (zero-gradient)
#   Reflecting()     ghost = +interior  (mirror)
#   Antireflecting() ghost = -interior  (zero value at the wall)
# N-D arrays pass one (left, right) pair per dimension.

section("2 — Boundary conditions")

function show_bc(label, bc)
    u = LocalHaloArray(Float64, (4,), 1; boundary_condition=bc)
    interior_view(u) .= [1.0, 2.0, 3.0, 4.0]
    synchronize_halo!(u)
    @printf("  %-15s %s\n", label, collect(parent(u)))
end
show_bc("periodic",       :periodic)
show_bc("repeating",      :repeating)
show_bc("reflecting",     ((Reflecting(), Reflecting()),))
show_bc("antireflecting", ((Antireflecting(), Antireflecting()),))

# One (left, right) pair per dimension:
u2d = LocalHaloArray(Float64, (3, 3), 1;
    boundary_condition=((Periodic(), Periodic()), (Reflecting(), Reflecting())))
println("  2-D mixed BC: size=", owned_size(u2d), " halo=", halo_width(u2d))

# ============================================================
# 3. RANGE HELPERS
# ============================================================
# These name the index ranges a stencil loop needs, so you don't hand-write
# CartesianIndices(interior_range(u)) every time:
#   CellRanges — cell-centred loops:  get_owned_cells,
#                get_colored_owned_cell_ranges (checkerboard / Gauss-Seidel)
#   FaceRanges — finite-volume flux loops:  get_left_face / get_internal_face /
#                get_right_face, and get_unit_vector (offset across a face)

section("3 — Range helpers")

u = LocalHaloArray(Float64, (6,), 1; boundary_condition=:periodic)
interior_view(u) .= Float64.(1:6)

cr = CellRanges(u)
println("owned cells  : ", collect(get_owned_cells(cr)))
println("checkerboard : ", collect.(get_colored_owned_cell_ranges(cr, 0)),
        " / ", collect.(get_colored_owned_cell_ranges(cr, 1)))

fr = FaceRanges(u)
println("faces (left / internal / right):")
println("  ", collect(get_left_face(fr, 1)), "  ",
              collect(get_internal_face(fr)), "  ",
              collect(get_right_face(fr, 1)))

# ============================================================
# 4. WORKED EXAMPLE — 1-D HEAT EQUATION
# ============================================================
# Solve  ∂u/∂t = α ∂²u/∂x²  on [0,1] with periodic BCs, explicit Euler.
# Each step: (1) synchronize_halo! to fill ghosts, (2) read parent(u) with
# ghost-safe stencil indices, (3) write the interior of u_next.

section("4 — Heat equation (LocalHaloArray)")

function heat_step_local!(u_next, u, alpha, dt, dx)
    synchronize_halo!(u)
    data    = parent(u)
    out     = parent(u_next)
    inv_dx2 = inv(dx^2)
    e       = CartesianIndex(1)
    for I in CartesianIndices(interior_range(u))
        out[I] = data[I] + alpha * dt * (data[I+e] - 2*data[I] + data[I-e]) * inv_dx2
    end
    return u_next
end

function run_heat_1d(; nx=64, alpha=1.0, nt=200, cfl=0.4)
    dx = 1.0 / nx
    dt = cfl * dx^2 / alpha

    u      = LocalHaloArray(Float64, (nx,), 1; boundary_condition=:periodic)
    u_next = similar(u)
    for i in 1:nx                       # Gaussian initial condition
        u[i] = exp(-100 * ((i/nx) - 0.5)^2)
    end
    u0max = maximum(u)

    for _ in 1:nt
        heat_step_local!(u_next, u, alpha, dt, dx)
        u, u_next = u_next, u
    end

    @printf("  nx=%d  nt=%d  max(u₀)=%.4f  max(u)=%.4f\n", nx, nt, u0max, maximum(u))
    return u
end

run_heat_1d()

# ============================================================
# 5. MULTIPLE FIELDS — LocalMultiHaloArray
# ============================================================
# Groups several fields on the same grid into one container. You can access
# fields by name (state.rho), synchronize all of them in one call, and
# broadcast across all of them at once.

section("5 — Multi-field arrays (LocalMultiHaloArray)")

# Upwind advection of a density and a passive scalar (a > 0, periodic).
function run_advection_multifield(; nx=64, nt=100, cfl=0.8)
    dx = 1.0 / nx
    a  = 1.0
    dt = cfl * dx / a

    state = LocalMultiHaloArray(Float64, (nx,), 1; boundary_conditions=(
        rho = ((Periodic(), Periodic()),),
        phi = ((Periodic(), Periodic()),),
    ))
    for i in 1:nx
        x = (i - 0.5) / nx
        state.rho[i] = 1.0 + exp(-100*(x - 0.25)^2)
        state.phi[i] = sin(2π * x)
    end

    rho_next = similar(state.rho)
    phi_next = similar(state.phi)
    e = CartesianIndex(1)
    for _ in 1:nt
        synchronize_halo!(state)        # refreshes rho AND phi at once
        # Stencil reads ghosts, so index the storage arrays (parent).
        rho, phi = parent(state.rho), parent(state.phi)
        rho_o, phi_o = parent(rho_next), parent(phi_next)
        for I in CartesianIndices(interior_range(state.rho))
            rho_o[I] = rho[I] - (a*dt/dx) * (rho[I] - rho[I-e])
            phi_o[I] = phi[I] - (a*dt/dx) * (phi[I] - phi[I-e])
        end
        copyto!(interior_view(state.rho), interior_view(rho_next))
        copyto!(interior_view(state.phi), interior_view(phi_next))
    end

    @printf("  advection: nx=%d  nt=%d  max(rho)=%.4f  max(phi)=%.4f\n",
        nx, nt, maximum(state.rho), maximum(state.phi))
    return state
end

run_advection_multifield()

# Broadcast across every field at once:
state = LocalMultiHaloArray(Float64, (8,), 1; boundary_conditions=(
    a=((Repeating(), Repeating()),), b=((Repeating(), Repeating()),)))
state.a .= 1.0
state.b .= 2.0
state .*= 3.0
println("  after .*= 3:  a[1]=", state.a[1], "  b[1]=", state.b[1])

# ============================================================
# 6. THREADED ARRAYS — ThreadedHaloArray
# ============================================================
# Tiles the grid across threads; each thread owns a tile with its own ghost
# layer.  After synchronize_halo!, tiles can be updated in parallel with no
# further coordination.  Construction takes (tile_size, halo) plus a `dims`
# tile layout; inside the loop work on tile_parent(u, tile_id) — a plain
# padded array — using the shared interior_range.

section("6 — Threaded arrays (ThreadedHaloArray)")

# One explicit-Euler step.  The @tasks loop reads `u` and writes `u_next`; both
# arrive as arguments and are never reassigned here.  (Reassigning a variable an
# @tasks closure captures — like the u/u_next swap in the runner — would raise
# "Attempted to capture and modify outer local variables", so the swap stays out
# in run_heat_1d_threaded where there is no @tasks closure.)
function heat_step_threaded!(u_next, u, alpha, dt, dx, range)
    synchronize_halo!(u)
    inv_dx2 = inv(dx^2)
    e       = CartesianIndex(1)
    @tasks for tile_id in 1:tile_count(u)        # one tile per thread
        d_old  = tile_parent(u,      tile_id)
        d_next = tile_parent(u_next, tile_id)
        for I in CartesianIndices(range)
            d_next[I] = d_old[I] +
                alpha * dt * (d_old[I+e] - 2*d_old[I] + d_old[I-e]) * inv_dx2
        end
    end
    return u_next
end

function run_heat_1d_threaded(; nx=64, alpha=1.0, nt=200, cfl=0.4,
        tile_dims=(Base.Threads.nthreads(),))
    nx % tile_dims[1] == 0 ||
        throw(ArgumentError("nx must be divisible by tile_dims[1]"))

    dx        = 1.0 / nx
    dt        = cfl * dx^2 / alpha
    tile_size = (nx ÷ tile_dims[1],)

    u      = ThreadedHaloArray(Float64, tile_size, 1; dims=tile_dims, boundary_condition=:periodic)
    u_next = similar(u)
    for I in CartesianIndices(axes(u))           # logical indexing, same as LocalHaloArray
        u[Tuple(I)...] = exp(-100 * ((I[1]/nx) - 0.5)^2)
    end
    u0max = maximum(u)

    range = interior_range(u)                    # same for every tile
    for _ in 1:nt
        heat_step_threaded!(u_next, u, alpha, dt, dx, range)
        u, u_next = u_next, u                    # safe: no @tasks closure in this scope
    end

    @printf("  threaded: nx=%d  tiles=%d  nt=%d  max(u₀)=%.4f  max(u)=%.4f\n",
        nx, prod(tile_dims), nt, u0max, maximum(u))
    return u
end

run_heat_1d_threaded()

# ============================================================
# 7. INDEXED FIELD COLLECTIONS — ArrayOfHaloArray
# ============================================================
# Like LocalMultiHaloArray, but fields are accessed by integer / Cartesian
# index instead of name — handy when the count is decided at runtime or the
# fields form a matrix (e.g. a 2×2 tensor).  synchronize_halo! and broadcast
# act on every component; CellRanges/FaceRanges accept the container directly.

section("7 — ArrayOfHaloArray")

# A 2-D velocity field as a (2,) array of LocalHaloArrays:
vel = ArrayOfHaloArray(LocalHaloArray, Float64, (2,), (16, 16), 1;
    boundary_condition=:periodic)
interior_view(vel[1]) .= 1.0          # u-component
interior_view(vel[2]) .= 0.0          # v-component
synchronize_halo!(vel)                # refreshes every field at once

println("  field shape / owned size : ", field_shape(vel), " / ", owned_size(vel[1]))
println("  owned cells              : ", size(get_owned_cells(CellRanges(vel))))

# A 2×2 stress-tensor field; broadcast hits all four components:
sigma = ArrayOfHaloArray(LocalHaloArray, Float64, (2, 2), (8, 8), 1;
    boundary_condition=:repeating)
for i in 1:2, j in 1:2
    fill!(interior_view(sigma[i, j]), Float64(i == j))
end
sigma .*= 2.0
println("  sigma diag / off-diag    : ", sigma[1, 1][1, 1], " / ", sigma[1, 2][1, 1])  # 2.0 / 0.0

println()
println("Tutorial complete.")
