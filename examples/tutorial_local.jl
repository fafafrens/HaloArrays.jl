# ============================================================
# HaloArrays.jl — hands-on tutorial (no MPI)
#
# This tutorial introduces the core ideas of HaloArrays.jl
# through progressively more complex examples, all running on
# a single process.  Run the whole file or execute sections
# interactively in the REPL.
#
# Sections:
#   1. What is a halo array? (storage layout)
#   2. Boundary conditions
#   3. CellRanges and FaceRanges (loop helpers)
#   4. Finite-difference heat equation (1-D)
#   5. Multi-field arrays — LocalMultiHaloArray
#   6. Threaded arrays — ThreadedHaloArray
#   7. ArrayOfHaloArray  (indexed field collections)
# ============================================================

using HaloArrays
using Printf

# ============================================================
# 1. WHAT IS A HALO ARRAY?
# ============================================================
#
# In stencil methods each process owns a patch of the grid but
# needs a thin "halo" of ghost cells filled from its neighbours
# (or from boundary conditions) before computing a stencil.
#
# LocalHaloArray wraps a plain Julia array and tracks:
#   • how many ghost cells surround the owned region (halo_width)
#   • the boundary condition to apply when the halo is refreshed
#
# Schematic for a 1-D array with 4 owned cells and halo_width=1:
#
#   storage:  [ ghost | 1 | 2 | 3 | 4 | ghost ]
#   indices:      1       2   3   4   5     6
#
#   interior indices (owned cells): 2..5

println("=" ^ 60)
println("Section 1 — Storage layout")
println("=" ^ 60)

u1d = LocalHaloArray(Float64, (4,), 1; boundary_condition=:periodic)

println("owned size  : ", owned_size(u1d))   # (4,)
println("storage size: ", storage_size(u1d)) # (6,)  — includes ghost cells
println("halo width  : ", halo_width(u1d))   # 1

# interior_view returns a view of only the owned cells.
# Use it to fill initial data.
interior_view(u1d) .= [10.0, 20.0, 30.0, 40.0]

println("interior    : ", collect(interior_view(u1d)))   # [10, 20, 30, 40]
println("full storage: ", collect(parent(u1d)))           # ghost cells are still 0

# The logical indexing through u1d (via [], axes, eachindex) always
# refers to the owned cells — ghost offsets are hidden from the user.
println("u1d[2]      : ", u1d[2])            # 20.0
println("axes(u1d)   : ", axes(u1d))          # (1:4,)

# ============================================================
# 2. BOUNDARY CONDITIONS
# ============================================================
#
# Calling synchronize_halo! (or boundary_condition!) fills the
# ghost cells according to the chosen boundary condition.
#
# Available conditions:
#   :periodic   — ghost cells wrap around from the other side
#   :repeating  — ghost cells mirror the nearest owned value (zero-gradient)
#   Reflecting()  — ghost = -interior (reflects with sign flip)
#   Antireflecting() — ghost = interior (zero-derivative)

println()
println("=" ^ 60)
println("Section 2 — Boundary conditions")
println("=" ^ 60)

function show_bc(label, bc)
    u = LocalHaloArray(Float64, (4,), 1; boundary_condition=bc)
    interior_view(u) .= [1.0, 2.0, 3.0, 4.0]
    synchronize_halo!(u)
    @printf("  %-20s storage = %s\n", label, string(collect(parent(u))))
end

show_bc("periodic",         :periodic)
show_bc("repeating",        :repeating)
show_bc("reflecting",       ((Reflecting(), Reflecting()),))
show_bc("antireflecting",   ((Antireflecting(), Antireflecting()),))

# Multi-dimensional arrays use a tuple-of-tuples: one pair per
# spatial dimension, each pair being (left_bc, right_bc).
u2d = LocalHaloArray(Float64, (3, 3), 1;
    boundary_condition=((Periodic(), Periodic()), (Reflecting(), Reflecting())))

println("2-D mixed BC created — size=$(owned_size(u2d)), halo=$(halo_width(u2d))")

# ============================================================
# 3. CELLRANGES AND FACERANGES
# ============================================================
#
# Writing "for I in CartesianIndices(interior_range(u))" is correct
# but tedious.  CellRanges and FaceRanges precompute the index
# ranges you need and give them readable names.
#
# CellRanges — for cell-centred loops (e.g. source terms):
#   get_owned_cells(r)          CartesianIndices of all owned cells
#   get_colored_owned_cell_ranges(r, color)   checkerboard split
#                               (useful for in-place Gauss-Seidel)
#
# FaceRanges — for finite-volume flux loops:
#   get_left_face(r, dim)       ghost | owned  face on the left
#   get_internal_face(r)        all internal faces (owned | owned)
#   get_right_face(r, dim)      owned | ghost  face on the right
#   get_unit_vector(r, dim)     CartesianIndex offset across a face

println()
println("=" ^ 60)
println("Section 3 — Range helpers")
println("=" ^ 60)

u = LocalHaloArray(Float64, (6,), 1; boundary_condition=:periodic)
interior_view(u) .= Float64.(1:6)
synchronize_halo!(u)

cr = CellRanges(u)
println("owned cells          : ", collect(get_owned_cells(cr)))
println("color-0 cells        : ", collect.(get_colored_owned_cell_ranges(cr, 0)))
println("color-1 cells        : ", collect.(get_colored_owned_cell_ranges(cr, 1)))

fr = FaceRanges(u)
e  = get_unit_vector(fr, 1)
println("left-face range  dim1: ", collect(get_left_face(fr, 1)))
println("internal-face range  : ", collect(get_internal_face(fr)))
println("right-face range dim1: ", collect(get_right_face(fr, 1)))

# A typical finite-volume flux loop visits every face exactly once.
# The left and right boundary faces only update one cell (the owned
# one); internal faces update both neighbouring cells.
println()
println("Face loop demonstration (sum of all face fluxes):")
data = parent(u)
total_flux = 0.0
for IL in get_left_face(fr, 1)
    IR = IL + e
    total_flux += data[IR] - data[IL]        # upwind example
end
for IL in get_internal_face(fr)
    IR = IL + e
    total_flux += data[IR] - data[IL]
end
for IL in get_right_face(fr, 1)
    IR = IL + e
    total_flux += data[IR] - data[IL]
end
println("  total_flux = ", total_flux)

# ============================================================
# 4. FINITE-DIFFERENCE HEAT EQUATION (1-D)
# ============================================================
#
# We solve  ∂u/∂t = α ∂²u/∂x²  on [0,1] with periodic BCs
# using an explicit Euler step.
#
# Key workflow for every time step:
#   1. synchronize_halo!(u)    — fill ghost cells
#   2. read parent(u) with ghost-safe stencil indices
#   3. write new values into parent(u_next) at interior indices

println()
println("=" ^ 60)
println("Section 4 — Heat equation (1-D, LocalHaloArray)")
println("=" ^ 60)

function heat_rhs_1d!(du, u, alpha, dx)
    data  = parent(u)
    ddata = parent(du)
    inv_dx2 = inv(dx^2)
    e = CartesianIndex(1)
    for I in CartesianIndices(interior_range(u))
        ddata[I] = alpha * (data[I+e] - 2*data[I] + data[I-e]) * inv_dx2
    end
    return du
end

function heat_step_local!(u_next, u, du, alpha, dt, dx)
    synchronize_halo!(u)
    heat_rhs_1d!(du, u, alpha, dx)
    interior_view(u_next) .= interior_view(u) .+ dt .* interior_view(du)
    return u_next
end

function run_heat_1d(; nx=64, alpha=1.0, nt=200, cfl=0.4)
    dx = 1.0 / nx
    dt = cfl * dx^2 / alpha

    u     = LocalHaloArray(Float64, (nx,), 1; boundary_condition=:periodic)
    u_next = similar(u)
    du    = similar(u)

    # Gaussian initial condition
    for i in 1:nx
        u[i] = exp(-100 * ((i/nx) - 0.5)^2)
    end
    synchronize_halo!(u)
    u0max = maximum(u)

    for _ in 1:nt
        heat_step_local!(u_next, u, du, alpha, dt, dx)
        u, u_next = u_next, u
    end
    synchronize_halo!(u)

    @printf("  nx=%d  nt=%d  dt=%.2e  max(u₀)=%.4f  max(u)=%.4f\n",
        nx, nt, dt, u0max, maximum(u))
    return u
end

u_heat = run_heat_1d()

# ============================================================
# 5. MULTI-FIELD ARRAYS — LocalMultiHaloArray
# ============================================================
#
# When evolving several fields on the same grid (e.g. density,
# velocity, pressure) it is more convenient to group them into a
# single container than to manage separate arrays.
#
# LocalMultiHaloArray creates a named-tuple of LocalHaloArrays,
# all sharing the same geometry.  You can:
#   • access individual fields with  state.rho,  state.vel, …
#   • call synchronize_halo!(state)  to refresh all fields at once
#   • broadcast over all fields simultaneously with  state .* 2

println()
println("=" ^ 60)
println("Section 5 — Multi-field arrays (LocalMultiHaloArray)")
println("=" ^ 60)

# Advection of density and a passive scalar on a 1-D grid
function run_advection_multifield(; nx=64, nt=100, cfl=0.8)
    dx = 1.0 / nx
    a  = 1.0          # advection speed
    dt = cfl * dx / a

    state = LocalMultiHaloArray(
        Float64, (nx,), 1;
        boundary_conditions = (
            rho = ((Periodic(), Periodic()),),
            phi = ((Periodic(), Periodic()),),
        ),
    )

    # Set initial conditions for each field independently
    for i in 1:nx
        x = (i - 0.5) / nx
        state.rho[i] = 1.0 + exp(-100*(x - 0.25)^2)
        state.phi[i] = sin(2π * x)
    end
    synchronize_halo!(state)

    rho_next = similar(state.rho)
    phi_next = similar(state.phi)
    e = CartesianIndex(1)

    for _ in 1:nt
        synchronize_halo!(state)
        rho_data = parent(state.rho)
        phi_data = parent(state.phi)
        r = interior_range(state.rho)

        # First-order upwind (a > 0 → left-biased)
        for I in CartesianIndices(r)
            rho_next[I] = state.rho[I] - (a*dt/dx) * (rho_data[I] - rho_data[I-e])
            phi_next[I] = state.phi[I] - (a*dt/dx) * (phi_data[I] - phi_data[I-e])
        end

        copyto!(interior_view(state.rho), interior_view(rho_next))
        copyto!(interior_view(state.phi), interior_view(phi_next))
    end
    synchronize_halo!(state)

    rho_max = maximum(state.rho)
    phi_max = maximum(state.phi)
    @printf("  advection: nx=%d  nt=%d  max(rho)=%.4f  max(phi)=%.4f\n",
        nx, nt, rho_max, phi_max)
    return state
end

run_advection_multifield()

# You can broadcast across all fields at once:
state = LocalMultiHaloArray(Float64, (8,), 1;
    boundary_conditions=(a=((Repeating(), Repeating()),), b=((Repeating(), Repeating()),)))
state.a .= 1.0
state.b .= 2.0
state .*= 3.0     # multiplies every field in-place
println("  After broadcast *= 3:  a[1]=", state.a[1], "  b[1]=", state.b[1])

# ============================================================
# 6. THREADED ARRAYS — ThreadedHaloArray
# ============================================================
#
# ThreadedHaloArray tiles the global grid across threads.  Each
# thread owns a rectangular tile; tiles share ghost layers so that
# stencil access across a tile boundary works without explicit
# synchronisation (after synchronize_halo! has run).
#
# Construction replaces (owned_dims, halo) with (tile_size, halo)
# and adds a dims keyword for the tile layout:
#
#   ThreadedHaloArray(T, tile_size, halo; dims=tile_layout, ...)
#
# Everything else (interior_range, owned_size, …) behaves the same.
# Inside a tile loop you work on tile_parent(u, tile_id) — a plain
# Julia array with ghost padding — using the same interior_range.

println()
println("=" ^ 60)
println("Section 6 — Threaded arrays (ThreadedHaloArray)")
println("=" ^ 60)

using OhMyThreads: tforeach, @tasks

function run_heat_1d_threaded(; nx=64, alpha=1.0, nt=200, cfl=0.4,
        tile_dims=(Base.Threads.nthreads(),))

    nx % tile_dims[1] == 0 ||
        throw(ArgumentError("nx must be divisible by tile_dims[1]"))

    dx        = 1.0 / nx
    dt        = cfl * dx^2 / alpha
    tile_size = (nx ÷ tile_dims[1],)

    u      = ThreadedHaloArray(Float64, tile_size, 1; dims=tile_dims, boundary_condition=:periodic)
    u_next = similar(u)

    # Set initial condition (logical indexing works the same way)
    for I in CartesianIndices(axes(u))
        u[Tuple(I)...] = exp(-100 * ((I[1]/nx) - 0.5)^2)
    end
    synchronize_halo!(u)
    u0max = maximum(u)

    range   = interior_range(u)    # same for all tiles
    inv_dx2 = inv(dx^2)
    e       = CartesianIndex(1)

    for _ in 1:nt
        synchronize_halo!(u)

        # Each tile is processed by one thread in parallel
        @tasks for tile_id in 1:tile_count(u)
            d_old  = tile_parent(u,      tile_id)
            d_next = tile_parent(u_next, tile_id)
            for I in CartesianIndices(range)
                d_next[I] = d_old[I] +
                    alpha * dt * (d_old[I+e] - 2*d_old[I] + d_old[I-e]) * inv_dx2
            end
        end

        u, u_next = u_next, u
    end
    synchronize_halo!(u)

    @printf("  threaded: nx=%d  tiles=%d  nt=%d  max(u₀)=%.4f  max(u)=%.4f\n",
        nx, prod(tile_dims), nt, u0max, maximum(u))
    return u
end

run_heat_1d_threaded()

# synchronize_halo! on a ThreadedHaloArray does two things:
#   (a) copies boundary data between adjacent tiles (thread-local)
#   (b) applies the boundary condition at the domain edges
# No MPI communication is involved.

# ============================================================
# 7. ARRAYOFHALOARRAY — indexed collection of fields
# ============================================================
#
# ArrayOfHaloArray holds an AbstractArray of AbstractSingleHaloArrays
# all with the same geometry.  Unlike MultiHaloArray (which uses
# named fields), here fields are accessed by integer or Cartesian
# index.  This is convenient when the number of fields is determined
# at runtime or when fields are organised in a matrix layout
# (e.g. a 2×2 tensor).

println()
println("=" ^ 60)
println("Section 7 — ArrayOfHaloArray")
println("=" ^ 60)

# A 2-D velocity field stored as a (2,) array of LocalHaloArrays:
# vel[1] = u-component,  vel[2] = v-component
vel = ArrayOfHaloArray(LocalHaloArray, Float64, (2,), (16, 16), 1;
    boundary_condition=:periodic)

# Fill each component independently
interior_view(vel[1]) .= 1.0
interior_view(vel[2]) .= 0.0

synchronize_halo!(vel)     # refreshes every field at once

println("  vel shape  : ", field_shape(vel))          # (2,)
println("  owned size : ", owned_size(vel[1]))         # (16,16)
println("  halo width : ", halo_width(vel))            # 1

# CellRanges and FaceRanges work directly on the container —
# no need to index into vel[1] just to get the geometry.
cr = CellRanges(vel)
println("  owned cells: ", size(get_owned_cells(cr)))  # (16,16)

# A 2×2 stress-tensor field:
sigma = ArrayOfHaloArray(LocalHaloArray, Float64, (2, 2), (8, 8), 1;
    boundary_condition=:repeating)

for i in 1:2, j in 1:2
    fill!(interior_view(sigma[i, j]), Float64(i == j ? 1.0 : 0.0))
end
println("  sigma[1,1][1,1] = ", sigma[1, 1][1, 1])   # 1.0  (diagonal)
println("  sigma[1,2][1,1] = ", sigma[1, 2][1, 1])   # 0.0  (off-diagonal)

# Broadcast works across all components simultaneously:
sigma .*= 2.0
println("  After *= 2:  sigma[1,1][1,1] = ", sigma[1, 1][1, 1])   # 2.0

println()
println("Tutorial complete.")
