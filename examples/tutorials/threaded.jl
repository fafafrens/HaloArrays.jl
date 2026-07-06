# ============================================================
# HaloArrays.jl — ThreadedHaloArray tutorial
#
# Run with:
#   julia --project=. -t 4 examples/tutorials/threaded.jl
#
# Sections:
#   1. ThreadedHaloArray — tiled shared-memory layout
#   2. Tile loop pattern
#   3. Synchronisation — halo exchange between tiles
#   4. Threaded finite-volume Burgers equation (1-D)
#   5. ThreadedMultiHaloArray — multiple fields
#   6. ArrayOfHaloArray with ThreadedHaloArray fields
# ============================================================

using HaloArrays
using OhMyThreads: tforeach, tmapreduce, @tasks
using Printf

println("Julia threads available: ", Threads.nthreads())

# ============================================================
# 1. ThreadedHaloArray — TILED SHARED-MEMORY LAYOUT
# ============================================================
#
# ThreadedHaloArray divides the global grid into rectangular tiles,
# one per thread (or any multiple thereof).  Each tile is stored
# as a separate Julia array with its own ghost cells, so threads
# can read/write their tile independently after synchronisation.
#
#   Global grid  4 x 1  (4 cells, 1 thread)  halo=1:
#
#     tile 1:  [ ghost | 1 | 2 | 3 | 4 | ghost ]
#
#   Global grid  4 x 1  (4 cells, 2 threads)  halo=1:
#
#     tile 1:  [ ghost | 1 | 2 | ghost ]
#     tile 2:  [ ghost | 3 | 4 | ghost ]
#
# Construction:
#   ThreadedHaloArray(T, tile_size, halo; dims=tile_layout, boundary_condition)
#
#   tile_size   — cells per tile in each dimension
#   dims        — number of tiles in each dimension (tile layout)
#   halo        — ghost-cell width

println("=" ^ 60)
println("Section 1 — ThreadedHaloArray layout")
println("=" ^ 60)

nthreads  = max(1, Threads.nthreads())
tile_dims = (nthreads,)    # 1-D layout: one tile per thread
nx        = 8 * nthreads   # global grid cells

tile_size = (nx ÷ tile_dims[1],)

u = ThreadedHaloArray(Float64, tile_size, 1; dims=tile_dims, boundary_condition=:periodic)

println("tile_size   : ", tile_size)
println("tile_dims   : ", tile_dims)
println("tile_count  : ", tile_count(u))
println("interior_size  : ", interior_size(u))    # full global interior extent
println("global_size : ", global_size(u))
println("halo_width  : ", halo_width(u))
println("storage per tile: ", storage_size(u))   # tile_size + 2*halo in each dim

# ============================================================
# 2. TILE LOOP PATTERN
# ============================================================
#
# Work on a ThreadedHaloArray by iterating over tile IDs.
# tile_parent(u, tile_id) returns the raw storage array for that
# tile, including its ghost cells.  interior_range(u) gives the
# ghost-free index range — it is the same for every tile.

println()
println("=" ^ 60)
println("Section 2 — Tile loop pattern")
println("=" ^ 60)

# Fill each tile sequentially (single-threaded, for demonstration)
range = interior_range(u)   # same CartesianIndices for all tiles

for tile_id in 1:tile_count(u)
    data = tile_parent(u, tile_id)
    coords = tile_coordinates(u, tile_id)
    for I in CartesianIndices(range)
        # Compute a tile-local index relative to global start of this tile
        global_i = (coords[1] - 1) * tile_size[1] + I[1]
        data[I] = Float64(global_i)
    end
end

println("Tile coordinates (tile_id => coords):")
for t in 1:tile_count(u)
    println("  tile $t => ", tile_coordinates(u, t))
end

# tile_parent and tile_count also work on collections (new fallback):
vel = ArrayOfHaloArray(ThreadedHaloArray, Float64, (2,), tile_size, 1;
    dims=tile_dims, boundary_condition=:periodic)
println("tile_count on ArrayOfHaloArray: ", tile_count(vel))

# ============================================================
# 3. SYNCHRONISATION — HALO EXCHANGE BETWEEN TILES
# ============================================================
#
# synchronize_halo! on a ThreadedHaloArray:
#   (a) copies boundary data between adjacent tiles (in memory)
#   (b) applies boundary conditions at the domain edges
#
# This must be called before any stencil that reads ghost cells.
# The exchange is a "pull": each tile copies its neighbours' interior
# edges into its OWN ghost cells.  So a tile only writes the ghost
# region it owns and only reads interior cells (which nobody writes),
# which is why a parallel sweep over tiles is race-free.
#
# synchronize_halo! itself is SERIAL (a plain loop over tiles).  A
# parallel variant exists — synchronize_halo_threads! — but for the
# usual case (halo width 1, tiles ≈ threads) the per-tile exchange is
# tiny and the task-spawn overhead dominates: benchmarks show the
# serial version winning by 7–25×, and allocating nothing.  Reach for
# synchronize_halo_threads! only when the exchange is genuinely large
# (3-D domains, wide halos, many tiles).  See
# benchmark/threaded_sync_variants.jl.

println()
println("=" ^ 60)
println("Section 3 — Halo exchange between tiles")
println("=" ^ 60)

synchronize_halo!(u)   # serial on purpose — see note above

# Verify: the ghost cell of tile 2 (left side) should equal the
# last interior cell of tile 1.
if tile_count(u) >= 2
    tile1_last_interior  = tile_parent(u, 1)[last(range[1])]
    tile2_left_ghost  = tile_parent(u, 2)[1]          # ghost at storage index 1
    println("tile 1 last interior cell : ", tile1_last_interior)
    println("tile 2 left ghost cell : ", tile2_left_ghost)
    println("match: ", tile1_last_interior == tile2_left_ghost)
end

# Async halo exchange (overlap sync with independent computation)
start_halo_exchange!(u)
# ... ghost-free local work here ...
finish_halo_exchange!(u)
boundary_condition!(u)

# ============================================================
# 4. THREADED FINITE-VOLUME BURGERS EQUATION (1-D)
# ============================================================
#
# ∂u/∂t + ∂(u²/2)/∂x = 0   on [0,1] periodic
# using an explicit Euler + Rusanov flux scheme.
#
# The parallel loop uses @tasks (OhMyThreads) to dispatch each
# tile to a thread.  Note the absence of synchronisation inside
# the loop — each thread reads its own tile's ghost cells (filled
# by synchronize_halo! before the loop) and writes only to its
# own tile's interior.

println()
println("=" ^ 60)
println("Section 4 — Threaded Burgers equation (1-D)")
println("=" ^ 60)

@inline rusanov_flux(ul, ur) =
    0.5*(0.5*ul^2 + 0.5*ur^2) - 0.5*max(abs(ul), abs(ur))*(ur - ul)

function burgers_rhs_tile!(du_data, u_data, ranges::FaceRanges, invdx)
    e = unit_vector(ranges, 1)
    for IL in interior_faces(ranges, 1)
        IR = IL + e
        f = rusanov_flux(u_data[IL], u_data[IR]) * invdx
        du_data[IL] -= f
        du_data[IR] += f
    end
    return du_data
end

function burgers_step_threaded!(u_next, u, du, dt, dx)
    fill!(du, 0)
    synchronize_halo!(u)

    fr    = FaceRanges(u)
    range = interior_range(u)
    invdx = inv(dx)

    @tasks for tile_id in 1:tile_count(u)
        burgers_rhs_tile!(tile_parent(du, tile_id), tile_parent(u, tile_id), fr, invdx)
    end

    @tasks for tile_id in 1:tile_count(u)
        d_old  = tile_parent(u,      tile_id)
        d_next = tile_parent(u_next, tile_id)
        du_d   = tile_parent(du,     tile_id)
        for I in CartesianIndices(range)
            d_next[I] = d_old[I] + dt * du_d[I]
        end
    end

    return u_next
end

function run_burgers_threaded(; nx_global=200, steps=300, cfl=0.4,
        tile_dims=(max(1, Threads.nthreads()),))
    nx_global % tile_dims[1] == 0 ||
        throw(ArgumentError("nx_global must be divisible by tile_dims[1]"))

    ts   = (nx_global ÷ tile_dims[1],)
    u    = ThreadedHaloArray(Float64, ts, 1; dims=tile_dims, boundary_condition=:periodic)
    u_nxt = similar(u)
    du   = similar(u)

    dx = 1.0 / nx_global
    dt = cfl * dx / 1.5

    for I in CartesianIndices(axes(u))
        x = (I[1] - 0.5) / nx_global
        u[Tuple(I)...] = 0.5 + exp(-100*(x - 0.35)^2)
    end
    synchronize_halo!(u)
    m0 = sum(u)

    cur, nxt = u, u_nxt
    for _ in 1:steps
        burgers_step_threaded!(nxt, cur, du, dt, dx)
        cur, nxt = nxt, cur
    end
    synchronize_halo!(cur)

    @printf("  threads=%d  nx=%d  steps=%d  max=%.4f  mass_err=%.2e\n",
        nthreads, nx_global, steps, maximum(cur), sum(cur) - m0)
    return cur
end

run_burgers_threaded()

# ============================================================
# 5. ThreadedMultiHaloArray — MULTIPLE FIELDS
# ============================================================
#
# ThreadedMultiHaloArray is a named-tuple container of
# ThreadedHaloArrays all sharing the same tile layout.
# synchronize_halo! exchanges every field in a single call.

println()
println("=" ^ 60)
println("Section 5 — ThreadedMultiHaloArray")
println("=" ^ 60)

state = ThreadedMultiHaloArray(Float64, tile_size, 1;
    dims=tile_dims,
    boundary_conditions=(
        rho = ((Periodic(), Periodic()),),
        vel = ((Periodic(), Periodic()),),
    ))

for I in CartesianIndices(axes(state.rho))
    state.rho[Tuple(I)...] = 1.0
    state.vel[Tuple(I)...] = Float64(I[1]) / nx
end
synchronize_halo!(state)        # exchanges rho AND vel in one call

println("tile_count(state) : ", tile_count(state))
println("max(rho)          : ", maximum(state.rho))
println("max(vel)          : ", maximum(state.vel))

# ============================================================
# 6. ArrayOfHaloArray WITH ThreadedHaloArray FIELDS
# ============================================================
#
# For a runtime-sized or matrix-indexed set of fields use
# ArrayOfHaloArray.  tile_count, tile_size, tile_coordinates,
# and neighbor_tile_id all delegate to the first field, so you
# no longer need to write arr[1].

println()
println("=" ^ 60)
println("Section 6 — ArrayOfHaloArray + ThreadedHaloArray")
println("=" ^ 60)

# A 2-component velocity vector field
vel2 = ArrayOfHaloArray(ThreadedHaloArray, Float64, (2,), tile_size, 1;
    dims=tile_dims, boundary_condition=:periodic)

for tile_id in 1:tile_count(vel2)
    interior_view(vel2[1], tile_id) .= 1.0
    interior_view(vel2[2], tile_id) .= 0.0
end
synchronize_halo!(vel2)

# tile_count, CellRanges, FaceRanges all accept the container directly
println("tile_count  : ", tile_count(vel2))
println("interior cells : ", size(interior_cells(CellRanges(vel2))))
println("face ranges : ", interior_faces(FaceRanges(vel2), 1))

println()
println("Threaded tutorial complete.")
