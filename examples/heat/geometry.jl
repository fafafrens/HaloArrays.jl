using Printf
using StaticArrays
using OhMyThreads: @tasks

include("common.jl")

# Heat diffusion as a conservative finite-volume scheme on a cell geometry.
#
# The update is the same in every coordinate system:
#
#     u_I += α dt / V_I · Σ_faces  A_f · (u_neighbour − u_I) / d_f
#
# with the cell volume V_I, the face area A_f and the centre-to-centre distance
# d_f read from a `cell_geometry`. On a uniform Cartesian grid this reduces to
# the finite-difference stencil of common.jl (checked below to round-off); on a
# polar disk with a graded radial axis the same kernel conserves the heat
# content Σ u V exactly, with no special treatment of the axis (its faces have
# zero area). One kernel serves LocalHaloArray and ThreadedHaloArray: the
# geometry helpers take the tile id, like `siteview`.

# ---- kernel -----------------------------------------------------------------

# Net diffusive flux into cell I through its 2N faces, summed by recursing on
# the direction as a static parameter, so `Dim(D)` is a compile-time constant
# (an `ntuple(d -> …, Val(N))` closure would not fold it and would dispatch
# dynamically on every face).
@inline _face_fluxes(old, g, sys, I::CartesianIndex, tile, ::Val{0}) = zero(eltype(old))
@inline function _face_fluxes(old, g, sys, I::CartesianIndex{N}, tile, ::Val{D}) where {N,D}
    e = unit_vector(Val(N), D)
    Ip, Im = I + e, I - e
    f = @inbounds (face_area(sys, g, Dim(D), I, tile) * (old[Ip] - old[I]) / face_distance(g, Dim(D), I, tile) -
                   face_area(sys, g, Dim(D), Im, tile) * (old[I] - old[Im]) / face_distance(g, Dim(D), Im, tile))
    return f + _face_fluxes(old, g, sys, I, tile, Val(D - 1))
end

# Net diffusive flux into cell I per unit volume.
@inline fv_laplacian(old, g, sys, I, tile, ::Val{N}) where {N} =
    _face_fluxes(old, g, sys, I, tile, Val(N)) / cell_volume(sys, g, I, tile)

function _fv_heat_tile!(next, old, g, sys, alpha, dt, rng, tile, ::Val{N}) where {N}
    @inbounds for I in CartesianIndices(rng)
        next[I] = old[I] + alpha * dt * fv_laplacian(old, g, sys, I, tile, Val(N))
    end
    return next
end

# The tile id the geometry helpers need: a tile index on a threaded array,
# `nothing` on a single block (the one "tile" of tile_parent).
_geometry_tile(::ThreadedHaloArray, t) = t
_geometry_tile(u, t) = nothing

function fv_heat_step!(u_next, u_old, g, sys, alpha, dt)
    N   = ndims(u_old)
    rng = interior_range(u_old)
    @tasks for t in 1:tile_count(u_old)
        _fv_heat_tile!(tile_parent(u_next, t), tile_parent(u_old, t), g, sys, alpha, dt,
            rng, _geometry_tile(u_old, t), Val(N))
    end
    return u_next
end

# Σ_faces A_f / d_f of cell I (same recursion as the fluxes).
@inline _face_weights(g, sys, I::CartesianIndex, tile, ::Val{0}) = 0.0
@inline function _face_weights(g, sys, I::CartesianIndex{N}, tile, ::Val{D}) where {N,D}
    Im = I - unit_vector(Val(N), D)
    w = face_area(sys, g, Dim(D), I, tile) / face_distance(g, Dim(D), I, tile) +
        face_area(sys, g, Dim(D), Im, tile) / face_distance(g, Dim(D), Im, tile)
    return w + _face_weights(g, sys, I, tile, Val(D - 1))
end

# Explicit Euler is stable while α dt Σ_faces A_f / (d_f V_I) ≤ 1 in every cell.
function fv_stable_dt(u, g, sys, alpha; cfl=0.8)
    N = ndims(u)
    worst = 0.0
    for t in 1:tile_count(u)
        tile = _geometry_tile(u, t)
        for I in CartesianIndices(interior_range(u))
            worst = max(worst, _face_weights(g, sys, I, tile, Val(N)) / cell_volume(sys, g, I, tile))
        end
    end
    return cfl / (alpha * worst)
end

function solve_fv_heat!(u, g, sys; alpha, dt, nt)
    current, next = u, similar(u)
    for _ in 1:nt
        synchronize_halo!(current)
        fv_heat_step!(next, current, g, sys, alpha, dt)
        current, next = next, current
    end
    synchronize_halo!(current)
    current === u || copyto!(u, current)
    return u
end

# ---- geometry-aware initial condition and diagnostics -----------------------

# Position of a cell centre in the embedding plane, for a Gaussian bump that
# means the same thing on a Cartesian grid and on a polar disk.
_embed(::Cartesian, c) = c
_embed(::Polar, c) = SVector(c[1] * cos(c[2]), c[1] * sin(c[2]))

function fill_gaussian!(u, g, sys, x0; width, baseline=1.0, amplitude=1.0)
    for t in 1:tile_count(u)
        tile = _geometry_tile(u, t)
        data = tile_parent(u, t)
        for I in CartesianIndices(interior_range(u))
            x = _embed(sys, cell_center(sys, g, I, tile))
            data[I] = baseline + amplitude * exp(-sum(abs2, x - x0) / width^2)
        end
    end
    synchronize_halo!(u)
    return u
end

# Σ u V over the interior: the conserved quantity of the scheme.
function heat_content(u, g, sys)
    total = 0.0
    for t in 1:tile_count(u)
        tile = _geometry_tile(u, t)
        data = tile_parent(u, t)
        for I in CartesianIndices(interior_range(u))
            total += data[I] * cell_volume(sys, g, I, tile)
        end
    end
    return total
end

# ---- (1) uniform Cartesian grid: finite volume == finite difference ---------

function cartesian_check(; n=(64, 64), nt=100, alpha=1.0, cfl=0.2)
    dx = ntuple(d -> 1.0 / n[d], Val(2))
    dt = stable_heat_dt(alpha, cfl, dx)

    u_fd = LocalHaloArray(Float64, n, 1; boundary_condition=:periodic)
    fill_centered_gaussian!(u_fd; baseline=1.0, amplitude=1.0)
    u_fv = copy(u_fd)
    g    = cell_geometry(u_fv, UniformAxis(0, 1), UniformAxis(0, 1))

    solve_heat!(u_fd; alpha, dt, dx, nt)                          # common.jl stencil
    solve_fv_heat!(u_fv, g, Cartesian(); alpha, dt, nt)           # geometry kernel
    return maximum(abs.(interior_view(u_fd) .- interior_view(u_fv)))
end

# ---- (2) polar disk, graded radial axis, local and threaded -----------------

const DISK_AXES = (EdgeAxis(cell_edges((0, 0.2, 1), (16, 32))), UniformAxis(0, 2π))
const DISK_BC   = ((Reflecting(), Reflecting()), (Periodic(), Periodic()))   # no flux at the rim

function disk_run(u; nt=200, alpha=0.05)
    sys = Polar()
    g   = cell_geometry(u, DISK_AXES; system=sys)
    fill_gaussian!(u, g, sys, SVector(0.4, 0.0); width=0.15)
    before = heat_content(u, g, sys)
    dt = fv_stable_dt(u, g, sys, alpha)
    solve_fv_heat!(u, g, sys; alpha, dt, nt)
    return u, before, heat_content(u, g, sys), dt
end

function main()
    err = cartesian_check()
    @printf("Cartesian 64×64:  |finite volume − finite difference| = %.2e\n", err)

    lu, b, a, dt = disk_run(LocalHaloArray(Float64, (48, 64), 1; boundary_condition=DISK_BC))
    @printf("polar disk Local:    dt=%.3e  heat content %.12f -> %.12f  (rel. change %.1e)\n",
        dt, b, a, (a - b) / b)

    tu, tb, ta, _ = disk_run(ThreadedHaloArray(Float64, (48, 32), 1; dims=(1, 2), boundary_condition=DISK_BC))
    diff = maximum(abs.(interior_view(LocalHaloArray(tu)) .- interior_view(lu)))
    @printf("polar disk Threaded: tiles=%d  heat content %.12f -> %.12f  |threaded − local| = %.2e\n",
        tile_count(tu), tb, ta, diff)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
