# ============================================================
# 1-D SR Hydrodynamics — cylindrical radial, geometric source, THREADED
#
#   julia -t auto --project=. \
#       examples/finite_volume/relativistic_hydro_cylindrical_threaded_1d.jl
#
# Threaded counterpart of relativistic_hydro_cylindrical_1d.jl. Same physics
# (ultrarelativistic gas, conserved (N, M, E), geometric source S=-(α/r)(Nv,Mv,M),
# axis BC), but the state is a ThreadedHaloArray decomposed into tiles along the
# radius, and the per-cell work (source + flux) runs per tile in parallel.
#
# What changes versus the serial/MPI version:
#  * Threading does NOT hide communication — the inter-tile halo exchange is a
#    synchronous shared-memory copy (no async transfer to overlap). The win is
#    parallelism of the per-cell loops, via tile_foreach over the tiles.
#  * Each cell's radius needs its GLOBAL index. With the tile decomposition that
#    is interior_to_global_index(field, tile_id, owned_idx) — the per-tile mapping,
#    uniform with HaloArray/LocalHaloArray.
#  * The inter-tile face flux is not double-counted: each tile updates only its
#    owned cells, so a shared face contributes -F to one tile's last cell and +F
#    to the neighbour's first cell — exactly conservative.
#
# Self-contained: HaloArrays, Printf, StaticArrays.
# ============================================================

using HaloArrays
using Printf
using StaticArrays

const GEOMETRY = 1.0     # α: 1 cylindrical, 2 spherical, 0 planar

# ─── Equation of state: ultrarelativistic classical ideal gas ─────────────────

struct UltraRelGas
    A::Float64
end

@inline _exp_muT(μ, T) = exp(clamp(μ / T, -700.0, 700.0))

@inline pressure(eos::UltraRelGas, T, μ)         = eos.A * T^4 * _exp_muT(μ, T)
@inline charge_density(eos::UltraRelGas, T, μ)   = eos.A * T^3 * _exp_muT(μ, T)
@inline enthalpy_density(eos::UltraRelGas, T, μ) = 4.0 * pressure(eos, T, μ)
@inline sound_speed(::UltraRelGas)               = 1.0 / sqrt(3.0)

# ─── Primitive ↔ conserved (N = nW, M = wW²v, E = wW² − p) ─────────────────────

@inline function cons_from_prim(eos, T, μ, v)
    W = 1.0 / sqrt(1.0 - v^2)
    p = pressure(eos, T, μ)
    n = charge_density(eos, T, μ)
    w = enthalpy_density(eos, T, μ)
    return SVector(n * W, w * W^2 * v, w * W^2 - p)
end

@inline function _residuals(eos, E, M2, N, T, μ)
    p = pressure(eos, T, μ)
    n = charge_density(eos, T, μ)
    w = enthalpy_density(eos, T, μ)
    X = E + p
    Z = sqrt(max(X^2 - M2, 1.0e-30))
    W = X / Z
    return n * W - N, w * W^2 - X
end

function prim_from_cons(eos, U; maxit=200, tol=1.0e-11)
    N, M, E = U
    M2 = M^2
    T = max(E / (3.0 * max(N, 1.0e-12)), 1.0e-8)
    μ = T * log(max(max(N, 1.0e-12) / (eos.A * T^3), 1.0e-300))
    for _ in 1:maxit
        R1, R2 = _residuals(eos, E, M2, N, T, μ)
        δT = max(1.0e-7 * abs(T), 1.0e-9)
        δμ = max(1.0e-7 * abs(μ), 1.0e-9)
        R1T, R2T = _residuals(eos, E, M2, N, T + δT, μ)
        R1m, R2m = _residuals(eos, E, M2, N, T, μ + δμ)
        J11 = (R1T - R1) / δT;  J21 = (R2T - R2) / δT
        J12 = (R1m - R1) / δμ;  J22 = (R2m - R2) / δμ
        det = J11 * J22 - J12 * J21
        abs(det) < 1.0e-300 && break
        dT = (-R1 * J22 + R2 * J12) / det
        dμ = ( R1 * J21 - R2 * J11) / det
        T  = max(T + dT, 1.0e-10)
        μ  = μ + dμ
        (abs(dT) < tol * (abs(T) + 1.0e-12) && abs(dμ) < tol * (abs(μ) + 1.0e-12)) && break
    end
    p = pressure(eos, T, μ)
    v = M / (E + p)
    return T, μ, v
end

# ─── Fluxes ───────────────────────────────────────────────────────────────────

@inline function physical_flux(eos, U)
    T, μ, v = prim_from_cons(eos, U)
    p = pressure(eos, T, μ)
    return SVector(U[1] * v, U[2] * v + p, U[2])
end

@inline function max_wave_speed(eos, U)
    _, _, v = prim_from_cons(eos, U)
    cs = sound_speed(eos)
    return max(abs((v - cs) / (1.0 - v * cs)), abs((v + cs) / (1.0 + v * cs)))
end

@inline function rusanov_flux(eos, UL, UR)
    smax = max(max_wave_speed(eos, UL), max_wave_speed(eos, UR))
    return 0.5 * (physical_flux(eos, UL) + physical_flux(eos, UR)) - 0.5 * smax * (UR - UL)
end

# ─── Field access (works on a tile's NamedTuple of arrays) ────────────────────

@inline conserved_cell(d, I) = SVector(d.N[I], d.M[I], d.E[I])

@inline function add_conserved!(d, I, scale, U)
    d.N[I] += scale * U[1]
    d.M[I] += scale * U[2]
    d.E[I] += scale * U[3]
    return d
end

# ─── Per-tile geometric source (SETS du on one tile's owned cells) ────────────
#
# `r_of(I)` maps a storage-frame owned index in this tile to its physical radius.

@inline function set_geometric_source_tile!(du_data, u_data, eos, cells, r_of)
    @inbounds for I in cells
        U = conserved_cell(u_data, I)
        _, _, v = prim_from_cons(eos, U)
        r = r_of(I)
        S = SVector(U[1] * v, U[2] * v, U[2]) * (-GEOMETRY / r)
        du_data.N[I] = S[1]
        du_data.M[I] = S[2]
        du_data.E[I] = S[3]
    end
    return du_data
end

# ─── RHS: per-tile source + flux in parallel ──────────────────────────────────

function rel_rhs!(du, u, eos, r_min, dr)
    synchronize_halo!(u)                       # inter-tile copies + axis/outflow BC
    ranges = FaceRanges(u)
    cells  = get_interior_cells(CellRanges(u))    # per-tile owned cells (same each tile)
    h      = halo_width(u.N)

    tile_foreach(thread_backend(u.N), tile_id -> begin
        du_data = tile_parent(du, tile_id)     # NamedTuple of this tile's arrays
        u_data  = tile_parent(u, tile_id)
        # global index of this tile's first owned cell − 1, via the uniform API
        offset  = interior_to_global_index(u.N, tile_id, (1,))[1] - 1
        set_geometric_source_tile!(du_data, u_data, eos, cells,
            I -> r_min + (offset + (I[1] - h) - 0.5) * dr)        # du = S(U, r)
        accumulate_flux_divergence!(du_data, u_data, ranges, 1, inv(dr),
            (UL, UR) -> rusanov_flux(eos, UL, UR), conserved_cell, add_conserved!)  # += -∂_r F
    end, 1:tile_count(u); scheduler=:static)

    return du
end

function ssprk2_step!(u, u1, du, eos, r_min, dr, dt)
    rel_rhs!(du, u, eos, r_min, dr)
    @. u1 = u + dt * du
    rel_rhs!(du, u1, eos, r_min, dr)
    @. u = 0.5 * u + 0.5 * (u1 + dt * du)
    return u
end

# ─── Diagnostics (serial reduction over tiles; area-weighted integrals) ───────

function diagnostics(u, eos, r_min, dr)
    h = halo_width(u.N)
    cells = get_interior_cells(CellRanges(u))
    charge = 0.0; energy = 0.0; vmax = 0.0
    for tile_id in 1:tile_count(u)
        d = tile_parent(u, tile_id)
        offset = interior_to_global_index(u.N, tile_id, (1,))[1] - 1
        @inbounds for I in cells
            U = conserved_cell(d, I)
            _, _, v = prim_from_cons(eos, U)
            r = r_min + (offset + (I[1] - h) - 0.5) * dr
            charge += U[1] * r * dr
            energy += U[3] * r * dr
            vmax    = max(vmax, abs(v))
        end
    end
    return charge, energy, vmax
end

# ─── Driver ───────────────────────────────────────────────────────────────────

function run_cylindrical_blast_threaded(; A=1.0, nx=400, ntiles=max(1, Threads.nthreads()),
        cfl=0.4, r_min=0.0, r_max=1.0, r_mid=0.3, t_end=0.40)
    nx % ntiles == 0 || error("nx=$nx must be divisible by ntiles=$ntiles")
    eos = UltraRelGas(A)
    dr  = (r_max - r_min) / nx

    # Axis (inner): N, E even → reflecting; M odd → antireflecting. Outer: outflow.
    u  = ThreadedMultiHaloArray(Float64, (nx ÷ ntiles,), 1; dims=(ntiles,),
        boundary_conditions=(
            N=((:reflecting, :repeating),),
            M=((:antireflecting, :repeating),),
            E=((:reflecting, :repeating),),
        ))
    u1 = similar(u)
    du = similar(u)

    # Initial condition: hot core (r < r_mid) / cool halo, per global radius.
    h = halo_width(u.N)
    for tile_id in 1:tile_count(u)
        dN = tile_parent(u.N, tile_id)
        dM = tile_parent(u.M, tile_id)
        dE = tile_parent(u.E, tile_id)
        for oi in 1:tile_size(u)[1]
            gi = interior_to_global_index(u.N, tile_id, (oi,))[1]
            r  = r_min + (gi - 0.5) * dr
            n, p = r < r_mid ? (1.0, 1.0) : (0.125, 0.1)
            T = p / n
            μ = T * log(n / (A * T^3))
            U = cons_from_prim(eos, T, μ, 0.0)
            dN[oi + h] = U[1]; dM[oi + h] = U[2]; dE[oi + h] = U[3]
        end
    end
    synchronize_halo!(u)

    q0, e0, _ = diagnostics(u, eos, r_min, dr)
    @printf("Relativistic radial blast — cylindrical, THREADED (%d tiles, %d threads)\n",
        tile_count(u), Threads.nthreads())
    @printf("  nx=%d  r∈[%.2f,%.2f]  t_end=%.2f\n", nx, r_min, r_max, t_end)
    @printf("  initial  ∫N r dr=%.6f  ∫E r dr=%.6f\n", q0, e0)

    dt = cfl * dr
    t = 0.0; step = 0
    while t < t_end
        dtn = min(dt, t_end - t)
        ssprk2_step!(u, u1, du, eos, r_min, dr, dtn)
        t += dtn; step += 1
    end

    q1, e1, vmax = diagnostics(u, eos, r_min, dr)
    @printf("  final    ∫N r dr=%.6f  ∫E r dr=%.6f  vmax=%.4f  steps=%d  t=%.4f\n",
        q1, e1, vmax, step, t)
    rel_q = abs(q1 - q0) / q0
    rel_e = abs(e1 - e0) / e0
    @printf("  area-weighted drift (rel): charge=%.2e  energy=%.2e\n", rel_q, rel_e)
    println(rel_q < 5.0e-3 && rel_e < 5.0e-3 && vmax > 0.0 ?
        "  ✓ threaded run consistent: integrals conserved to discretization order, flow moving" :
        "  ✗ unexpected result")

    return u
end

run_cylindrical_blast_threaded()
