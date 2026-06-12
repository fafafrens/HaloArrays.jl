# ============================================================
# 1-D Special Relativistic Hydrodynamics — cylindrical radial, geometric source
#
#   julia --project=. examples/finite_volume/relativistic_hydro_cylindrical_1d.jl
#
# Axisymmetric radial flow of the charge-carrying ultrarelativistic gas
# (conserved (N, M, E), 2-D primitive recovery, as in relativistic_hydro_Tmu_1d).
# The cylindrical divergence (1/r)∂_r(r F) is split into a planar flux term plus
# a GEOMETRIC SOURCE that this example fills with a per-cell loop — the
# coupled+positional source pattern.
#
# Derivation. With area ∝ r^α (α = 1 cylindrical, 2 spherical, 0 planar),
#
#   ∂_t U + (1/r^α) ∂_r(r^α F) = (azimuthal pressure stress),
#   (1/r^α) ∂_r(r^α F) = ∂_r F + (α/r) F,
#
# so  ∂_t U + ∂_r F = −(α/r) F + (pressure stress). In the radial momentum
# equation the transverse pressure stress +α p/r cancels the pressure piece of
# −(α/r)(Mv + p), leaving only −(α/r) Mv. Hence
#
#   S = −(α/r) · (N v,  M v,  M).
#
# Note S vanishes while the fluid is at rest (M = 0, v = 0) and switches on as
# flow develops — a radial blast then dilutes faster than its planar twin.
#
# Axis boundary (inner, r = 0). By symmetry the scalars N, E are EVEN across the
# axis (reflecting), while the radial momentum M is ODD and must vanish there
# (antireflecting): M ~ r near the axis, which keeps the 1/r source regular
# (S_N ~ N, S_M ~ r, S_E ~ const all stay bounded). The outer boundary is
# zeroth-order outflow (:repeating).
#
# The CONSERVED integral is area-weighted: ∂_t ∫ U r^α dr = −[r^α F] (boundary
# flux). This operator-split discretization (planar flux + pointwise source)
# conserves ∫ N r dr and ∫ E r dr only to truncation order — the residual drift
# (~0.1 % here) is the splitting error. A *well-balanced* scheme that uses
# r-weighted face areas A_{i±1/2}=r_{i±1/2} in the flux would conserve them to
# machine precision, but then the geometry lives in the areas, not in a source.
#
# Self-contained: HaloArrays, Printf, StaticArrays.
# ============================================================

using HaloArrays
using Printf
using StaticArrays

const GEOMETRY = 1.0     # α: 1 = cylindrical, 2 = spherical, 0 = planar

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

# ─── Fluxes (planar part: F_N = Nv, F_M = Mv+p, F_E = M) ───────────────────────

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

# ─── Field access ─────────────────────────────────────────────────────────────

@inline conserved_cell(d, I) = SVector(d.N[I], d.M[I], d.E[I])

@inline function add_conserved!(d, I, scale, U)
    d.N[I] += scale * U[1]
    d.M[I] += scale * U[2]
    d.E[I] += scale * U[3]
    return d
end

# ─── Geometric source (coupled + positional, cell loop) ───────────────────────
#
# du[I] = S(U_I, r_I)  with  S = −(α/r)(Nv, Mv, M).  This SETS du — it is the
# first contribution to the RHS, so it replaces the fill!(du,0) + accumulate
# pair; the flux divergence is then added on top.  Iterates owned cells via
# get_interior_cells(CellRanges(u)) — the cell analogue of the face loop.  Because
# it reads ONLY owned cells (never ghosts), it can run while the halo exchange
# is still in flight (see rel_rhs!).

function add_geometric_source!(du, u, eos, r_min, dr)
    du_data = parent(du)
    u_data  = parent(u)
    h = halo_width(u.N)
    @inbounds for I in get_interior_cells(CellRanges(u))
        U = conserved_cell(u_data, I)
        _, _, v = prim_from_cons(eos, U)
        r = r_min + (I[1] - h - 0.5) * dr
        S = SVector(U[1] * v, U[2] * v, U[2]) * (-GEOMETRY / r)
        du_data.N[I] = S[1]
        du_data.M[I] = S[2]
        du_data.E[I] = S[3]
    end
    return du
end

# ─── RHS and SSP-RK2 ──────────────────────────────────────────────────────────

function rel_rhs!(du, u, eos, r_min, dr)
    # Overlap communication with computation: post the halo exchange, then run
    # the purely local geometric source (it reads only owned cells) while the
    # exchange is in flight, then finish the exchange and apply the physical
    # boundary condition before the flux, which needs the ghosts. start_/finish_
    # are no-ops for the serial LocalHaloArray but make this MPI-ready.
    start_halo_exchange!(u)                          # post exchange
    add_geometric_source!(du, u, eos, r_min, dr)     # du = S(U, r)  — overlaps comm
    finish_halo_exchange!(u)                         # wait for exchange
    boundary_condition!(u)                           # axis / outflow physical edges
    accumulate_flux_divergence!(parent(du), parent(u), FaceRanges(u), 1, inv(dr),
        (UL, UR) -> rusanov_flux(eos, UL, UR), conserved_cell, add_conserved!)  # du += -∂_r F
    return du
end

function ssprk2_step!(u, u1, du, eos, r_min, dr, dt)
    rel_rhs!(du, u, eos, r_min, dr)
    @. u1 = u + dt * du
    rel_rhs!(du, u1, eos, r_min, dr)
    @. u = 0.5 * u + 0.5 * (u1 + dt * du)
    return u
end

# ─── Diagnostics (area-weighted, the cylindrical-conserved integrals) ─────────

function diagnostics(u, eos, r_min, dr)
    d = parent(u)
    h = halo_width(u.N)
    charge = 0.0; energy = 0.0; vmax = 0.0
    @inbounds for I in get_interior_cells(CellRanges(u))
        U = conserved_cell(d, I)
        _, _, v = prim_from_cons(eos, U)
        r = r_min + (I[1] - h - 0.5) * dr
        charge += U[1] * r * dr        # ∫ N r dr
        energy += U[3] * r * dr        # ∫ E r dr
        vmax    = max(vmax, abs(v))
    end
    return charge, energy, vmax
end

# ─── Driver: radial relativistic blast ────────────────────────────────────────

function run_cylindrical_blast(; A=1.0, nx=400, cfl=0.4, r_min=0.0, r_max=1.0,
        r_mid=0.3, t_end=0.40)
    eos = UltraRelGas(A)
    dr  = (r_max - r_min) / nx

    # Axis (inner/side-1): N, E even → reflecting; M odd → antireflecting (M→0).
    # Outer (side-2): zeroth-order outflow.
    u  = LocalMultiHaloArray(Float64, (nx,), 1; boundary_conditions=(
        N=((:reflecting, :repeating),),
        M=((:antireflecting, :repeating),),
        E=((:reflecting, :repeating),),
    ))
    u1 = similar(u)
    du = similar(u)

    function set_state!(i, n, p)
        T = p / n
        μ = T * log(n / (A * T^3))
        U = cons_from_prim(eos, T, μ, 0.0)
        interior_view(u.N)[i] = U[1]
        interior_view(u.M)[i] = U[2]
        interior_view(u.E)[i] = U[3]
    end
    for i in 1:nx
        r = r_min + (i - 0.5) * dr
        r < r_mid ? set_state!(i, 1.0, 1.0) : set_state!(i, 0.125, 0.1)   # hot core / cool halo
    end
    synchronize_halo!(u)

    q0, e0, _ = diagnostics(u, eos, r_min, dr)
    @printf("Relativistic radial blast — cylindrical (α=%.0f), geometric source\n", GEOMETRY)
    @printf("  nx=%d  r∈[%.2f,%.2f]  t_end=%.2f\n", nx, r_min, r_max, t_end)
    @printf("  initial  ∫N r dr=%.6f  ∫E r dr=%.6f\n", q0, e0)

    # |λ| ≤ 1 ⇒ dt = cfl·dr is CFL-stable.
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
    @printf("  area-weighted drift (rel): charge=%.2e  energy=%.2e  (splitting truncation error)\n",
        rel_q, rel_e)
    println(rel_q < 5.0e-3 && rel_e < 5.0e-3 && vmax > 0.0 ?
        "  ✓ geometric source consistent: area-weighted integrals conserved to discretization order, flow moving" :
        "  ✗ unexpected result")

    return u
end

run_cylindrical_blast()
