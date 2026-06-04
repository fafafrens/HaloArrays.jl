# ============================================================
# 1-D Special Relativistic Hydrodynamics — Sod shock tube
#
#   julia --project=. examples/finite_volume/relativistic_hydro_1d.jl
#
# Solves the Valencia form of the 1-D special relativistic Euler equations:
#
#   ∂_t [D, S, τ]ᵀ + ∂_x [Dv, Sv+p, S−Dv]ᵀ = 0
#
# with c = 1 and the ideal-gas EOS  p = (γ−1)ρε.  The conserved variables are
#
#   D = ρW              rest-mass density
#   S = ρhW²v           momentum
#   τ = ρhW²−p−D        energy (rest mass excluded)
#
# where  W = (1−v²)^{−½}  and  h = 1+ε+p/ρ.
#
# Initial condition: relativistic Sod shock tube
#   left  ρ=1.0, v=0, p=1.0    right  ρ=0.125, v=0, p=0.1
#
# Boundary condition: coupled outflow ghost
#   At each domain edge the primitive variables are recovered from the last
#   interior cell, the velocity is clamped to prevent inflow, and all ghost
#   cells are filled by converting back to (D,S,τ).  Because the primitive
#   recovery couples all three conserved fields, this is implemented as an
#   AbstractCoupledBoundaryCondition so the three fields are always treated
#   together.
#
# Time integration: SSP-RK2 with CFL-adaptive dt.
# ============================================================

using HaloArrays
using Printf

const GAMMA = 5.0 / 3.0

# ─── EOS helpers ──────────────────────────────────────────────────────────────

@inline _eps_from_prim(ρ, p)  = p / ((GAMMA - 1.0) * ρ)
@inline _h_from_prim(ρ, p)    = 1.0 + _eps_from_prim(ρ, p) + p / ρ

# ─── Conserved ← primitive ────────────────────────────────────────────────────

@inline function cons_from_prim(ρ, v, p)
    W  = 1.0 / sqrt(1.0 - v^2)
    h  = _h_from_prim(ρ, p)
    D  = ρ * W
    S  = ρ * h * W^2 * v
    τ  = ρ * h * W^2 - p - D
    D, S, τ
end

# ─── Primitive recovery: Newton–Raphson on pressure ───────────────────────────
#
# Given conserved (D, S, τ), with E = τ+D, we solve
#
#   f(p) = (E+p) − D·W(p) − (γ/(γ−1))·p·W(p)² = 0
#
# where  W(p) = (E+p)/√((E+p)²−S²).  The analytical derivative is
#
#   f′(p) = 1 + D·S²/Z³ − (γ/(γ−1))·W²·(1 − 2pS²/(X·Z²))
#
# with X = E+p, Z = √(X²−S²).  Converges in 3–6 iterations.

function prim_from_cons(D, S, τ; maxit=50, tol=1.0e-12)
    E    = τ + D
    S2   = S^2
    gog1 = GAMMA / (GAMMA - 1.0)

    p = max((GAMMA - 1.0) / GAMMA * (E - D), 1.0e-14)

    for _ in 1:maxit
        X  = E + p
        X2 = X^2
        Z2 = max(X2 - S2, 1.0e-30)
        Z  = sqrt(Z2)
        W  = X / Z
        W2 = W^2

        f  = X - D * W - gog1 * p * W2
        df = 1.0 + D * S2 / Z^3 - gog1 * W2 * (1.0 - 2.0 * p * S2 / (X * Z2))

        dp = -f / df
        p  = max(p + dp, 1.0e-14)
        abs(dp) < tol * (p + 1.0e-14) && break
    end

    X = E + p
    v = S / X
    W = 1.0 / sqrt(max(1.0 - v^2, 1.0e-14))
    ρ = D / W
    return ρ, v, p
end

# ─── Wave speeds (Balsara 1994) ───────────────────────────────────────────────

@inline function wave_speeds(ρ, v, p)
    h   = _h_from_prim(ρ, p)
    cs2 = GAMMA * p / (ρ * h)         # c_s² in the fluid rest frame
    cs  = sqrt(cs2)
    v_m = (v - cs) / (1.0 - v * cs)   # left-going eigenvalue (λ₋)
    v_p = (v + cs) / (1.0 + v * cs)   # right-going eigenvalue (λ₊)
    v_m, v_p
end

# ─── Physical fluxes ──────────────────────────────────────────────────────────

@inline function physical_flux(D, S, _, v, p)
    F_D   = D * v
    F_S   = S * v + p
    F_τ   = S - D * v          # = (τ+p)v − Dv²  in primitive form
    F_D, F_S, F_τ
end

# ─── HLL numerical flux ───────────────────────────────────────────────────────

function hll_flux(D_L, S_L, τ_L, D_R, S_R, τ_R)
    ρ_L, v_L, p_L = prim_from_cons(D_L, S_L, τ_L)
    ρ_R, v_R, p_R = prim_from_cons(D_R, S_R, τ_R)

    s_L = min(wave_speeds(ρ_L, v_L, p_L)[1], wave_speeds(ρ_R, v_R, p_R)[1])
    s_R = max(wave_speeds(ρ_L, v_L, p_L)[2], wave_speeds(ρ_R, v_R, p_R)[2])

    if s_L >= 0.0
        return physical_flux(D_L, S_L, τ_L, v_L, p_L)
    elseif s_R <= 0.0
        return physical_flux(D_R, S_R, τ_R, v_R, p_R)
    end

    fD_L, fS_L, fτ_L = physical_flux(D_L, S_L, τ_L, v_L, p_L)
    fD_R, fS_R, fτ_R = physical_flux(D_R, S_R, τ_R, v_R, p_R)

    c = 1.0 / (s_R - s_L)
    fD = (s_R * fD_L - s_L * fD_R + s_L * s_R * (D_R - D_L)) * c
    fS = (s_R * fS_L - s_L * fS_R + s_L * s_R * (S_R - S_L)) * c
    fτ = (s_R * fτ_L - s_L * fτ_R + s_L * s_R * (τ_R - τ_L)) * c
    fD, fS, fτ
end

# ─── Coupled outflow boundary condition ───────────────────────────────────────
#
# Ghost cells are filled by recovering primitive variables from the last
# interior cell, clamping the velocity (so there is no supersonic inflow),
# and converting back to conserved form.  All three fields are written
# simultaneously because prim_from_cons couples them.

struct RelativisticOutflow <: AbstractCoupledBoundaryCondition end

function HaloArrays.apply_coupled_bc!(::RelativisticOutflow,
        state, ::Side{S}, ::Dim{1}) where {S}
    D_f, S_f, τ_f = eachfield(state)

    D_e = get_send_view(Side(S), Dim(1), D_f)[1]
    S_e = get_send_view(Side(S), Dim(1), S_f)[1]
    τ_e = get_send_view(Side(S), Dim(1), τ_f)[1]

    ρ_e, v_e, p_e = prim_from_cons(D_e, S_e, τ_e)

    # prevent supersonic inflow: clamp velocity to zero at the respective wall
    v_e = S == 1 ? max(v_e, 0.0) : min(v_e, 0.0)

    D_g, S_g, τ_g = cons_from_prim(ρ_e, v_e, p_e)

    fill!(get_recv_view(Side(S), Dim(1), D_f), D_g)
    fill!(get_recv_view(Side(S), Dim(1), S_f), S_g)
    fill!(get_recv_view(Side(S), Dim(1), τ_f), τ_g)
    return nothing
end

# ─── RHS: finite-volume update in (D, S, τ) ───────────────────────────────────

function rhs!(state, bc, dx)
    apply_coupled_bc!(bc, state)

    D_f, S_f, τ_f = eachfield(state)
    pD  = parent(D_f);  pS = parent(S_f);  pτ = parent(τ_f)
    rng = interior_range(D_f)[1]
    invdx = 1.0 / dx

    du = zeros(3, length(rng))   # [field, cell]

    for (k, i) in enumerate(rng)
        fD_r, fS_r, fτ_r = hll_flux(pD[i],   pS[i],   pτ[i],   pD[i+1], pS[i+1], pτ[i+1])
        fD_l, fS_l, fτ_l = hll_flux(pD[i-1], pS[i-1], pτ[i-1], pD[i],   pS[i],   pτ[i])
        du[1, k] = -(fD_r - fD_l) * invdx
        du[2, k] = -(fS_r - fS_l) * invdx
        du[3, k] = -(fτ_r - fτ_l) * invdx
    end

    du
end

# ─── CFL time step ────────────────────────────────────────────────────────────

function cfl_dt(state, dx, cfl)
    D_f, S_f, τ_f = eachfield(state)
    pD = parent(D_f); pS = parent(S_f); pτ = parent(τ_f)
    amax = 0.0
    for i in interior_range(D_f)[1]
        ρ, v, p = prim_from_cons(pD[i], pS[i], pτ[i])
        vm, vp = wave_speeds(ρ, v, p)
        amax   = max(amax, abs(vp), abs(vm))
    end
    cfl * dx / max(amax, 1.0e-14)
end

# ─── SSP-RK2 (Shu–Osher) ─────────────────────────────────────────────────────

function ssprk2_step!(state, state1, bc, dt, dx)
    D_f, S_f, τ_f   = eachfield(state)
    D1_f, S1_f, τ1_f = eachfield(state1)

    # Save u^n in state1
    copyto!(parent(D1_f), parent(D_f))
    copyto!(parent(S1_f), parent(S_f))
    copyto!(parent(τ1_f), parent(τ_f))
    synchronize_halo!(state1)

    # Stage 1:  u¹ = u^n + dt·L(u^n)
    du = rhs!(state, bc, dx)
    rng = interior_range(D_f)[1]
    for (k, i) in enumerate(rng)
        parent(D_f)[i] = parent(D1_f)[i] + dt * du[1, k]
        parent(S_f)[i] = parent(S1_f)[i] + dt * du[2, k]
        parent(τ_f)[i] = parent(τ1_f)[i] + dt * du[3, k]
    end
    synchronize_halo!(state)

    # Stage 2:  u^{n+1} = ½u^n + ½(u¹ + dt·L(u¹))
    du = rhs!(state, bc, dx)
    for (k, i) in enumerate(rng)
        parent(D_f)[i] = 0.5 * parent(D1_f)[i] + 0.5 * (parent(D_f)[i] + dt * du[1, k])
        parent(S_f)[i] = 0.5 * parent(S1_f)[i] + 0.5 * (parent(S_f)[i] + dt * du[2, k])
        parent(τ_f)[i] = 0.5 * parent(τ1_f)[i] + 0.5 * (parent(τ_f)[i] + dt * du[3, k])
    end
    synchronize_halo!(state)

    return state
end

# ─── Diagnostics ──────────────────────────────────────────────────────────────

function diagnostics(state, dx)
    D_f, S_f, τ_f = eachfield(state)
    pD = parent(D_f); pS = parent(S_f); pτ = parent(τ_f)
    rng = interior_range(D_f)[1]

    mass   = 0.0
    energy = 0.0
    vmax   = 0.0
    for i in rng
        _, v, _ = prim_from_cons(pD[i], pS[i], pτ[i])
        mass   += pD[i] * dx
        energy += (pτ[i] + pD[i]) * dx
        vmax    = max(vmax, abs(v))
    end
    mass, energy, vmax
end

# ─── Driver ───────────────────────────────────────────────────────────────────

function run_relativistic_sod(; nx=400, cfl=0.5, t_end=0.4)
    dx = 1.0 / nx
    bc = RelativisticOutflow()

    # state = (D, S, τ) as an ArrayOfHaloArray; x-boundaries use coupled outflow
    state  = ArrayOfHaloArray(LocalHaloArray, Float64, (3,), (nx,), 1;
        boundary_condition=((:noboundary, :noboundary),))
    state1 = ArrayOfHaloArray(LocalHaloArray, Float64, (3,), (nx,), 1;
        boundary_condition=((:noboundary, :noboundary),))

    D_f, S_f, τ_f = eachfield(state)

    # Relativistic Sod initial condition (discontinuity at x = 0.5)
    for i in 1:nx
        x = (i - 0.5) * dx
        ρ = x < 0.5 ? 1.0   : 0.125
        p = x < 0.5 ? 1.0   : 0.1
        D_f[i], S_f[i], τ_f[i] = cons_from_prim(ρ, 0.0, p)
    end
    synchronize_halo!(state)

    mass0, energy0, _ = diagnostics(state, dx)
    @printf("Relativistic Sod shock tube  nx=%d  t_end=%.2f\n", nx, t_end)
    @printf("  initial  mass=%.6f  energy=%.6f\n", mass0, energy0)

    t   = 0.0
    step = 0
    while t < t_end
        dt  = min(cfl_dt(state, dx, cfl), t_end - t)
        ssprk2_step!(state, state1, bc, dt, dx)
        t    += dt
        step += 1
    end

    mass1, energy1, vmax = diagnostics(state, dx)
    @printf("  final    mass=%.6f  energy=%.6f  vmax=%.4f  steps=%d  t=%.4f\n",
        mass1, energy1, vmax, step, t)

    # Qualitative checks:
    #   1. Bulk velocity appeared (rarefaction + shock created motion)
    #   2. Density at x≈0.75 (in the originally right half) rose above 0.125
    #      because the right-going shock compressed that region
    pD_p = parent(D_f); pS_p = parent(S_f); pτ_p = parent(τ_f)
    h = halo_width(D_f)
    i_probe = h + round(Int, 0.75 * nx)    # storage index for x ≈ 0.75
    ρ_probe, _, _ = prim_from_cons(pD_p[i_probe], pS_p[i_probe], pτ_p[i_probe])

    @printf("  ρ at x≈0.75 = %.4f  (initial right = 0.125, shock raises it)\n", ρ_probe)
    @printf("  vmax        = %.4f  (> 0 confirms bulk motion)\n", vmax)
    println(ρ_probe > 0.125 && vmax > 0.0 ?
        "  ✓ shock formed: right side compressed and flow is moving" :
        "  ✗ unexpected result")

    return state
end

run_relativistic_sod()
