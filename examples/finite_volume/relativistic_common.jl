# ============================================================
# Shared 1-D special relativistic hydrodynamics kernels
#
# Valencia form of the 1-D special relativistic Euler equations (c = 1):
#
#   ∂_t U + ∂_x F(U) = 0,   U = (D, S, τ),  F = (Dv, Sv+p, S−Dv)
#
# with ideal-gas EOS  p = (γ−1)ρε  and conserved variables
#
#   D = ρW              W = (1−v²)^{−½}
#   S = ρhW²v           h = 1 + ε + p/ρ
#   τ = ρhW²−p−D
#
# Numerics: Rusanov flux, FaceRanges FV update, SSP-RK2 time stepping.
# The boundary-condition strategy is supplied by the caller, so this file
# is shared by:
#   * relativistic_hydro_1d.jl            — coupled characteristic outflow
#   * relativistic_hydro_repeating_1d.jl  — zeroth-order (:repeating) outflow
# ============================================================

using HaloArrays
using Printf
using StaticArrays

# ─── Equation of state ────────────────────────────────────────────────────────
#
# The relativistic inversion below is pure geometry: given a trial pressure it
# recovers W, v, ρ.  The *thermodynamics* — how pressure relates to the specific
# enthalpy and the sound speed — lives entirely in the EOS.  To add an EOS,
# subtype AbstractEOS and implement these three methods; nothing else changes.

abstract type AbstractEOS end

"""    specific_enthalpy(eos, ρ, p) -> h = 1 + ε + p/ρ"""
function specific_enthalpy end

"""    sound_speed(eos, ρ, p) -> c_s"""
function sound_speed end

"""    pressure_guess(eos, U) -> initial pressure for the Newton solve"""
function pressure_guess end

# Ideal gas:  p = (γ−1)ρε  ⇒  h = 1 + γ/(γ−1)·p/ρ,  c_s² = γp/(ρh).
struct IdealGas <: AbstractEOS
    gamma::Float64
end

@inline specific_enthalpy(eos::IdealGas, ρ, p) = 1.0 + eos.gamma / (eos.gamma - 1.0) * p / ρ

@inline function sound_speed(eos::IdealGas, ρ, p)
    h = specific_enthalpy(eos, ρ, p)
    return sqrt(eos.gamma * p / (ρ * h))
end

# (γ−1)/γ · τ  — exact in the cold/static limit (W≈1), a fine Newton seed.
@inline pressure_guess(eos::IdealGas, U) =
    max((eos.gamma - 1.0) / eos.gamma * U[3], 1.0e-10)

# ─── Primitive ↔ conserved conversion (all SVector-based) ─────────────────────

@inline function cons_from_prim(eos, ρ, v, p)
    W = 1.0 / sqrt(1.0 - v^2)
    h = specific_enthalpy(eos, ρ, p)
    D = ρ * W
    S = ρ * h * W^2 * v
    τ = ρ * h * W^2 - p - D
    return SVector(D, S, τ)
end

# Primitive recovery: Newton on pressure.  With E = τ+D, X = E+p, Z = √(X²−S²),
# the trial p fixes the geometry (W = X/Z, v = S/X, ρ = D/W); the EOS-agnostic
# consistency condition ρhW² = E+p becomes
#
#   f(p) = X − D·W·h(ρ, p) = 0
#
# which needs only `specific_enthalpy` from the EOS.  A forward-difference
# derivative keeps the solver independent of the EOS (a production code would
# supply the analytic Jacobian).
function prim_from_cons(eos, U; maxit=100, tol=1.0e-12)
    D, S, τ = U
    E  = τ + D
    S2 = S^2

    residual(p) = let X = E + p, Z = sqrt(max((E + p)^2 - S2, 1.0e-30))
        W = X / Z
        X - D * W * specific_enthalpy(eos, D / W, p)
    end

    p = pressure_guess(eos, U)
    for _ in 1:maxit
        f0 = residual(p)
        δ  = max(1.0e-7 * abs(p), 1.0e-10)
        df = (residual(p + δ) - f0) / δ
        dp = -f0 / df
        p  = max(p + dp, 1.0e-14)
        abs(dp) < tol * (p + 1.0e-14) && break
    end

    X = E + p
    v = S / X
    W = 1.0 / sqrt(max(1.0 - v^2, 1.0e-14))
    ρ = D / W
    return ρ, v, p
end

@inline function physical_flux(eos, U)
    _, v, p = prim_from_cons(eos, U)
    return SVector(U[1] * v, U[2] * v + p, U[2] - U[1] * v)
end

# Largest |eigenvalue| (Balsara 1994): λ± = (v ± c_s)/(1 ± v·c_s)
@inline function max_wave_speed(eos, U)
    ρ, v, p = prim_from_cons(eos, U)
    cs = sound_speed(eos, ρ, p)
    return max(abs((v - cs) / (1.0 - v * cs)), abs((v + cs) / (1.0 + v * cs)))
end

@inline function rusanov_flux(eos, UL, UR)
    smax = max(max_wave_speed(eos, UL), max_wave_speed(eos, UR))
    return 0.5 * (physical_flux(eos, UL) + physical_flux(eos, UR)) - 0.5 * smax * (UR - UL)
end

# ─── Field access on the NamedTuple returned by parent(state) ─────────────────

@inline conserved_cell(d, I) = SVector(d.D[I], d.S[I], d.tau[I])

@inline function add_conserved!(d, I, scale, U)
    d.D[I]   += scale * U[1]
    d.S[I]   += scale * U[2]
    d.tau[I] += scale * U[3]
    return d
end

# ─── RHS and SSP-RK2 time stepping ────────────────────────────────────────────
#
# The whole left/internal/right conservative face update is one library call:
# `accumulate_flux_divergence!` evaluates `rusanov_flux` per face and scatters
# it onto the owned cells, reading/writing the (D,S,τ) state via `conserved_cell`
# and `add_conserved!`.
#
# `apply_bc!` is the caller's boundary strategy.  For a per-field BC
# (e.g. :repeating) `synchronize_halo!` already fills the ghosts, so pass a
# no-op.  For a coupled BC mark the edges :noboundary and pass a closure that
# fills them (see relativistic_hydro_1d.jl).

function rel_rhs!(du, u, eos, apply_bc!, dx)
    fill!(du, 0.0)
    synchronize_halo!(u)
    apply_bc!(u)
    accumulate_flux_divergence!(parent(du), parent(u), FaceRanges(u), 1, inv(dx),
        (UL, UR) -> rusanov_flux(eos, UL, UR), conserved_cell, add_conserved!)
    return du
end

function ssprk2_step!(u, u1, du, eos, apply_bc!, dt, dx)
    rel_rhs!(du, u, eos, apply_bc!, dx)
    @. u1 = u + dt * du
    rel_rhs!(du, u1, eos, apply_bc!, dx)
    @. u = 0.5 * u + 0.5 * (u1 + dt * du)
    return u
end

# ─── Diagnostics ──────────────────────────────────────────────────────────────

function cfl_dt(u, eos, dx, cfl)
    d = parent(u)
    amax = 0.0
    for I in CartesianIndices(interior_range(u.D))
        amax = max(amax, max_wave_speed(eos, conserved_cell(d, I)))
    end
    return cfl * dx / max(amax, 1.0e-14)
end

function diagnostics(u, eos, dx)
    d = parent(u)
    mass = 0.0; energy = 0.0; vmax = 0.0
    for I in CartesianIndices(interior_range(u.D))
        U = conserved_cell(d, I)
        _, v, _ = prim_from_cons(eos, U)
        mass   += U[1] * dx
        energy += (U[3] + U[1]) * dx
        vmax    = max(vmax, abs(v))
    end
    return mass, energy, vmax
end

# ─── Shared driver: relativistic Sod shock tube ───────────────────────────────
#
# `make_state(nx)` builds the LocalMultiHaloArray with the chosen boundary
# condition; `apply_bc!(u)` is the per-step boundary strategy (see above);
# `eos` is any AbstractEOS (defaults to a γ=5/3 ideal gas).

function run_relativistic_sod(make_state, apply_bc!;
        label, eos::AbstractEOS=IdealGas(5.0 / 3.0), nx=400, cfl=0.4, t_end=0.4)
    dx = 1.0 / nx
    u  = make_state(nx)
    u1 = similar(u)
    du = similar(u)

    # Relativistic Sod: left (ρ=1, p=1) / right (ρ=0.125, p=0.1), v=0
    for i in 1:nx
        x = (i - 0.5) * dx
        ρ = x < 0.5 ? 1.0 : 0.125
        p = x < 0.5 ? 1.0 : 0.1
        U = cons_from_prim(eos, ρ, 0.0, p)
        interior_view(u.D)[i]   = U[1]
        interior_view(u.S)[i]   = U[2]
        interior_view(u.tau)[i] = U[3]
    end
    synchronize_halo!(u)

    mass0, energy0, _ = diagnostics(u, eos, dx)
    @printf("Relativistic Sod shock tube — %s\n", label)
    @printf("  nx=%d  t_end=%.2f  initial mass=%.6f energy=%.6f\n",
        nx, t_end, mass0, energy0)

    t = 0.0; step = 0
    while t < t_end
        dt = min(cfl_dt(u, eos, dx, cfl), t_end - t)
        ssprk2_step!(u, u1, du, eos, apply_bc!, dt, dx)
        t += dt; step += 1
    end

    mass1, energy1, vmax = diagnostics(u, eos, dx)
    @printf("  final    mass=%.6f  energy=%.6f  vmax=%.4f  steps=%d  t=%.4f\n",
        mass1, energy1, vmax, step, t)

    # The right-going shock compresses the originally low-density right state.
    d = parent(u)
    h = halo_width(u.D)
    i_probe = h + round(Int, 0.75 * nx)
    ρ_probe, _, _ = prim_from_cons(eos, conserved_cell(d, CartesianIndex(i_probe)))
    @printf("  ρ at x≈0.75 = %.4f  (initial right = 0.125, shock raises it)\n", ρ_probe)
    println(ρ_probe > 0.125 && vmax > 0.0 ?
        "  ✓ shock formed: right side compressed and flow is moving" :
        "  ✗ unexpected result")

    return u
end
