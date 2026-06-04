# ============================================================
# 1-D Special Relativistic Hydrodynamics — (T, μ) primitive variables
#
#   julia --project=. examples/finite_volume/relativistic_hydro_Tmu_1d.jl
#
# A finite-temperature reformulation: instead of (ρ, v, p) with a conserved
# rest-mass density, the fluid carries a conserved CHARGE and the primitive
# thermodynamic variables are the temperature T and chemical potential μ. The
# equation of state is given as p(T, μ); everything else follows from standard
# grand-canonical relations:
#
#   n = ∂p/∂μ|_T      (charge density)
#   s = ∂p/∂T|_μ      (entropy density)
#   e = T s + μ n − p (energy density,  Euler / Gibbs–Duhem)
#   w = e + p = T s + μ n   (enthalpy density)
#
# Conserved variables — the raw stress-energy components plus the charge:
#
#   N = n W            (charge density)        W = (1−v²)^{−½}
#   M = T^0i = w W² v  (momentum density)
#   E = T^00 = w W² − p (energy density)
#
# We evolve E = T^00 directly rather than the Valencia-style τ = T^00 − N. The
# subtraction exists in mass-based schemes to avoid cancelling against the rest
# mass ρc²; here the gas is ultrarelativistic (no rest mass dominates), so the
# raw components are cleaner.
#
# EOS — ultrarelativistic classical (Maxwell–Boltzmann) ideal gas, massless:
#
#   p(T,μ) = A T⁴ e^{μ/T},  A = g/π²,   n = p/T,  e = 3p,  w = 4p,  c_s² = 1/3
#
# This is analytic, so the conserved→primitive inversion is fully testable, and
# c_s² = 1/3 gives a clean relativistic Sod shock.
#
# Why the inversion is 2-D: the single conserved charge N = nW does not pin a
# primitive (n depends on BOTH T and μ), so recovery solves a 2×2 Newton system
# in (T, μ); v then follows from momentum. Contrast the mass-based scheme, where
# D = ρW fixes ρ directly and the root-find is 1-D in p.
# ============================================================

using HaloArrays
using Printf
using StaticArrays

# ─── Equation of state: ultrarelativistic classical ideal gas ─────────────────

struct UltraRelGas
    A::Float64        # degeneracy g/π²
end

@inline _exp_muT(μ, T) = exp(clamp(μ / T, -700.0, 700.0))

@inline pressure(eos::UltraRelGas, T, μ)         = eos.A * T^4 * _exp_muT(μ, T)
@inline charge_density(eos::UltraRelGas, T, μ)   = eos.A * T^3 * _exp_muT(μ, T)   # = p/T
@inline enthalpy_density(eos::UltraRelGas, T, μ) = 4.0 * pressure(eos, T, μ)      # w = e+p = 4p
@inline sound_speed(::UltraRelGas)               = 1.0 / sqrt(3.0)                 # c_s² = 1/3

# ─── Primitive → conserved ────────────────────────────────────────────────────

@inline function cons_from_prim(eos, T, μ, v)
    W = 1.0 / sqrt(1.0 - v^2)
    p = pressure(eos, T, μ)
    n = charge_density(eos, T, μ)
    w = enthalpy_density(eos, T, μ)
    N = n * W
    M = w * W^2 * v
    E = w * W^2 - p
    return SVector(N, M, E)
end

# ─── Conserved → primitive: 2-D Newton on (T, μ) ──────────────────────────────
#
# Given U = (N, M, E). For a trial (T, μ): p, n, w from the EOS,
# X = E + p, Z = √(X²−M²), W = X/Z, v = M/X. Two residuals enforce the charge
# and enthalpy identities:
#
#   R₁ = n·W − N        R₂ = w·W² − X      (X = wW² when consistent)
#
# A forward-difference 2×2 Jacobian keeps the solver EOS-agnostic.

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

    # W ≈ 1 seed: E ≈ 3p = 3nT and N ≈ n  ⇒  T ≈ E/(3N), then invert n for μ.
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
# For U = (N, M, E):  F_N = N v,  F_M = T^xx = M v + p,  F_E = T^0x = M.

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

# ─── Field access (conserved fields N, M, E) ──────────────────────────────────

@inline conserved_cell(d, I) = SVector(d.N[I], d.M[I], d.E[I])

@inline function add_conserved!(d, I, scale, U)
    d.N[I] += scale * U[1]
    d.M[I] += scale * U[2]
    d.E[I] += scale * U[3]
    return d
end

# ─── RHS and SSP-RK2 ──────────────────────────────────────────────────────────

function rel_rhs!(du, u, eos, dx)
    fill!(du, 0.0)
    synchronize_halo!(u)              # :repeating fills the outflow ghosts
    accumulate_flux_divergence!(parent(du), parent(u), FaceRanges(u), 1, inv(dx),
        (UL, UR) -> rusanov_flux(eos, UL, UR), conserved_cell, add_conserved!)
    return du
end

function ssprk2_step!(u, u1, du, eos, dt, dx)
    rel_rhs!(du, u, eos, dx)
    @. u1 = u + dt * du
    rel_rhs!(du, u1, eos, dx)
    @. u = 0.5 * u + 0.5 * (u1 + dt * du)
    return u
end

# ─── Diagnostics ──────────────────────────────────────────────────────────────

function cfl_dt(u, eos, dx, cfl)
    d = parent(u)
    amax = 0.0
    for I in CartesianIndices(interior_range(u.N))
        amax = max(amax, max_wave_speed(eos, conserved_cell(d, I)))
    end
    return cfl * dx / max(amax, 1.0e-14)
end

function diagnostics(u, eos, dx)
    d = parent(u)
    charge = 0.0; energy = 0.0; vmax = 0.0
    for I in CartesianIndices(interior_range(u.N))
        U = conserved_cell(d, I)
        _, _, v = prim_from_cons(eos, U)
        charge += U[1] * dx              # ∫ N dx   (conserved charge)
        energy += U[3] * dx              # ∫ E dx = ∫ T^00 dx
        vmax    = max(vmax, abs(v))
    end
    return charge, energy, vmax
end

# ─── Driver: charge-based relativistic Sod tube ───────────────────────────────

function run_Tmu_sod(; A=1.0, nx=400, cfl=0.4, t_end=0.4)
    eos = UltraRelGas(A)
    dx  = 1.0 / nx

    u  = LocalMultiHaloArray(Float64, (nx,), 1;
        fields=(:N, :M, :E), boundary_condition=:repeating)
    u1 = similar(u)
    du = similar(u)

    # Sod-like states given as (n, p); T = p/n, μ = T·ln(n/(A T³)), v = 0.
    function set_state!(i, n, p)
        T = p / n
        μ = T * log(n / (A * T^3))
        U = cons_from_prim(eos, T, μ, 0.0)
        interior_view(u.N)[i] = U[1]
        interior_view(u.M)[i] = U[2]
        interior_view(u.E)[i] = U[3]
    end
    for i in 1:nx
        x = (i - 0.5) * dx
        x < 0.5 ? set_state!(i, 1.0, 1.0) : set_state!(i, 0.125, 0.1)
    end
    synchronize_halo!(u)

    q0, e0, _ = diagnostics(u, eos, dx)
    @printf("Relativistic Sod — (T, μ) primitives, ultrarelativistic gas (c_s²=1/3)\n")
    @printf("  nx=%d  t_end=%.2f  initial charge=%.6f energy=%.6f\n", nx, t_end, q0, e0)

    t = 0.0; step = 0
    while t < t_end
        dt = min(cfl_dt(u, eos, dx, cfl), t_end - t)
        ssprk2_step!(u, u1, du, eos, dt, dx)
        t += dt; step += 1
    end

    q1, e1, vmax = diagnostics(u, eos, dx)
    @printf("  final    charge=%.6f  energy=%.6f  vmax=%.4f  steps=%d  t=%.4f\n",
        q1, e1, vmax, step, t)

    # round-trip + shock checks at x ≈ 0.75 (originally the low-density right state)
    d = parent(u)
    h = halo_width(u.N)
    i_probe = h + round(Int, 0.75 * nx)
    T_p, μ_p, _ = prim_from_cons(eos, conserved_cell(d, CartesianIndex(i_probe)))
    n_probe = charge_density(eos, T_p, μ_p)
    @printf("  n at x≈0.75 = %.4f  (initial right = 0.125, shock raises it)\n", n_probe)
    @printf("  vmax        = %.4f  (> 0 confirms bulk motion)\n", vmax)
    println(n_probe > 0.125 && vmax > 0.0 ?
        "  ✓ shock formed: charge compressed and flow is moving" :
        "  ✗ unexpected result")

    return u
end

run_Tmu_sod()
