# ============================================================
# 1-D Special Relativistic Hydrodynamics — (T, μ) primitives, DiffEq stepping
#
#   julia --project=. examples/finite_volume/relativistic_hydro_Tmu_diffeq_1d.jl
#
# Same physics as relativistic_hydro_Tmu_1d.jl (charge-carrying ultrarelativistic
# gas, conserved (N, M, E), 2-D primitive recovery), but the hand-written SSP-RK2
# loop is replaced by OrdinaryDiffEq: the spatial scheme is exposed as an ODE RHS
# and the time integrator does the stepping.
#
# The conservative face update already computes  du = −∂ₓF = ∂ₜU,  so the RHS is
# exactly what ODEProblem expects. Relativistic signal speeds are bounded by
# c = 1, so dt = cfl·dx is unconditionally CFL-stable; we use a fixed-step
# explicit RK (Tsit5, adaptive=false) — the same pattern as the other DiffEq
# examples here.
#
# Self-contained: HaloArrays, OrdinaryDiffEq, Printf, StaticArrays.
# ============================================================

using HaloArrays
using DiffEqBase: ODEProblem
using OrdinaryDiffEq
using Printf
using StaticArrays

# ─── Equation of state: ultrarelativistic classical ideal gas ─────────────────

struct UltraRelGas
    A::Float64        # degeneracy g/π²
end

@inline _exp_muT(μ, T) = exp(clamp(μ / T, -700.0, 700.0))

@inline pressure(eos::UltraRelGas, T, μ)         = eos.A * T^4 * _exp_muT(μ, T)
@inline charge_density(eos::UltraRelGas, T, μ)   = eos.A * T^3 * _exp_muT(μ, T)   # = p/T
@inline enthalpy_density(eos::UltraRelGas, T, μ) = 4.0 * pressure(eos, T, μ)      # w = 4p
@inline sound_speed(::UltraRelGas)               = 1.0 / sqrt(3.0)                 # c_s² = 1/3

# ─── Primitive → conserved ────────────────────────────────────────────────────
#   N = nW,  M = T^0i = wW²v,  E = T^00 = wW² − p

@inline function cons_from_prim(eos, T, μ, v)
    W = 1.0 / sqrt(1.0 - v^2)
    p = pressure(eos, T, μ)
    n = charge_density(eos, T, μ)
    w = enthalpy_density(eos, T, μ)
    return SVector(n * W, w * W^2 * v, w * W^2 - p)
end

# ─── Conserved → primitive: 2-D Newton on (T, μ) ──────────────────────────────

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
#   F_N = N v,  F_M = M v + p,  F_E = M

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

# ─── ODE right-hand side ──────────────────────────────────────────────────────
#
# du = ∂ₜU = −∂ₓF.  accumulate_flux_divergence! builds exactly that (it already
# carries the minus sign of the conservative scatter), so this is a drop-in
# ODEProblem RHS.  params = (eos, dx); t is unused (autonomous system).

function rhs!(du, u, params, t)
    eos, dx = params
    fill!(du, 0.0)
    synchronize_halo!(u)              # :repeating fills the outflow ghosts
    accumulate_flux_divergence!(parent(du), parent(u), FaceRanges(u), 1, inv(dx),
        (UL, UR) -> rusanov_flux(eos, UL, UR), conserved_cell, add_conserved!)
    return nothing
end

# ─── Diagnostics ──────────────────────────────────────────────────────────────

function diagnostics(u, eos, dx)
    d = parent(u)
    charge = 0.0; energy = 0.0; vmax = 0.0
    for I in CartesianIndices(interior_range(u.N))
        U = conserved_cell(d, I)
        _, _, v = prim_from_cons(eos, U)
        charge += U[1] * dx
        energy += U[3] * dx
        vmax    = max(vmax, abs(v))
    end
    return charge, energy, vmax
end

# ─── Driver: DiffEq time integration of the charge-based Sod tube ─────────────

function run_Tmu_diffeq(; A=1.0, nx=400, cfl=0.4, t_end=0.4)
    eos = UltraRelGas(A)
    dx  = 1.0 / nx

    u0 = LocalMultiHaloArray(Float64, (nx,), 1;
        fields=(:N, :M, :E), boundary_condition=:repeating)

    function set_state!(i, n, p)
        T = p / n
        μ = T * log(n / (A * T^3))
        U = cons_from_prim(eos, T, μ, 0.0)
        interior_view(u0.N)[i] = U[1]
        interior_view(u0.M)[i] = U[2]
        interior_view(u0.E)[i] = U[3]
    end
    for i in 1:nx
        x = (i - 0.5) * dx
        x < 0.5 ? set_state!(i, 1.0, 1.0) : set_state!(i, 0.125, 0.1)
    end
    synchronize_halo!(u0)

    q0, e0, _ = diagnostics(u0, eos, dx)
    @printf("Relativistic Sod — (T, μ), DiffEq stepping (Tsit5, fixed dt)\n")
    @printf("  nx=%d  t_end=%.2f  initial charge=%.6f energy=%.6f\n", nx, t_end, q0, e0)

    # |λ| ≤ c = 1  ⇒  dt = cfl·dx is always CFL-stable.
    dt   = cfl * dx
    prob = ODEProblem{true}(rhs!, u0, (0.0, t_end), (eos, dx))
    sol  = solve(prob, Tsit5(); dt=dt, adaptive=false, save_everystep=false)

    u = sol.u[end]
    synchronize_halo!(u)

    q1, e1, vmax = diagnostics(u, eos, dx)
    @printf("  final    charge=%.6f  energy=%.6f  vmax=%.4f  diffeq_steps=%d  t=%.4f\n",
        q1, e1, vmax, sol.stats.naccept, sol.t[end])

    d = parent(u)
    h = halo_width(u.N)
    i_probe = h + round(Int, 0.75 * nx)
    T_p, μ_p, _ = prim_from_cons(eos, conserved_cell(d, CartesianIndex(i_probe)))
    n_probe = charge_density(eos, T_p, μ_p)
    @printf("  n at x≈0.75 = %.4f  (initial right = 0.125, shock raises it)\n", n_probe)
    println(n_probe > 0.125 && vmax > 0.0 ?
        "  ✓ shock formed: charge compressed and flow is moving" :
        "  ✗ unexpected result")

    return u
end

run_Tmu_diffeq()
