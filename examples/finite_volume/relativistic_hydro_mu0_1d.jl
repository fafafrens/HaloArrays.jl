# ============================================================
# 1-D Special Relativistic Hydrodynamics — μ = 0 (charge-free, conformal)
#
#   julia --project=. examples/finite_volume/relativistic_hydro_mu0_1d.jl
#
# The high-temperature / zero-net-charge limit relevant to top-energy
# heavy-ion collisions (μ_B ≈ 0 at RHIC/LHC): there is NO conserved charge, so
# the only thermodynamic variable is the temperature T. The EOS is the
# conformal (Stefan–Boltzmann) ideal gas
#
#   p(T) = a T⁴,   e = 3p,   w = e + p = 4p,   c_s² = 1/3.
#
# Conserved variables are the raw stress-energy components (momentum, energy);
# with no charge there is nothing to subtract, so E is simply T^00:
#
#   M = T^0i = w W² v,    E = T^00 = w W² − p          W = (1−v²)^{−½}
#
# and the primitives are (T, v).
#
# Conserved → primitive is CLOSED FORM here. With X = E + p = wW²:
# eliminating v and W from M = X v, W² = X²/(X²−M²) gives a quadratic in X,
#
#   3X² − 4E X + M² = 0   ⇒   X = (2E + √(4E²−3M²)) / 3,
#
# then p = X − E,  v = M/X,  T = (p/a)^{1/4}.  No Newton iteration is needed —
# contrast relativistic_hydro_Tmu_1d.jl, where a finite chemical potential adds
# a conserved charge and the recovery becomes a 2-D root-find in (T, μ).
#
# Self-contained: depends only on HaloArrays, Printf, StaticArrays.
# ============================================================

using HaloArrays
using Printf
using StaticArrays

# ─── Conformal (Stefan–Boltzmann) equation of state ───────────────────────────

struct ConformalGas
    a::Float64        # p = a T⁴
end

@inline pressure(eos::ConformalGas, T)    = eos.a * T^4
@inline temperature(eos::ConformalGas, p) = (max(p, 0.0) / eos.a)^0.25
@inline sound_speed(::ConformalGas)       = 1.0 / sqrt(3.0)     # c_s² = 1/3

# ─── Primitive → conserved ────────────────────────────────────────────────────

@inline function cons_from_prim(eos, T, v)
    W = 1.0 / sqrt(1.0 - v^2)
    p = pressure(eos, T)
    w = 4.0 * p                       # w = e + p = 4p
    M = w * W^2 * v
    E = w * W^2 - p
    return SVector(M, E)
end

# ─── Conserved → primitive (closed form) ──────────────────────────────────────

@inline function prim_from_cons(eos, U)
    M, E = U
    X = (2.0 * E + sqrt(max(4.0 * E^2 - 3.0 * M^2, 0.0))) / 3.0   # = E + p
    p = max(X - E, 0.0)
    v = M / max(X, 1.0e-30)
    T = temperature(eos, p)
    return T, v
end

# ─── Fluxes ───────────────────────────────────────────────────────────────────
# For U = (M, E):  F_M = T^xx = M v + p,  F_E = T^0x = M.

@inline function physical_flux(eos, U)
    T, v = prim_from_cons(eos, U)
    p = pressure(eos, T)
    return SVector(U[1] * v + p, U[1])
end

@inline function max_wave_speed(eos, U)
    _, v = prim_from_cons(eos, U)
    cs = sound_speed(eos)
    return max(abs((v - cs) / (1.0 - v * cs)), abs((v + cs) / (1.0 + v * cs)))
end

@inline function rusanov_flux(eos, UL, UR)
    smax = max(max_wave_speed(eos, UL), max_wave_speed(eos, UR))
    return 0.5 * (physical_flux(eos, UL) + physical_flux(eos, UR)) - 0.5 * smax * (UR - UL)
end

# ─── Field access (conserved fields M, E) ─────────────────────────────────────

@inline conserved_cell(d, I) = SVector(d.M[I], d.E[I])

@inline function add_conserved!(d, I, scale, U)
    d.M[I] += scale * U[1]
    d.E[I] += scale * U[2]
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
    for I in CartesianIndices(interior_range(u.E))
        amax = max(amax, max_wave_speed(eos, conserved_cell(d, I)))
    end
    return cfl * dx / max(amax, 1.0e-14)
end

function diagnostics(u, eos, dx)
    d = parent(u)
    energy = 0.0; vmax = 0.0; tmax = 0.0
    for I in CartesianIndices(interior_range(u.E))
        U = conserved_cell(d, I)
        T, v = prim_from_cons(eos, U)
        energy += U[2] * dx            # ∫ E dx = ∫ T^00 dx
        vmax    = max(vmax, abs(v))
        tmax    = max(tmax, T)
    end
    return energy, vmax, tmax
end

# ─── Driver: conformal relativistic Sod tube ──────────────────────────────────

function run_mu0_sod(; a=1.0, nx=400, cfl=0.4, t_end=0.4)
    eos = ConformalGas(a)
    dx  = 1.0 / nx

    u  = LocalMultiHaloArray(Float64, (nx,), 1;
        fields=(:M, :E), boundary_condition=:repeating)
    u1 = similar(u)
    du = similar(u)

    # Sod-like states given as pressure; T = (p/a)^{1/4}, v = 0.
    function set_state!(i, p)
        T = temperature(eos, p)
        U = cons_from_prim(eos, T, 0.0)
        interior_view(u.M)[i] = U[1]
        interior_view(u.E)[i] = U[2]
    end
    for i in 1:nx
        x = (i - 0.5) * dx
        set_state!(i, x < 0.5 ? 1.0 : 0.1)
    end
    synchronize_halo!(u)

    e0, _, _ = diagnostics(u, eos, dx)
    @printf("Relativistic Sod — μ=0 conformal gas (p=aT⁴, c_s²=1/3)\n")
    @printf("  nx=%d  t_end=%.2f  initial energy=%.6f\n", nx, t_end, e0)

    t = 0.0; step = 0
    while t < t_end
        dt = min(cfl_dt(u, eos, dx, cfl), t_end - t)
        ssprk2_step!(u, u1, du, eos, dt, dx)
        t += dt; step += 1
    end

    e1, vmax, _ = diagnostics(u, eos, dx)
    @printf("  final    energy=%.6f  vmax=%.4f  steps=%d  t=%.4f\n", e1, vmax, step, t)

    # The shock reheats the originally cool right state (x ≈ 0.75).
    d = parent(u)
    h = halo_width(u.E)
    i_probe = h + round(Int, 0.75 * nx)
    T_probe, _ = prim_from_cons(eos, conserved_cell(d, CartesianIndex(i_probe)))
    T_right0 = temperature(eos, 0.1)
    @printf("  T at x≈0.75 = %.4f  (initial right = %.4f, shock heats it)\n", T_probe, T_right0)
    @printf("  vmax        = %.4f  (> 0 confirms bulk motion)\n", vmax)
    println(T_probe > T_right0 && vmax > 0.0 ?
        "  ✓ shock formed: matter reheated and flow is moving" :
        "  ✗ unexpected result")

    return u
end

run_mu0_sod()
