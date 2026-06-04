# ============================================================
# 2-D Special Relativistic Hydrodynamics — (T, μ) primitives, conserved charge
#
#   julia --project=. examples/finite_volume/relativistic_hydro_Tmu_2d.jl
#
# The 2-D extension of relativistic_hydro_Tmu_1d.jl: a finite-chemical-potential
# fluid carrying a conserved CHARGE, with primitive thermodynamic variables
# (T, μ) and the ultrarelativistic classical EOS
#
#   p(T,μ) = A T⁴ e^{μ/T},  n = p/T,  e = 3p,  w = e+p = 4p,  c_s² = 1/3.
#
# Conserved variables — charge + the two momentum components + energy:
#
#   N = n W,   Mⁱ = T^0i = w W² vⁱ,   E = T^00 = w W² − p,
#   W = (1 − v²)^{−½},   v² = vₓ² + v_y².
#
# Conserved → primitive is a 2×2 Newton in (T, μ): the recovery only sees the
# momentum through |M|² = Mₓ² + M_y², so the 1-D solver carries over with that
# substitution; then vⁱ = Mⁱ/X with X = E + p. Fluxes are direction-aware
# (charge flux N vʲ, T^ij = Mⁱ vʲ + p δ^ij, energy flux T^0j = Mʲ) and the update
# is two `accumulate_flux_divergence!` sweeps. The signal speed uses the
# directional relativistic eigenvalues (Mignone & Bodo 2005).
#
# Test: a centred over-density/over-pressure disk drives a circular blast; both
# the charge and the energy are conserved. Self-contained.
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
@inline enthalpy_density(eos::UltraRelGas, T, μ) = 4.0 * pressure(eos, T, μ)      # w = 4p
@inline sound_speed_sq(::UltraRelGas)            = 1.0 / 3.0                       # c_s² = 1/3

# ─── Primitive ↔ conserved ────────────────────────────────────────────────────

@inline function cons_from_prim(eos, T, μ, vx, vy)
    W = 1.0 / sqrt(1.0 - vx^2 - vy^2)
    p = pressure(eos, T, μ)
    n = charge_density(eos, T, μ)
    w = enthalpy_density(eos, T, μ)
    N  = n * W
    Mx = w * W^2 * vx
    My = w * W^2 * vy
    E  = w * W^2 - p
    return SVector(N, Mx, My, E)
end

# Two residuals enforcing the charge and enthalpy identities at trial (T, μ),
# with |M|² entering exactly as in 1-D (here |M|² = Mₓ² + M_y²).
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
    N, Mx, My, E = U
    M2 = Mx^2 + My^2

    T = max(E / (3.0 * max(N, 1.0e-12)), 1.0e-8)             # W≈1 seed
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
    X = E + p
    invX = 1.0 / max(X, 1.0e-30)
    return T, μ, Mx * invX, My * invX                        # T, μ, vx, vy
end

# ─── Direction-aware flux and signal speed ────────────────────────────────────

@inline function physical_flux(eos, U, dim::Int)
    T, μ, vx, vy = prim_from_cons(eos, U)
    p = pressure(eos, T, μ)
    N, Mx, My, _ = U
    if dim == 1
        return SVector(N * vx, Mx * vx + p, My * vx, Mx)
    else
        return SVector(N * vy, Mx * vy, My * vy + p, My)
    end
end

@inline function max_wave_speed(eos, U, dim::Int)
    _, _, vx, vy = prim_from_cons(eos, U)
    v2  = vx^2 + vy^2
    vn  = dim == 1 ? vx : vy
    cs2 = sound_speed_sq(eos)
    disc = (1.0 - v2) * ((1.0 - v2 * cs2) - vn^2 * (1.0 - cs2))
    root = sqrt(max(disc, 0.0)) * sqrt(cs2)
    den  = 1.0 - v2 * cs2
    return max(abs((vn * (1.0 - cs2) + root) / den), abs((vn * (1.0 - cs2) - root) / den))
end

@inline function rusanov_flux(eos, UL, UR, dim::Int)
    smax = max(max_wave_speed(eos, UL, dim), max_wave_speed(eos, UR, dim))
    return 0.5 * (physical_flux(eos, UL, dim) + physical_flux(eos, UR, dim)) -
           0.5 * smax * (UR - UL)
end

# ─── Field access (conserved fields N, Mx, My, E) ─────────────────────────────

@inline conserved_cell(d, I) = SVector(d.N[I], d.Mx[I], d.My[I], d.E[I])

@inline function add_conserved!(d, I, scale, U)
    d.N[I]  += scale * U[1]
    d.Mx[I] += scale * U[2]
    d.My[I] += scale * U[3]
    d.E[I]  += scale * U[4]
    return d
end

# ─── RHS (x then y sweep) and SSP-RK2 ─────────────────────────────────────────

function rel_rhs!(du, u, eos, dx, dy)
    fill!(du, 0.0)
    synchronize_halo!(u)                  # :repeating fills the outflow ghosts
    fr = FaceRanges(u)
    accumulate_flux_divergence!(parent(du), parent(u), fr, 1, inv(dx),
        (UL, UR) -> rusanov_flux(eos, UL, UR, 1), conserved_cell, add_conserved!)
    accumulate_flux_divergence!(parent(du), parent(u), fr, 2, inv(dy),
        (UL, UR) -> rusanov_flux(eos, UL, UR, 2), conserved_cell, add_conserved!)
    return du
end

function ssprk2_step!(u, u1, du, eos, dt, dx, dy)
    rel_rhs!(du, u, eos, dx, dy)
    @. u1 = u + dt * du
    rel_rhs!(du, u1, eos, dx, dy)
    @. u = 0.5 * u + 0.5 * (u1 + dt * du)
    return u
end

# ─── Diagnostics ──────────────────────────────────────────────────────────────

function cfl_dt(u, eos, dx, dy, cfl)
    d = parent(u)
    amax = 0.0
    for I in CartesianIndices(interior_range(u.E))
        U = conserved_cell(d, I)
        amax = max(amax, max_wave_speed(eos, U, 1) / dx + max_wave_speed(eos, U, 2) / dy)
    end
    return cfl / max(amax, 1.0e-14)
end

function diagnostics(u, eos, dx, dy)
    d = parent(u)
    charge = 0.0; energy = 0.0; vmax = 0.0
    for I in CartesianIndices(interior_range(u.E))
        U = conserved_cell(d, I)
        _, _, vx, vy = prim_from_cons(eos, U)
        charge += U[1] * dx * dy
        energy += U[4] * dx * dy
        vmax    = max(vmax, sqrt(vx^2 + vy^2))
    end
    return charge, energy, vmax
end

function probe_n(u, eos, i, j)
    d = parent(u)
    h = halo_width(u.E)
    T, μ, _, _ = prim_from_cons(eos, conserved_cell(d, CartesianIndex(i + h, j + h)))
    return charge_density(eos, T, μ)
end

# ─── Driver: 2-D charge-carrying relativistic blast wave ──────────────────────

function run_Tmu_blast_2d(; A=1.0, n=160, cfl=0.3, t_end=0.20,
        n_in=1.0, p_in=1.0, n_out=0.125, p_out=0.1, radius=0.15)
    eos = UltraRelGas(A)
    dx = 1.0 / n;  dy = 1.0 / n

    u  = LocalMultiHaloArray(Float64, (n, n), 1;
        fields=(:N, :Mx, :My, :E), boundary_condition=:repeating)
    u1 = similar(u)
    du = similar(u)

    function set_cell!(i, j, nv, pv)
        T = pv / nv                          # n = p/T ⇒ T = p/n
        μ = T * log(nv / (A * T^3))
        U = cons_from_prim(eos, T, μ, 0.0, 0.0)
        interior_view(u.N)[i, j]  = U[1]
        interior_view(u.Mx)[i, j] = U[2]
        interior_view(u.My)[i, j] = U[3]
        interior_view(u.E)[i, j]  = U[4]
    end
    for j in 1:n, i in 1:n
        x = (i - 0.5) * dx;  y = (j - 0.5) * dy
        hypot(x - 0.5, y - 0.5) < radius ? set_cell!(i, j, n_in, p_in) :
                                           set_cell!(i, j, n_out, p_out)
    end
    synchronize_halo!(u)

    q0, e0, _ = diagnostics(u, eos, dx, dy)
    @printf("2-D relativistic blast — (T,μ) charge-carrying gas (c_s²=1/3)\n")
    @printf("  grid=%d×%d  t_end=%.2f  initial charge=%.6f energy=%.6f\n", n, n, t_end, q0, e0)

    t = 0.0; step = 0
    while t < t_end
        dt = min(cfl_dt(u, eos, dx, dy, cfl), t_end - t)
        ssprk2_step!(u, u1, du, eos, dt, dx, dy)
        t += dt; step += 1
    end

    q1, e1, vmax = diagnostics(u, eos, dx, dy)
    @printf("  final    charge=%.6f (Δ=%.2e)  energy=%.6f (Δ=%.2e)  vmax=%.4f  steps=%d\n",
        q1, q1 - q0, e1, e1 - e0, vmax, step)

    c = n ÷ 2;  off = round(Int, 0.30 * n)
    nx_p = probe_n(u, eos, c + off, c)
    ny_p = probe_n(u, eos, c, c + off)
    sym  = abs(nx_p - ny_p) / max(nx_p, ny_p)
    @printf("  n(+x)=%.4f  n(+y)=%.4f  asymmetry=%.2e  vmax=%.4f\n", nx_p, ny_p, sym, vmax)
    println((abs(q1 - q0) / q0 < 1e-3 && abs(e1 - e0) / e0 < 1e-3 && vmax > 0.0 && sym < 5e-2) ?
        "  ✓ circular blast: charge & energy conserved, flow moving, x/y symmetric" :
        "  ✗ unexpected result")
    return u
end

run_Tmu_blast_2d()
