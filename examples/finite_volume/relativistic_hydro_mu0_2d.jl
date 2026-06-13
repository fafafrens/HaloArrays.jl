# ============================================================
# 2-D Special Relativistic Hydrodynamics — μ = 0 (charge-free, conformal)
#
#   julia --project=. examples/finite_volume/relativistic_hydro_mu0_2d.jl
#
# The 2-D extension of relativistic_hydro_mu0_1d.jl: the charge-free,
# high-temperature limit (μ_B ≈ 0) with the conformal EOS
#
#   p(T) = a T⁴,   e = 3p,   w = e + p = 4p,   c_s² = 1/3.
#
# Conserved variables are the raw stress-energy components — two momentum
# components and the energy (no charge to subtract):
#
#   Mⁱ = T^0i = w W² vⁱ,   E = T^00 = w W² − p,   W = (1 − v²)^{−½},  v² = vₓ² + v_y²
#
# Conserved → primitive stays CLOSED FORM. The recovery only ever sees the
# momentum through |M|² = Mₓ² + M_y², so the 1-D quadratic carries over verbatim;
# with X = E + p = wW²,
#
#   3X² − 4E X + |M|² = 0  ⇒  X = (2E + √(4E²−3|M|²))/3,  p = X−E,  vⁱ = Mⁱ/X.
#
# Fluxes are direction-aware (T^ij = Mⁱ vʲ + p δ^ij, energy flux T^0j = Mʲ), and
# the per-cell update is two `accumulate_flux_divergence!` sweeps (x then y).
# The signal speed uses the directional relativistic eigenvalues
# (Mignone & Bodo 2005), which reduce to (vₙ ± c_s)/(1 ± vₙ c_s) when the
# transverse velocity vanishes.
#
# Test: a centred over-pressure disk drives a circular blast wave. Self-contained
# (HaloArrays, Printf, StaticArrays).
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
@inline sound_speed_sq(::ConformalGas)    = 1.0 / 3.0          # c_s² = 1/3

# ─── Primitive ↔ conserved ────────────────────────────────────────────────────

@inline function cons_from_prim(eos, T, vx, vy)
    W = 1.0 / sqrt(1.0 - vx^2 - vy^2)
    p = pressure(eos, T)
    w = 4.0 * p                          # w = e + p = 4p
    Mx = w * W^2 * vx
    My = w * W^2 * vy
    E  = w * W^2 - p
    return SVector(Mx, My, E)
end

@inline function prim_from_cons(eos, U)
    Mx, My, E = U
    M2 = Mx^2 + My^2
    X  = (2.0 * E + sqrt(max(4.0 * E^2 - 3.0 * M2, 0.0))) / 3.0    # = E + p
    p  = max(X - E, 0.0)
    invX = 1.0 / max(X, 1.0e-30)
    return temperature(eos, p), Mx * invX, My * invX               # T, vx, vy
end

# ─── Direction-aware flux and signal speed ────────────────────────────────────
# dim 1 → x-flux  (Mₓvₓ+p, M_yvₓ, Mₓ);  dim 2 → y-flux  (Mₓv_y, M_yv_y+p, M_y).

@inline function physical_flux(eos, U, dim::Int)
    T, vx, vy = prim_from_cons(eos, U)
    p = pressure(eos, T)
    Mx, My, _ = U
    if dim == 1
        return SVector(Mx * vx + p, My * vx, Mx)
    else
        return SVector(Mx * vy, My * vy + p, My)
    end
end

# Extremal relativistic eigenvalues along `dim` (Mignone & Bodo 2005).
@inline function max_wave_speed(eos, U, dim::Int)
    _, vx, vy = prim_from_cons(eos, U)
    v2  = vx^2 + vy^2
    vn  = dim == 1 ? vx : vy
    cs2 = sound_speed_sq(eos)
    disc = (1.0 - v2) * ((1.0 - v2 * cs2) - vn^2 * (1.0 - cs2))
    root = sqrt(max(disc, 0.0)) * sqrt(cs2)
    den  = 1.0 - v2 * cs2
    λp = (vn * (1.0 - cs2) + root) / den
    λm = (vn * (1.0 - cs2) - root) / den
    return max(abs(λp), abs(λm))
end

@inline function rusanov_flux(eos, UL, UR, dim::Int)
    smax = max(max_wave_speed(eos, UL, dim), max_wave_speed(eos, UR, dim))
    return 0.5 * (physical_flux(eos, UL, dim) + physical_flux(eos, UR, dim)) -
           0.5 * smax * (UR - UL)
end

# ─── Field access (conserved fields Mx, My, E) ────────────────────────────────

@inline conserved_cell(d, I) = SVector(d.Mx[I], d.My[I], d.E[I])

@inline function add_conserved!(d, I, scale, U)
    d.Mx[I] += scale * U[1]
    d.My[I] += scale * U[2]
    d.E[I]  += scale * U[3]
    return d
end

# ─── RHS (x then y sweep) and SSP-RK2 ─────────────────────────────────────────
#
# Two conservative `accumulate_flux_divergence!` sweeps (x then y). The internal
# faces are direction-aware (transverse-full), so a uniform state gives zero
# update everywhere, including the edges.

function rel_rhs!(du, u, eos, dx, dy)
    fill!(du, 0.0)
    synchronize_halo!(u)                  # :repeating fills the outflow ghosts
    fr = FaceRanges(u)
    accumulate_flux_divergence!(field_storages(du), field_storages(u), fr, 1, inv(dx),
        (UL, UR) -> rusanov_flux(eos, UL, UR, 1), conserved_cell, add_conserved!)
    accumulate_flux_divergence!(field_storages(du), field_storages(u), fr, 2, inv(dy),
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
    d = field_storages(u)
    amax = 0.0
    for I in CartesianIndices(interior_range(u.E))
        U = conserved_cell(d, I)
        amax = max(amax, max_wave_speed(eos, U, 1) / dx + max_wave_speed(eos, U, 2) / dy)
    end
    return cfl / max(amax, 1.0e-14)
end

function diagnostics(u, eos, dx, dy)
    d = field_storages(u)
    energy = 0.0; vmax = 0.0; tmax = 0.0
    for I in CartesianIndices(interior_range(u.E))
        U = conserved_cell(d, I)
        T, vx, vy = prim_from_cons(eos, U)
        energy += U[3] * dx * dy           # ∫ T^00 dV
        vmax    = max(vmax, sqrt(vx^2 + vy^2))
        tmax    = max(tmax, T)
    end
    return energy, vmax, tmax
end

# probe temperature at global cell (i, j)
function probe_T(u, eos, i, j)
    d = field_storages(u)
    h = halo_width(u.E)
    T, _, _ = prim_from_cons(eos, conserved_cell(d, CartesianIndex(i + h, j + h)))
    return T
end

# ─── Driver: 2-D conformal relativistic blast wave ────────────────────────────

function run_mu0_blast_2d(; a=1.0, n=160, cfl=0.3, t_end=0.20,
        p_in=1.0, p_out=0.05, radius=0.15)
    eos = ConformalGas(a)
    dx = 1.0 / n;  dy = 1.0 / n

    u  = LocalMultiHaloArray(Float64, (n, n), 1;
        fields=(:Mx, :My, :E), boundary_condition=:repeating)
    u1 = similar(u)
    du = similar(u)

    # centred over-pressure disk, fluid initially at rest
    for j in 1:n, i in 1:n
        x = (i - 0.5) * dx;  y = (j - 0.5) * dy
        p = hypot(x - 0.5, y - 0.5) < radius ? p_in : p_out
        U = cons_from_prim(eos, temperature(eos, p), 0.0, 0.0)
        interior_view(u.Mx)[i, j] = U[1]
        interior_view(u.My)[i, j] = U[2]
        interior_view(u.E)[i, j]  = U[3]
    end
    synchronize_halo!(u)

    e0, _, _ = diagnostics(u, eos, dx, dy)
    @printf("2-D relativistic blast — μ=0 conformal gas (p=aT⁴, c_s²=1/3)\n")
    @printf("  grid=%d×%d  t_end=%.2f  initial energy=%.6f\n", n, n, t_end, e0)

    t = 0.0; step = 0
    while t < t_end
        dt = min(cfl_dt(u, eos, dx, dy, cfl), t_end - t)
        ssprk2_step!(u, u1, du, eos, dt, dx, dy)
        t += dt; step += 1
    end

    e1, vmax, _ = diagnostics(u, eos, dx, dy)
    @printf("  final    energy=%.6f  (Δ=%.2e)  vmax=%.4f  steps=%d  t=%.4f\n",
        e1, e1 - e0, vmax, step, t)

    # radial symmetry: T sampled at the same radius along +x and +y from centre
    c = n ÷ 2;  off = round(Int, 0.30 * n)
    Tx = probe_T(u, eos, c + off, c)
    Ty = probe_T(u, eos, c, c + off)
    sym = abs(Tx - Ty) / max(Tx, Ty)
    @printf("  T(+x)=%.4f  T(+y)=%.4f  asymmetry=%.2e  vmax=%.4f\n", Tx, Ty, sym, vmax)
    println((abs(e1 - e0) / e0 < 1e-3 && vmax > 0.0 && sym < 5e-2) ?
        "  ✓ circular blast: energy conserved, flow moving, x/y symmetric" :
        "  ✗ unexpected result")
    return u
end

#u=run_mu0_blast_2d()


#using Plots
#Mx,My,E=interior_view.(eachfield(u))

#heatmap(E)
#surface(E)can you app