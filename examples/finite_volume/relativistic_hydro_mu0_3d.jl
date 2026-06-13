# ============================================================
# 3-D Special Relativistic Hydrodynamics — μ = 0 (charge-free, conformal)
#
#   julia --project=. examples/finite_volume/relativistic_hydro_mu0_3d.jl
#
# The 3-D extension of relativistic_hydro_mu0_2d.jl: charge-free conformal fluid,
#
#   p(T) = a T⁴,   e = 3p,   w = 4p,   c_s² = 1/3,
#
# conserved variables = the three momentum components + energy,
#
#   Mⁱ = w W² vⁱ,   E = w W² − p,   W = (1 − v²)^{−½},  v² = vₓ²+v_y²+v_z².
#
# Recovery is still the same closed form — it sees the momentum only through
# |M|² = Mₓ²+M_y²+M_z² — and the update is three `accumulate_flux_divergence!`
# sweeps (x, y, z) with Mignone–Bodo directional wave speeds. A centred
# over-pressure sphere drives a spherical blast. Self-contained.
# ============================================================

using HaloArrays
using Printf
using StaticArrays

struct ConformalGas
    a::Float64
end

@inline pressure(eos::ConformalGas, T)    = eos.a * T^4
@inline temperature(eos::ConformalGas, p) = (max(p, 0.0) / eos.a)^0.25
@inline sound_speed_sq(::ConformalGas)    = 1.0 / 3.0

@inline function cons_from_prim(eos, T, vx, vy, vz)
    W = 1.0 / sqrt(1.0 - vx^2 - vy^2 - vz^2)
    p = pressure(eos, T)
    w = 4.0 * p
    return SVector(w * W^2 * vx, w * W^2 * vy, w * W^2 * vz, w * W^2 - p)
end

@inline function prim_from_cons(eos, U)
    Mx, My, Mz, E = U
    M2 = Mx^2 + My^2 + Mz^2
    X  = (2.0 * E + sqrt(max(4.0 * E^2 - 3.0 * M2, 0.0))) / 3.0   # = E + p
    p  = max(X - E, 0.0)
    invX = 1.0 / max(X, 1.0e-30)
    return temperature(eos, p), Mx * invX, My * invX, Mz * invX   # T, vx, vy, vz
end

# Direction-aware flux: F_Mi = Mi·vₙ + p·δ_{i,dim},  F_E = M_dim  (vₙ = v[dim]).
@inline function physical_flux(eos, U, dim::Int)
    T, vx, vy, vz = prim_from_cons(eos, U)
    p = pressure(eos, T)
    Mx, My, Mz, _ = U
    if dim == 1
        return SVector(Mx * vx + p, My * vx, Mz * vx, Mx)
    elseif dim == 2
        return SVector(Mx * vy, My * vy + p, Mz * vy, My)
    else
        return SVector(Mx * vz, My * vz, Mz * vz + p, Mz)
    end
end

@inline function max_wave_speed(eos, U, dim::Int)
    _, vx, vy, vz = prim_from_cons(eos, U)
    v2  = vx^2 + vy^2 + vz^2
    vn  = dim == 1 ? vx : dim == 2 ? vy : vz
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

@inline conserved_cell(d, I) = SVector(d.Mx[I], d.My[I], d.Mz[I], d.E[I])

@inline function add_conserved!(d, I, scale, U)
    d.Mx[I] += scale * U[1]
    d.My[I] += scale * U[2]
    d.Mz[I] += scale * U[3]
    d.E[I]  += scale * U[4]
    return d
end

function rel_rhs!(du, u, eos, dx, dy, dz)
    fill!(du, 0.0)
    synchronize_halo!(u)
    fr = FaceRanges(u)
    for (dim, scale) in ((1, inv(dx)), (2, inv(dy)), (3, inv(dz)))
        accumulate_flux_divergence!(field_storages(du), field_storages(u), fr, dim, scale,
            (UL, UR) -> rusanov_flux(eos, UL, UR, dim), conserved_cell, add_conserved!)
    end
    return du
end

function ssprk2_step!(u, u1, du, eos, dt, dx, dy, dz)
    rel_rhs!(du, u, eos, dx, dy, dz)
    @. u1 = u + dt * du
    rel_rhs!(du, u1, eos, dx, dy, dz)
    @. u = 0.5 * u + 0.5 * (u1 + dt * du)
    return u
end

function cfl_dt(u, eos, dx, dy, dz, cfl)
    d = field_storages(u)
    amax = 0.0
    for I in CartesianIndices(interior_range(u.E))
        U = conserved_cell(d, I)
        amax = max(amax, max_wave_speed(eos, U, 1) / dx +
                         max_wave_speed(eos, U, 2) / dy +
                         max_wave_speed(eos, U, 3) / dz)
    end
    return cfl / max(amax, 1.0e-14)
end

function diagnostics(u, eos, dV)
    d = field_storages(u)
    energy = 0.0; vmax = 0.0
    for I in CartesianIndices(interior_range(u.E))
        U = conserved_cell(d, I)
        _, vx, vy, vz = prim_from_cons(eos, U)
        energy += U[4] * dV
        vmax    = max(vmax, sqrt(vx^2 + vy^2 + vz^2))
    end
    return energy, vmax
end

function probe_T(u, eos, i, j, k)
    d = field_storages(u)
    h = halo_width(u.E)
    T, _, _, _ = prim_from_cons(eos, conserved_cell(d, CartesianIndex(i + h, j + h, k + h)))
    return T
end

function run_mu0_blast_3d(; a=1.0, n=64, cfl=0.3, t_end=0.16,
        p_in=1.0, p_out=0.05, radius=0.15)
    eos = ConformalGas(a)
    dx = 1.0 / n;  dy = dx;  dz = dx;  dV = dx * dy * dz

    u  = LocalMultiHaloArray(Float64, (n, n, n), 1;
        fields=(:Mx, :My, :Mz, :E), boundary_condition=:repeating)
    u1 = similar(u)
    du = similar(u)

    for k in 1:n, j in 1:n, i in 1:n
        x = (i - 0.5) * dx;  y = (j - 0.5) * dy;  z = (k - 0.5) * dz
        p = sqrt((x - 0.5)^2 + (y - 0.5)^2 + (z - 0.5)^2) < radius ? p_in : p_out
        U = cons_from_prim(eos, temperature(eos, p), 0.0, 0.0, 0.0)
        interior_view(u.Mx)[i, j, k] = U[1]
        interior_view(u.My)[i, j, k] = U[2]
        interior_view(u.Mz)[i, j, k] = U[3]
        interior_view(u.E)[i, j, k]  = U[4]
    end
    synchronize_halo!(u)

    e0, _ = diagnostics(u, eos, dV)
    @printf("3-D relativistic blast — μ=0 conformal gas (p=aT⁴, c_s²=1/3)\n")
    @printf("  grid=%d³  t_end=%.2f  initial energy=%.6f\n", n, t_end, e0)

    t = 0.0; step = 0
    while t < t_end
        dt = min(cfl_dt(u, eos, dx, dy, dz, cfl), t_end - t)
        ssprk2_step!(u, u1, du, eos, dt, dx, dy, dz)
        t += dt; step += 1
    end

    e1, vmax = diagnostics(u, eos, dV)
    @printf("  final    energy=%.6f  (Δ=%.2e)  vmax=%.4f  steps=%d  t=%.4f\n",
        e1, e1 - e0, vmax, step, t)

    # spherical symmetry: T at the same radius along +x, +y, +z
    c = n ÷ 2;  off = round(Int, 0.30 * n)
    Tx = probe_T(u, eos, c + off, c, c)
    Ty = probe_T(u, eos, c, c + off, c)
    Tz = probe_T(u, eos, c, c, c + off)
    sym = (max(Tx, Ty, Tz) - min(Tx, Ty, Tz)) / max(Tx, Ty, Tz)
    @printf("  T(+x,+y,+z)=%.4f,%.4f,%.4f  asymmetry=%.2e  vmax=%.4f\n", Tx, Ty, Tz, sym, vmax)
    println((abs(e1 - e0) / e0 < 1e-3 && vmax > 0.0 && sym < 5e-2) ?
        "  ✓ spherical blast: energy conserved, flow moving, x/y/z symmetric" :
        "  ✗ unexpected result")
    return u
end

#u=run_mu0_blast_3d()
