# ============================================================
# 3-D Special Relativistic Hydrodynamics — (T, μ) primitives, conserved charge
#
#   julia --project=. examples/finite_volume/relativistic_hydro_Tmu_3d.jl
#
# The 3-D extension of relativistic_hydro_Tmu_2d.jl: a charge-carrying fluid with
# the ultrarelativistic classical EOS  p(T,μ) = A T⁴ e^{μ/T},  n = p/T,  e = 3p,
# w = 4p,  c_s² = 1/3. Conserved variables = charge + three momenta + energy:
#
#   N = n W,   Mⁱ = w W² vⁱ,   E = w W² − p,   v² = vₓ²+v_y²+v_z².
#
# Recovery is a 2×2 Newton in (T, μ) using |M|² = Mₓ²+M_y²+M_z²; then vⁱ = Mⁱ/X.
# The update is three `accumulate_flux_divergence!` sweeps with Mignone–Bodo
# directional wave speeds. A centred over-density/over-pressure sphere drives a
# spherical blast; both charge and energy are conserved. (Newton recovery is
# heavier than the μ=0 closed form, hence the smaller default grid.)
# Self-contained.
# ============================================================

using HaloArrays
using Printf
using StaticArrays

struct UltraRelGas
    A::Float64
end

@inline _exp_muT(μ, T) = exp(clamp(μ / T, -700.0, 700.0))
@inline pressure(eos::UltraRelGas, T, μ)         = eos.A * T^4 * _exp_muT(μ, T)
@inline charge_density(eos::UltraRelGas, T, μ)   = eos.A * T^3 * _exp_muT(μ, T)
@inline enthalpy_density(eos::UltraRelGas, T, μ) = 4.0 * pressure(eos, T, μ)
@inline sound_speed_sq(::UltraRelGas)            = 1.0 / 3.0

@inline function cons_from_prim(eos, T, μ, vx, vy, vz)
    W = 1.0 / sqrt(1.0 - vx^2 - vy^2 - vz^2)
    p = pressure(eos, T, μ)
    n = charge_density(eos, T, μ)
    w = enthalpy_density(eos, T, μ)
    return SVector(n * W, w * W^2 * vx, w * W^2 * vy, w * W^2 * vz, w * W^2 - p)
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
    N, Mx, My, Mz, E = U
    M2 = Mx^2 + My^2 + Mz^2

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
    invX = 1.0 / max(E + p, 1.0e-30)
    return T, μ, Mx * invX, My * invX, Mz * invX
end

@inline function physical_flux(eos, U, dim::Int)
    T, μ, vx, vy, vz = prim_from_cons(eos, U)
    p = pressure(eos, T, μ)
    N, Mx, My, Mz, _ = U
    if dim == 1
        return SVector(N * vx, Mx * vx + p, My * vx, Mz * vx, Mx)
    elseif dim == 2
        return SVector(N * vy, Mx * vy, My * vy + p, Mz * vy, My)
    else
        return SVector(N * vz, Mx * vz, My * vz, Mz * vz + p, Mz)
    end
end

@inline function max_wave_speed(eos, U, dim::Int)
    _, _, vx, vy, vz = prim_from_cons(eos, U)
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

@inline conserved_cell(d, I) = SVector(d.N[I], d.Mx[I], d.My[I], d.Mz[I], d.E[I])

@inline function add_conserved!(d, I, scale, U)
    d.N[I]  += scale * U[1]
    d.Mx[I] += scale * U[2]
    d.My[I] += scale * U[3]
    d.Mz[I] += scale * U[4]
    d.E[I]  += scale * U[5]
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
    charge = 0.0; energy = 0.0; vmax = 0.0
    for I in CartesianIndices(interior_range(u.E))
        U = conserved_cell(d, I)
        _, _, vx, vy, vz = prim_from_cons(eos, U)
        charge += U[1] * dV
        energy += U[5] * dV
        vmax    = max(vmax, sqrt(vx^2 + vy^2 + vz^2))
    end
    return charge, energy, vmax
end

function probe_n(u, eos, i, j, k)
    d = field_storages(u)
    h = halo_width(u.E)
    T, μ, _, _, _ = prim_from_cons(eos, conserved_cell(d, CartesianIndex(i + h, j + h, k + h)))
    return charge_density(eos, T, μ)
end

function run_Tmu_blast_3d(; A=1.0, n=32, cfl=0.3, t_end=0.14,
        n_in=1.0, p_in=1.0, n_out=0.125, p_out=0.1, radius=0.15)
    eos = UltraRelGas(A)
    dx = 1.0 / n;  dy = dx;  dz = dx;  dV = dx * dy * dz

    u  = LocalMultiHaloArray(Float64, (n, n, n), 1;
        fields=(:N, :Mx, :My, :Mz, :E), boundary_condition=:repeating)
    u1 = similar(u)
    du = similar(u)

    function set_cell!(i, j, k, nv, pv)
        T = pv / nv
        μ = T * log(nv / (A * T^3))
        U = cons_from_prim(eos, T, μ, 0.0, 0.0, 0.0)
        interior_view(u.N)[i, j, k]  = U[1]
        interior_view(u.Mx)[i, j, k] = U[2]
        interior_view(u.My)[i, j, k] = U[3]
        interior_view(u.Mz)[i, j, k] = U[4]
        interior_view(u.E)[i, j, k]  = U[5]
    end
    for k in 1:n, j in 1:n, i in 1:n
        x = (i - 0.5) * dx;  y = (j - 0.5) * dy;  z = (k - 0.5) * dz
        sqrt((x - 0.5)^2 + (y - 0.5)^2 + (z - 0.5)^2) < radius ?
            set_cell!(i, j, k, n_in, p_in) : set_cell!(i, j, k, n_out, p_out)
    end
    synchronize_halo!(u)

    q0, e0, _ = diagnostics(u, eos, dV)
    @printf("3-D relativistic blast — (T,μ) charge-carrying gas (c_s²=1/3)\n")
    @printf("  grid=%d³  t_end=%.2f  initial charge=%.6f energy=%.6f\n", n, t_end, q0, e0)

    t = 0.0; step = 0
    while t < t_end
        dt = min(cfl_dt(u, eos, dx, dy, dz, cfl), t_end - t)
        ssprk2_step!(u, u1, du, eos, dt, dx, dy, dz)
        t += dt; step += 1
    end

    q1, e1, vmax = diagnostics(u, eos, dV)
    @printf("  final    charge=%.6f (Δ=%.2e)  energy=%.6f (Δ=%.2e)  vmax=%.4f  steps=%d\n",
        q1, q1 - q0, e1, e1 - e0, vmax, step)

    c = n ÷ 2;  off = round(Int, 0.30 * n)
    nx_p = probe_n(u, eos, c + off, c, c)
    ny_p = probe_n(u, eos, c, c + off, c)
    nz_p = probe_n(u, eos, c, c, c + off)
    sym  = (max(nx_p, ny_p, nz_p) - min(nx_p, ny_p, nz_p)) / max(nx_p, ny_p, nz_p)
    @printf("  n(+x,+y,+z)=%.4f,%.4f,%.4f  asymmetry=%.2e  vmax=%.4f\n", nx_p, ny_p, nz_p, sym, vmax)
    println((abs(q1 - q0) / q0 < 1e-3 && abs(e1 - e0) / e0 < 1e-3 && vmax > 0.0 && sym < 5e-2) ?
        "  ✓ spherical blast: charge & energy conserved, flow moving, x/y/z symmetric" :
        "  ✗ unexpected result")
    return u
end

#u=run_Tmu_blast_3d(n=64)

#N, Mx,My, Mz,E = interior_view.(eachfield(u))

#heatmap(N[16,:,:])
#surface(N[16,:,:])
#heatmap(E[16,:,:])
#surface(E[16,:,:])
#heatmap(Mx[16,:,:])
#surface(Mx[16,:,:])
