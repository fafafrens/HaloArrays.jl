# ============================================================
# Benchmark: serial vs threaded cylindrical relativistic hydro RHS+step
#
#   for t in 1 2 4 8; do julia -t $t --project benchmarks/bench_cylindrical_threaded.jl; done
#
# Times the SSP-RK2 step (synchronize + geometric source + Rusanov flux) of the
# (N,M,E) ultrarelativistic cylindrical blast, for a serial LocalHaloArray state
# and a ThreadedHaloArray state tiled into Threads.nthreads() tiles, at equal
# problem size. Prints one row per process: threads, ms/step (serial),
# ms/step (threaded), speedup.
# ============================================================

using HaloArrays
using Printf
using StaticArrays

const GEOMETRY = 1.0

struct UltraRelGas; A::Float64; end
@inline _exp_muT(μ, T) = exp(clamp(μ / T, -700.0, 700.0))
@inline pressure(eos::UltraRelGas, T, μ)         = eos.A * T^4 * _exp_muT(μ, T)
@inline charge_density(eos::UltraRelGas, T, μ)   = eos.A * T^3 * _exp_muT(μ, T)
@inline enthalpy_density(eos::UltraRelGas, T, μ) = 4.0 * pressure(eos, T, μ)
@inline sound_speed(::UltraRelGas)               = 1.0 / sqrt(3.0)

@inline function cons_from_prim(eos, T, μ, v)
    W = 1.0 / sqrt(1.0 - v^2)
    p = pressure(eos, T, μ); n = charge_density(eos, T, μ); w = enthalpy_density(eos, T, μ)
    return SVector(n * W, w * W^2 * v, w * W^2 - p)
end

@inline function _residuals(eos, E, M2, N, T, μ)
    p = pressure(eos, T, μ); n = charge_density(eos, T, μ); w = enthalpy_density(eos, T, μ)
    X = E + p; Z = sqrt(max(X^2 - M2, 1.0e-30)); W = X / Z
    return n * W - N, w * W^2 - X
end

function prim_from_cons(eos, U; maxit=200, tol=1.0e-11)
    N, M, E = U; M2 = M^2
    T = max(E / (3.0 * max(N, 1.0e-12)), 1.0e-8)
    μ = T * log(max(max(N, 1.0e-12) / (eos.A * T^3), 1.0e-300))
    for _ in 1:maxit
        R1, R2 = _residuals(eos, E, M2, N, T, μ)
        δT = max(1.0e-7 * abs(T), 1.0e-9); δμ = max(1.0e-7 * abs(μ), 1.0e-9)
        R1T, R2T = _residuals(eos, E, M2, N, T + δT, μ)
        R1m, R2m = _residuals(eos, E, M2, N, T, μ + δμ)
        J11 = (R1T - R1) / δT; J21 = (R2T - R2) / δT
        J12 = (R1m - R1) / δμ; J22 = (R2m - R2) / δμ
        det = J11 * J22 - J12 * J21; abs(det) < 1.0e-300 && break
        dT = (-R1 * J22 + R2 * J12) / det; dμ = (R1 * J21 - R2 * J11) / det
        T = max(T + dT, 1.0e-10); μ = μ + dμ
        (abs(dT) < tol * (abs(T) + 1.0e-12) && abs(dμ) < tol * (abs(μ) + 1.0e-12)) && break
    end
    p = pressure(eos, T, μ); v = M / (E + p)
    return T, μ, v
end

@inline function physical_flux(eos, U)
    T, μ, v = prim_from_cons(eos, U); p = pressure(eos, T, μ)
    return SVector(U[1] * v, U[2] * v + p, U[2])
end
@inline function max_wave_speed(eos, U)
    _, _, v = prim_from_cons(eos, U); cs = sound_speed(eos)
    return max(abs((v - cs) / (1.0 - v * cs)), abs((v + cs) / (1.0 + v * cs)))
end
@inline function rusanov_flux(eos, UL, UR)
    smax = max(max_wave_speed(eos, UL), max_wave_speed(eos, UR))
    return 0.5 * (physical_flux(eos, UL) + physical_flux(eos, UR)) - 0.5 * smax * (UR - UL)
end

@inline conserved_cell(d, I) = SVector(d.N[I], d.M[I], d.E[I])
@inline function add_conserved!(d, I, scale, U)
    d.N[I] += scale * U[1]; d.M[I] += scale * U[2]; d.E[I] += scale * U[3]; return d
end
@inline function set_source_tile!(du_data, u_data, eos, cells, r_of)
    @inbounds for I in cells
        U = conserved_cell(u_data, I); _, _, v = prim_from_cons(eos, U); r = r_of(I)
        S = SVector(U[1] * v, U[2] * v, U[2]) * (-GEOMETRY / r)
        du_data.N[I] = S[1]; du_data.M[I] = S[2]; du_data.E[I] = S[3]
    end
    return du_data
end

# ── serial RHS / step ──
function rel_rhs_serial!(du, u, eos, r_min, dr)
    synchronize_halo!(u)
    h = halo_width(u.N)
    set_source_tile!(parent(du), parent(u), eos, get_owned_cells(CellRanges(u)),
        I -> r_min + (I[1] - h - 0.5) * dr)
    accumulate_flux_divergence!(parent(du), parent(u), FaceRanges(u), 1, inv(dr),
        (UL, UR) -> rusanov_flux(eos, UL, UR), conserved_cell, add_conserved!)
    return du
end

# ── threaded RHS / step ──
function rel_rhs_threaded!(du, u, eos, r_min, dr)
    synchronize_halo!(u)
    ranges = FaceRanges(u); cells = get_owned_cells(CellRanges(u)); h = halo_width(u.N)
    tile_foreach(thread_backend(u.N), tile_id -> begin
        du_data = tile_parent(du, tile_id); u_data = tile_parent(u, tile_id)
        offset = owned_to_global_index(u.N, tile_id, (1,))[1] - 1
        set_source_tile!(du_data, u_data, eos, cells, I -> r_min + (offset + (I[1] - h) - 0.5) * dr)
        accumulate_flux_divergence!(du_data, u_data, ranges, 1, inv(dr),
            (UL, UR) -> rusanov_flux(eos, UL, UR), conserved_cell, add_conserved!)
    end, 1:tile_count(u); scheduler=:static)
    return du
end

function step!(rhs!, u, u1, du, eos, r_min, dr, dt)
    rhs!(du, u, eos, r_min, dr)
    @. u1 = u + dt * du
    rhs!(du, u1, eos, r_min, dr)
    @. u = 0.5 * u + 0.5 * (u1 + dt * du)
    return u
end

# ── timing ──
function time_steps(rhs!, u, u1, du, eos, r_min, dr, dt, nsteps; reps=4)
    step!(rhs!, u, u1, du, eos, r_min, dr, dt)            # warm up (compile)
    best = Inf
    for _ in 1:reps
        t = @elapsed for _ in 1:nsteps
            step!(rhs!, u, u1, du, eos, r_min, dr, dt)
        end
        best = min(best, t)
    end
    return best / nsteps
end

function run_bench(; A=1.0, nx=12000, cfl=0.4, r_min=0.0, r_max=1.0, r_mid=0.3, nsteps=30)
    eos = UltraRelGas(A); dr = (r_max - r_min) / nx; dt = cfl * dr
    nt  = max(1, Threads.nthreads())
    nx % nt == 0 || (nx -= nx % nt)        # make divisible by tile count

    prim(r) = r < r_mid ? (1.0, 1.0) : (0.125, 0.1)
    ic(gi)  = (r = r_min + (gi - 0.5) * dr; (n, p) = prim(r); T = p / n;
               cons_from_prim(eos, T, T * log(n / (A * T^3)), 0.0))

    # serial state
    us = LocalMultiHaloArray(Float64, (nx,), 1; boundary_conditions=(
        N=((:reflecting, :repeating),), M=((:antireflecting, :repeating),),
        E=((:reflecting, :repeating),)))
    for i in 1:nx
        U = ic(i)
        interior_view(us.N)[i] = U[1]; interior_view(us.M)[i] = U[2]; interior_view(us.E)[i] = U[3]
    end
    synchronize_halo!(us)
    serial_ms = 1e3 * time_steps(rel_rhs_serial!, us, similar(us), similar(us), eos, r_min, dr, dt, nsteps)

    # threaded state (nt tiles)
    ut = ThreadedMultiHaloArray(Float64, (nx ÷ nt,), 1; dims=(nt,), boundary_conditions=(
        N=((:reflecting, :repeating),), M=((:antireflecting, :repeating),),
        E=((:reflecting, :repeating),)))
    h = halo_width(ut.N)
    for tile_id in 1:tile_count(ut)
        dN = tile_parent(ut.N, tile_id); dM = tile_parent(ut.M, tile_id); dE = tile_parent(ut.E, tile_id)
        for oi in 1:tile_size(ut)[1]
            U = ic(owned_to_global_index(ut.N, tile_id, (oi,))[1])
            dN[oi + h] = U[1]; dM[oi + h] = U[2]; dE[oi + h] = U[3]
        end
    end
    synchronize_halo!(ut)
    threaded_ms = 1e3 * time_steps(rel_rhs_threaded!, ut, similar(ut), similar(ut), eos, r_min, dr, dt, nsteps)

    @printf("threads=%-2d  nx=%-6d  serial=%7.3f ms/step  threaded=%7.3f ms/step  speedup=%.2fx\n",
        nt, nx, serial_ms, threaded_ms, serial_ms / threaded_ms)
end

run_bench()
