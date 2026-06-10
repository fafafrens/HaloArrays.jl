# ============================================================
# Benchmark: 2-D μ=0 conformal relativistic hydro step — serial / threaded / MPI
#
#   julia -t 1 --project benchmarks/bench_relhydro_mu0_2d.jl        # serial baseline
#   julia -t 8 --project benchmarks/bench_relhydro_mu0_2d.jl        # serial vs threaded
#   mpiexec -n 8 julia --project benchmarks/bench_relhydro_mu0_2d.jl # serial vs MPI
#
# Conformal (Stefan–Boltzmann, μ=0) gas, p=aT⁴, e=3p, c_s²=1/3, on a periodic
# square. Conserved variables are the raw stress-energy components (Mx, My, E);
# conserved→primitive is closed form (M²=Mx²+My²). One SSP-RK2 step = two RHS
# evals, each a Rusanov flux divergence in x and y via accumulate_flux_divergence!.
# Prints serial vs the parallel backend with the speedup at this rank/thread count.
# ============================================================

using HaloArrays, MPI, Printf, StaticArrays
MPI.Init()

const A  = 1.0
const CS = 1.0 / sqrt(3.0)

@inline pressure(T)    = A * T^4
@inline temperature(p) = (max(p, 0.0) / A)^0.25

@inline function cons_from_prim(T, vx, vy)
    W = 1.0 / sqrt(1.0 - vx^2 - vy^2)
    p = pressure(T); w = 4.0 * p
    return SVector(w * W^2 * vx, w * W^2 * vy, w * W^2 - p)
end

@inline function prim_from_cons(U)                       # closed form (μ=0 conformal)
    Mx, My, E = U
    M2 = Mx^2 + My^2
    X  = (2.0 * E + sqrt(max(4.0 * E^2 - 3.0 * M2, 0.0))) / 3.0   # = E + p
    p  = max(X - E, 0.0)
    iX = 1.0 / max(X, 1.0e-30)
    return temperature(p), Mx * iX, My * iX
end

@inline function physical_flux(U, dim)
    T, vx, vy = prim_from_cons(U); p = pressure(T)
    return dim == 1 ? SVector(U[1] * vx + p, U[2] * vx, U[1]) :
                      SVector(U[1] * vy, U[2] * vy + p, U[2])
end
@inline function max_wave_speed(U, dim)
    _, vx, vy = prim_from_cons(U); vd = dim == 1 ? vx : vy
    return max(abs((vd - CS) / (1.0 - vd * CS)), abs((vd + CS) / (1.0 + vd * CS)))
end
@inline function rusanov(UL, UR, dim)
    s = max(max_wave_speed(UL, dim), max_wave_speed(UR, dim))
    return 0.5 * (physical_flux(UL, dim) + physical_flux(UR, dim)) - 0.5 * s * (UR - UL)
end

@inline conserved_cell(d, I) = SVector(d.Mx[I], d.My[I], d.E[I])
@inline function add_conserved!(d, I, s, U)
    d.Mx[I] += s * U[1]; d.My[I] += s * U[2]; d.E[I] += s * U[3]; return d
end

# ── flat (Local / MPI) RHS ──
function rhs_flat!(du, u, dx, dy)
    fill!(du, 0.0)
    synchronize_halo!(u)
    r = FaceRanges(u)
    accumulate_flux_divergence!(parent(du), parent(u), r, 1, inv(dx), (a, b) -> rusanov(a, b, 1), conserved_cell, add_conserved!)
    accumulate_flux_divergence!(parent(du), parent(u), r, 2, inv(dy), (a, b) -> rusanov(a, b, 2), conserved_cell, add_conserved!)
    return du
end

# ── threaded RHS (per-tile flux divergence) ──
function rhs_threaded!(du, u, dx, dy)
    fill!(du, 0.0)
    synchronize_halo!(u)
    r = FaceRanges(u)
    tile_foreach(thread_backend(u.Mx), tid -> begin
        dd = tile_parent(du, tid); ud = tile_parent(u, tid)
        accumulate_flux_divergence!(dd, ud, r, 1, inv(dx), (a, b) -> rusanov(a, b, 1), conserved_cell, add_conserved!)
        accumulate_flux_divergence!(dd, ud, r, 2, inv(dy), (a, b) -> rusanov(a, b, 2), conserved_cell, add_conserved!)
    end, 1:tile_count(u); scheduler=:static)
    return du
end

function step!(rhs!, u, u1, du, dx, dy, dt)
    rhs!(du, u, dx, dy);  @. u1 = u + dt * du
    rhs!(du, u1, dx, dy); @. u = 0.5 * u + 0.5 * (u1 + dt * du)
    return u
end

function fill_ic!(u)
    nx, ny = global_size(u.E)
    fill_from_global_indices!(_ -> 0.0, u.Mx)
    fill_from_global_indices!(_ -> 0.0, u.My)
    fill_from_global_indices!(u.E) do I              # at-rest hot spot ⇒ E = 3p = 3aT⁴
        x = (I[1] - 0.5) / nx; y = (I[2] - 0.5) / ny
        T = 1.0 + 0.5 * exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)
        3.0 * A * T^4
    end
    synchronize_halo!(u)
    return u
end

const NSTEPS = 12
function time_step(rhs!, u, dx, dy, dt; reps=3, sync=false)
    u1 = similar(u); du = similar(u)
    step!(rhs!, u, u1, du, dx, dy, dt)               # warm up
    best = Inf
    for _ in 1:reps
        sync && MPI.Barrier(MPI.COMM_WORLD)
        t = @elapsed for _ in 1:NSTEPS
            step!(rhs!, u, u1, du, dx, dy, dt)
        end
        best = min(best, t)
    end
    return best / NSTEPS
end

bcs = (Mx=:periodic, My=:periodic, E=:periodic)

function main(; G=(1024, 1024))
    comm = MPI.COMM_WORLD
    P = MPI.Comm_size(comm); rank = MPI.Comm_rank(comm); T = Threads.nthreads()
    dx = 1.0 / G[1]; dy = 1.0 / G[2]; dt = 0.2 * min(dx, dy)

    if P > 1
        # ── MPI mode ──
        topo = CartesianTopology(comm, (0, 0); periodic=(true, true))
        all(G .% topo.dims .== 0) || error("grid $G not divisible by ranks $(topo.dims)")
        owned = G .÷ topo.dims
        u = MultiHaloArray(HaloArray, Float64, owned, 1, topo; boundary_conditions=bcs)
        fill_ic!(u)
        t_mpi = MPI.Allreduce(time_step(rhs_flat!, u, dx, dy, dt; sync=true), max, comm)
        if rank == 0
            us = LocalMultiHaloArray(Float64, G, 1; boundary_conditions=bcs); fill_ic!(us)
            t_ser = time_step(rhs_flat!, us, dx, dy, dt)
            @printf("MPI  ranks=%-2d dims=%-7s  serial=%7.3f ms  MPI=%7.3f ms  speedup=%.2fx  (grid %dx%d)\n",
                P, string(topo.dims), 1e3 * t_ser, 1e3 * t_mpi, t_ser / t_mpi, G...)
        end
    else
        # ── single process: serial baseline (+ threaded if -t>1) ──
        us = LocalMultiHaloArray(Float64, G, 1; boundary_conditions=bcs); fill_ic!(us)
        t_ser = time_step(rhs_flat!, us, dx, dy, dt)
        if T > 1
            G[2] % T == 0 || error("grid y=$(G[2]) not divisible by threads=$T")
            ut = ThreadedMultiHaloArray(Float64, (G[1], G[2] ÷ T), 1; dims=(1, T), boundary_conditions=bcs)
            fill_ic!(ut)
            t_thr = time_step(rhs_threaded!, ut, dx, dy, dt)
            @printf("THR  threads=%-2d           serial=%7.3f ms  threaded=%7.3f ms  speedup=%.2fx  (grid %dx%d)\n",
                T, 1e3 * t_ser, 1e3 * t_thr, t_ser / t_thr, G...)
        else
            @printf("SER  threads=1            serial=%7.3f ms/step  (grid %dx%d)\n", 1e3 * t_ser, G...)
        end
    end
end

main()
