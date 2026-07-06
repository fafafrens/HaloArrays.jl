# ============================================================
# HaloArrays.jl — fused-kernel CG: fewer barriers, fewer sweeps
#
# Run with:
#   julia --project=. -t 4 examples/poisson/cg_fused.jl
#
# The textbook CG in poisson/krylov_solvers.jl touches the unknown only
# through mul!, dot and broadcast. That is the right way to *write* a solver —
# it runs unchanged on every backend — but it costs six array sweeps per
# iteration, each of which is a separate parallel operation:
#
#   mul!(Ap,A,p)   dot(p,Ap)   x.+=αp   r.-=αAp   dot(r,r)   p.=r.+βp
#
# Each sweep is memory-bandwidth-bound, and on the threaded backend each one
# also pays a task spawn/join barrier. For thin BLAS-1 sweeps that fee never
# amortizes, so threaded CG can come out *slower* than serial.
#
# This example shows the standard cure — fuse operations that traverse the
# same data, so per-tile work gets fatter and the barrier count drops 6 → 3:
#
#   1. dot(p, Ap) is accumulated INSIDE the stencil sweep: p[I] and the fresh
#      Ap[I] are already in registers, so the inner product is free.
#   2. x .+= αp, r .-= αAp and dot(r,r) become ONE pass per tile; r never
#      makes an extra round trip through memory between update and reduction.
#
# Measured on an M2 laptop at 1024² (ms per iteration, best of 3 solves):
#
#                     textbook cg!   cg_fused!
#   Local, 4 threads       4.9          4.5
#   Threaded, 4 threads    4.6          3.5      (1.3-1.4x, reproducibly)
#
# Fused-threaded is the fastest configuration, and the gap widens whenever the
# OS schedules the plain version unluckily (up to 3x observed): fewer barriers
# also means fewer chances to wait on a straggling thread. Laptop threading
# timings swing with thread placement — the driver below reports the best of a
# few solves; treat single runs as smoke, real numbers belong on a cluster. The price is that the solver now knows about tiles
# (tile_count / tile_parent / interior_range) instead of staying coordinate-
# free — which is why the textbook version remains the reference.
#
# The fused reductions stay MPI-correct through one hook: `_sum_ranks`
# Allreduces the two scalar accumulations on the MPI backend and is the
# identity elsewhere (the textbook version gets this for free from the global
# dot/norm).
# ============================================================

using HaloArrays
using LinearAlgebra: LinearAlgebra, dot, norm, mul!
using OhMyThreads: tmapreduce
using MPI
using Printf

include("krylov_solvers.jl")        # the textbook cg! we compare against

# ------------------------------------------------------------
# Backend-generic scalar reduction across ranks
# ------------------------------------------------------------

_sum_ranks(_u, acc) = acc
_sum_ranks(u::HaloArray, acc) = MPI.Allreduce(acc, +, communicator(u))

# ------------------------------------------------------------
# The operator: -∇² with the p·Ap inner product fused in
# ------------------------------------------------------------
# Antireflecting (odd) ghosts ⇒ homogeneous Dirichlet ⇒ SPD, so CG applies.
# The tile loop covers every backend: a Local/MPI array is its own single
# tile, a ThreadedHaloArray runs one task per tile.

struct FusedLaplacian{U}
    tmp::U          # halo scratch: the input with refreshed ghosts
    inv_h2::Float64
end

"`Ap = A*p` and return `dot(p, Ap)` — one sweep, no extra traffic for the dot."
function mul_dot!(Ap, A::FusedLaplacian, p)
    copyto!(A.tmp, p)
    synchronize_halo!(A.tmp)
    u = A.tmp
    offs = unit_vector(u)
    inv_h2 = A.inv_h2
    acc = tmapreduce(+, 1:tile_count(u); scheduler = :static) do t
        data = tile_parent(u, t)
        out = tile_parent(Ap, t)
        pAp = 0.0
        @inbounds for I in CartesianIndices(interior_range(u, t))
            lap = zero(eltype(data))
            for d in 1:ndims(u)
                lap += 2 * data[I] - data[I + offs[d]] - data[I - offs[d]]
            end
            v = lap * inv_h2
            out[I] = v
            pAp += data[I] * v
        end
        pAp
    end
    return _sum_ranks(p, acc)
end

# The plain mul! (for the textbook cg! baseline) reuses the fused sweep.
LinearAlgebra.mul!(Ap, A::FusedLaplacian, p) = (mul_dot!(Ap, A, p); Ap)

# ------------------------------------------------------------
# x .+= a.*p ; r .-= a.*Ap ; return dot(r, r)   — one pass per tile
# ------------------------------------------------------------

function fused_update!(x, r, p, Ap, a)
    acc = tmapreduce(+, 1:tile_count(x); scheduler = :static) do t
        xd = tile_parent(x, t)
        rd = tile_parent(r, t)
        pd = tile_parent(p, t)
        ad = tile_parent(Ap, t)
        rr = 0.0
        @inbounds for I in CartesianIndices(interior_range(x, t))
            xd[I] += a * pd[I]
            rn = rd[I] - a * ad[I]
            rd[I] = rn
            rr += rn * rn
        end
        rr
    end
    return _sum_ranks(x, acc)
end

# ------------------------------------------------------------
# CG with 3 parallel operations per iteration instead of 6
# ------------------------------------------------------------

function cg_fused!(x, A, b; tol = 1e-10, maxiter = length(b))
    Ap = similar(b)
    mul!(Ap, A, x)
    r = copy(b); r .-= Ap
    p = copy(r)
    rsold = real(dot(r, r))
    bnorm = max(norm(b), eps(float(real(eltype(b)))))
    res = sqrt(rsold); iters = 0
    for k in 1:maxiter
        iters = k
        pAp = mul_dot!(Ap, A, p)                 # sweep 1: Ap and p·Ap together
        a = rsold / pAp
        rsnew = fused_update!(x, r, p, Ap, a)    # sweep 2: x, r and r·r together
        res = sqrt(rsnew)
        res <= tol * bnorm && break
        p .= r .+ (rsnew / rsold) .* p           # sweep 3
        rsold = rsnew
    end
    return x, iters, res
end

# ------------------------------------------------------------
# Driver: same problem through both solvers, on Local and Threaded
# ------------------------------------------------------------

function _fill_rhs!(b)
    n = global_size(b)
    fill_from_global_indices!(b) do I
        x = I[1] / (n[1] + 1)
        y = I[2] / (n[2] + 1)
        return 2 * (x * (1 - x) + y * (1 - y))
    end
    return b
end

# Best of `reps` solves per solver: thread placement on a laptop varies a lot
# from run to run, and the minimum is the standard de-noised statistic.
function run_case(name, make; n, maxiter, reps = 3)
    b = _fill_rhs!(make())
    A = FusedLaplacian(make(), Float64((n + 1)^2))

    x1 = make(); it1 = 0; res1 = 0.0
    t1 = minimum(1:reps) do _
        fill!(x1, 0.0)
        @elapsed (_, it1, res1) = cg!(x1, A, b; tol = 1e-12, maxiter)
    end

    x2 = make(); it2 = 0; res2 = 0.0
    t2 = minimum(1:reps) do _
        fill!(x2, 0.0)
        @elapsed (_, it2, res2) = cg_fused!(x2, A, b; tol = 1e-12, maxiter)
    end

    dev = norm(x1 .- x2) / norm(x1)
    agree = it1 == it2 && dev < 1e-6
    @printf("  %-9s textbook %6.2f ms/iter | fused %6.2f ms/iter | %4.2fx | ‖Δx‖/‖x‖ = %.1e %s\n",
        name, 1e3 * t1 / it1, 1e3 * t2 / it2, t1 / t2, dev, agree ? "✓" : "✗ DISAGREE")
    agree || error("fused CG disagrees with the textbook CG")
    return nothing
end

# n = 1024 keeps the problem out of cache — that is where both effects live
# (in-cache, e.g. 512², the sweeps are compute-fast and fusion barely shows).
function main(; n = 1024, maxiter = 200)
    println("Fused vs textbook CG — Poisson $(n)x$(n), $(maxiter) iterations, ",
        Threads.nthreads(), " thread(s)")
    run_case("Local", () -> LocalHaloArray(Float64, (n, n), 1;
            boundary_condition = :antireflecting); n, maxiter)
    if Threads.nthreads() > 1 && n % Threads.nthreads() == 0
        nt = Threads.nthreads()
        run_case("Threaded", () -> ThreadedHaloArray(Float64, (n, n ÷ nt), 1;
                dims = (1, nt), boundary_condition = :antireflecting); n, maxiter)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
