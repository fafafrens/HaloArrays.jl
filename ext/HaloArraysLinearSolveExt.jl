module HaloArraysLinearSolveExt

using HaloArrays
using LinearSolve
using Krylov
using LinearAlgebra: dot, norm, mul!

const SciMLBase = LinearSolve.SciMLBase

# ============================================================
# Path 1 — KrylovJL on a *1-D* halo array (Krylov.jl, mature, cached)
#
# `HaloKrylov(:method)` is a thin KrylovJL alias; the init_cacheval override
# below builds Krylov's workspace via KrylovConstructor (similar-based) instead
# of `S(undef, n)`, so the stock `KrylovJL_*` work on a halo array too — and are
# cached by LinearSolve's solve!. Krylov.jl requires `b::AbstractVector`, so this
# path only applies to 1-D halo-array states (an N-D halo array is not a vector).
# For N-D states use the coordinate-free solvers in Path 2.
# ============================================================

const _HALOKRYLOV_METHODS = (:gmres, :cg, :bicgstab, :minres, :dqgmres, :diom,
                             :fom, :cgs, :minares, :minres_qlp, :symmlq)

function HaloArrays.HaloKrylov(method::Symbol; kwargs...)
    method in _HALOKRYLOV_METHODS || throw(ArgumentError(string(
        "HaloKrylov: unsupported method :", method, ". Supported (matrix-free, ",
        "cached via LinearSolve's KrylovJL): ", join(_HALOKRYLOV_METHODS, ", "), ". ",
        ":fgmres/:qmr/:bilq allocate via S(undef, n) mid-solve; :car/:cg_lanczos ",
        "are not wrapped by LinearSolve's KrylovJL.")))
    return KrylovJL(; KrylovAlg = getproperty(Krylov, Symbol(method, :!)), kwargs...)
end

function LinearSolve.init_cacheval(alg::KrylovJL, A, b::AbstractHaloArray, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearSolve.LinearVerbosity, Bool},
        assumptions::LinearSolve.OperatorAssumptions; zeroinit = true)
    KS = LinearSolve.get_KrylovJL_solver(alg.KrylovAlg)
    kc = Krylov.KrylovConstructor(b)
    memory_methods = (Krylov.dqgmres!, Krylov.diom!, Krylov.gmres!, Krylov.fgmres!,
                      Krylov.gpmr!, Krylov.fom!)
    solver = if zeroinit
        alg.KrylovAlg in memory_methods ? KS(kc; memory = 1) : KS(kc)
    else
        knt = NamedTuple(alg.kwargs)
        memory = haskey(knt, :memory) ? knt[:memory] :
                 (alg.gmres_restart == 0 ? min(20, size(A, 1)) : alg.gmres_restart)
        if alg.KrylovAlg in memory_methods
            KS(kc; memory)
        elseif alg.KrylovAlg in (Krylov.minres!, Krylov.symmlq!, Krylov.lslq!,
                                 Krylov.lsqr!, Krylov.lsmr!)
            alg.window != 0 ? KS(kc; window = alg.window) : KS(kc)
        else
            KS(kc)
        end
    end
    solver.x = u
    return solver
end

# ============================================================
# Path 2 — coordinate-free solvers for *N-D* halo arrays
#
# Krylov.jl / SimpleGMRES model the unknown as a flat `AbstractVector`, so they
# can't take an N-D halo array. These solvers (standard algorithms, Saad)
# touch the unknown only through similar/copy/broadcast/dot/norm/mul!, so they
# run on a halo array of any dimensionality — and stay MPI-collective, since
# HaloArrays defines dot/norm as global reductions. Each is wrapped as a
# LinearSolve algorithm: the workspace is built once in `init_cacheval` (cached
# in `cache.cacheval`) and reused by `solve!`, which reports a proper retcode.
# ============================================================

@inline _threshold(b, abstol, reltol) =
    max(abstol, reltol * max(norm(b), eps(float(real(eltype(b))))))

# ---- Conjugate Gradient — symmetric positive-definite ----
struct CGWorkspace{T}
    Ap::T
    r::T
    p::T
end
_cg_workspace(b) = CGWorkspace(similar(b), similar(b), similar(b))

function _cg!(x, A, b, w::CGWorkspace; abstol, reltol, maxiter)
    Ap, r, p = w.Ap, w.r, w.p
    mul!(Ap, A, x); copyto!(r, b); r .-= Ap        # r = b - A*x
    copyto!(p, r)
    rsold = real(dot(r, r))
    thr = _threshold(b, abstol, reltol)
    res = sqrt(rsold)
    res ≤ thr && return x, 0, res, true
    iters = 0
    for k in 1:maxiter
        iters = k
        mul!(Ap, A, p)
        α = rsold / real(dot(p, Ap))
        x .+= α .* p
        r .-= α .* Ap
        rsnew = real(dot(r, r))
        res = sqrt(rsnew)
        res ≤ thr && return x, iters, res, true
        p .= r .+ (rsnew / rsold) .* p
        rsold = rsnew
    end
    return x, iters, res, res ≤ thr
end

# ---- BiCGStab — general square systems (short recurrence) ----
struct BiCGStabWorkspace{T}
    r::T; tmp::T; rhat::T; p::T; v::T; s::T; t::T
end
_bicgstab_workspace(b) =
    BiCGStabWorkspace(similar(b), similar(b), similar(b), similar(b), similar(b), similar(b), similar(b))

function _bicgstab!(x, A, b, w::BiCGStabWorkspace; abstol, reltol, maxiter)
    r, tmp, rhat, p, v, s, t = w.r, w.tmp, w.rhat, w.p, w.v, w.s, w.t
    copyto!(r, b); mul!(tmp, A, x); r .-= tmp       # r = b - A*x
    copyto!(rhat, r)
    fill!(p, 0); fill!(v, 0)
    Tr = real(eltype(b))
    ρ_old = α = ω = one(Tr)
    thr = _threshold(b, abstol, reltol)
    res = norm(r)
    res ≤ thr && return x, 0, res, true
    iters = 0
    for k in 1:maxiter
        iters = k
        ρ = dot(rhat, r)
        β = (ρ / ρ_old) * (α / ω)
        p .-= ω .* v; p .*= β; p .+= r              # p = r + β(p - ω v)
        mul!(v, A, p)
        α = ρ / dot(rhat, v)
        s .= r .- α .* v
        res = norm(s)
        if res ≤ thr
            x .+= α .* p
            return x, iters, res, true
        end
        mul!(t, A, s)
        ω = dot(t, s) / dot(t, t)
        x .+= α .* p; x .+= ω .* s
        r .= s .- ω .* t
        res = norm(r)
        res ≤ thr && return x, iters, res, true
        ρ_old = ρ
    end
    return x, iters, res, res ≤ thr
end

# ---- GMRES with restart — general square systems ----
struct GMRESWorkspace{TV,TH,TG,TC,TW}
    V::TV; H::TH; g::TG; cs::TC; sn::TG; w::TW; Ax::TW
end
function _gmres_workspace(b; restart)
    K = float(eltype(b)); T = real(K); m = restart   # Hessenberg/g/sn in the element field
    GMRESWorkspace([similar(b) for _ in 1:m+1], zeros(K, m + 1, m), zeros(K, m + 1),
                   zeros(T, m), zeros(K, m), similar(b), similar(b))
end

# Givens rotation [c s; -conj(s) c] that zeros `b`; `c` is real. Correct for both
# real and complex scalars (so GMRES converges on complex systems too).
@inline function _givens(a, b)
    iszero(b) && return (one(real(a)), zero(a))
    iszero(a) && return (zero(real(a)), one(a))
    ρ = hypot(abs(a), abs(b))
    return (abs(a) / ρ, (a / abs(a)) * conj(b) / ρ)
end

function _gmres!(x, A, b, work::GMRESWorkspace; abstol, reltol, maxiter)
    V, H, g, cs, sn, w, Ax = work.V, work.H, work.g, work.cs, work.sn, work.w, work.Ax
    T = real(float(eltype(b))); m = length(cs)
    thr = _threshold(b, abstol, reltol)
    res = norm(b); total = 0
    while total < maxiter
        mul!(Ax, A, x)
        r = V[1]; copyto!(r, b); r .-= Ax           # residual in V[1]
        β = norm(r); res = β
        res ≤ thr && break
        V[1] .*= inv(β)
        fill!(g, 0); g[1] = β
        k = 0
        inner = min(m, maxiter - total)             # never exceed the iteration budget
        for j in 1:inner
            k = j; total += 1
            mul!(w, A, V[j])
            for i in 1:j                            # modified Gram-Schmidt
                H[i, j] = dot(V[i], w)              # conjugate inner product (complex-safe)
                w .-= H[i, j] .* V[i]
            end
            hnext = norm(w)                         # real; H is the (possibly complex) field
            H[j+1, j] = hnext
            happy = hnext ≤ eps(T)                  # the next Arnoldi vector can't be formed
            happy || (V[j+1] .= w .* inv(hnext))
            for i in 1:j-1                          # apply previous Givens rotations
                τ         =  cs[i] * H[i, j] + sn[i] * H[i+1, j]
                H[i+1, j] = -conj(sn[i]) * H[i, j] + cs[i] * H[i+1, j]
                H[i, j]   = τ
            end
            cs[j], sn[j] = _givens(H[j, j], H[j+1, j])
            H[j, j]   = cs[j] * H[j, j] + sn[j] * H[j+1, j]
            H[j+1, j] = 0
            g[j+1] = -conj(sn[j]) * g[j]
            g[j]   =  cs[j] * g[j]
            res = abs(g[j+1])
            (res ≤ thr || happy) && break           # stop on convergence or happy breakdown
        end
        # back-substitution H[1:k,1:k] y = g[1:k], solved in place into g (no allocation)
        for i in k:-1:1
            for l in i+1:k
                g[i] -= H[i, l] * g[l]
            end
            g[i] /= H[i, i]
        end
        for i in 1:k
            x .+= g[i] .* V[i]
        end
        res ≤ thr && break
    end
    return x, total, res, res ≤ thr
end

# ---- MINRES — symmetric / Hermitian (in)definite (Paige–Saunders) ----
# The Lanczos tridiagonal is real symmetric even for a Hermitian operator, so all
# scalars stay real; only the basis/solution vectors carry the element type.
struct MINRESWorkspace{T}
    v::T; y::T; r1::T; r2::T; w::T; w1::T; w2::T
end
_minres_workspace(b) =
    MINRESWorkspace(similar(b), similar(b), similar(b), similar(b), similar(b), similar(b), similar(b))

function _minres!(x, A, b, work::MINRESWorkspace; abstol, reltol, maxiter)
    v, y, r1, r2, w, w1, w2 = work.v, work.y, work.r1, work.r2, work.w, work.w1, work.w2
    T = real(float(eltype(b)))
    mul!(y, A, x); r1 .= b .- y                 # r0 = b - A*x0
    β = norm(r1)
    thr = _threshold(b, abstol, reltol)
    res = β
    β ≤ thr && return x, 0, res, true
    copyto!(y, r1); copyto!(r2, r1)
    fill!(w, 0); fill!(w1, 0); fill!(w2, 0)
    oldb = zero(T); dbar = zero(T); epsln = zero(T); phibar = β
    cs = -one(T); sn = zero(T)
    iters = 0
    for k in 1:maxiter
        iters = k
        v .= y .* inv(β)                         # vₖ = yₖ / βₖ
        mul!(y, A, v)
        k ≥ 2 && (y .-= (β / oldb) .* r1)
        α = real(dot(v, y))                      # real for a Hermitian operator
        y .-= (α / β) .* r2
        copyto!(r1, r2); copyto!(r2, y)
        oldb = β; β = norm(r2)
        oldeps = epsln                           # apply previous rotation to the tridiagonal
        δ    = cs * dbar + sn * α
        gbar = sn * dbar - cs * α
        epsln = sn * β
        dbar  = -cs * β
        γ = max(hypot(gbar, β), eps(T))          # next plane rotation
        cs = gbar / γ; sn = β / γ
        φ = cs * phibar; phibar = sn * phibar
        copyto!(w1, w2); copyto!(w2, w)          # solution direction + update
        w .= (v .- oldeps .* w1 .- δ .* w2) .* inv(γ)
        x .+= φ .* w
        res = phibar
        res ≤ thr && return x, iters, res, true
    end
    return x, iters, res, res ≤ thr
end

# ---- LinearSolve algorithm wiring ------------------------------------------
# The exported entry points `HaloCG`/`HaloBiCGStab`/`HaloGMRES` (declared in the
# core package) return these algorithm singletons.
struct HaloCGAlg       <: LinearSolve.AbstractKrylovSubspaceMethod end
struct HaloBiCGStabAlg <: LinearSolve.AbstractKrylovSubspaceMethod end
struct HaloMINRESAlg   <: LinearSolve.AbstractKrylovSubspaceMethod end
struct HaloGMRESAlg    <: LinearSolve.AbstractKrylovSubspaceMethod
    restart::Int
end

HaloArrays.HaloCG()       = HaloCGAlg()
HaloArrays.HaloBiCGStab() = HaloBiCGStabAlg()
HaloArrays.HaloMINRES()   = HaloMINRESAlg()
function HaloArrays.HaloGMRES(; restart = 30)
    restart ≥ 1 || throw(ArgumentError("HaloGMRES: `restart` must be ≥ 1, got $restart"))
    return HaloGMRESAlg(restart)
end

LinearSolve.init_cacheval(::HaloCGAlg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose, assump) =
    _cg_workspace(b)
LinearSolve.init_cacheval(::HaloBiCGStabAlg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose, assump) =
    _bicgstab_workspace(b)
LinearSolve.init_cacheval(::HaloMINRESAlg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose, assump) =
    _minres_workspace(b)
LinearSolve.init_cacheval(alg::HaloGMRESAlg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose, assump) =
    _gmres_workspace(b; restart = alg.restart)

_retcode(converged) = converged ? SciMLBase.ReturnCode.Success : SciMLBase.ReturnCode.MaxIters

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::HaloCGAlg; kwargs...)
    x, iters, _res, conv = _cg!(cache.u, cache.A, cache.b, cache.cacheval;
        abstol = cache.abstol, reltol = cache.reltol, maxiter = cache.maxiters)
    return SciMLBase.build_linear_solution(alg, x, nothing, cache; retcode = _retcode(conv), iters = iters)
end

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::HaloMINRESAlg; kwargs...)
    x, iters, _res, conv = _minres!(cache.u, cache.A, cache.b, cache.cacheval;
        abstol = cache.abstol, reltol = cache.reltol, maxiter = cache.maxiters)
    return SciMLBase.build_linear_solution(alg, x, nothing, cache; retcode = _retcode(conv), iters = iters)
end

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::HaloBiCGStabAlg; kwargs...)
    x, iters, _res, conv = _bicgstab!(cache.u, cache.A, cache.b, cache.cacheval;
        abstol = cache.abstol, reltol = cache.reltol, maxiter = cache.maxiters)
    return SciMLBase.build_linear_solution(alg, x, nothing, cache; retcode = _retcode(conv), iters = iters)
end

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::HaloGMRESAlg; kwargs...)
    x, iters, _res, conv = _gmres!(cache.u, cache.A, cache.b, cache.cacheval;
        abstol = cache.abstol, reltol = cache.reltol, maxiter = cache.maxiters)
    return SciMLBase.build_linear_solution(alg, x, nothing, cache; retcode = _retcode(conv), iters = iters)
end

end # module
