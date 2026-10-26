module HaloArraysLinearSolveExt

using HaloArrays
using LinearSolve
using LinearAlgebra: dot, norm, mul!, ldiv!, UniformScaling

const SciMLBase = LinearSolve.SciMLBase
const _IdentityOp = LinearSolve.SciMLOperators.IdentityOperator

# ============================================================
# Coordinate-free solvers for halo arrays of any dimensionality
#
# LinearSolve's stock Krylov wrappers model the unknown as a flat `AbstractVector`,
# so they can't take an N-D halo array. These solvers (standard algorithms, Saad)
# touch the unknown only through similar/copy/broadcast/dot/norm/mul!, so they
# run on a halo array of any dimensionality — and stay MPI-collective, since
# HaloArrays defines dot/norm as global reductions. Each is wrapped as a
# LinearSolve algorithm: the workspace is built once in `init_cacheval` (cached
# in `cache.cacheval`) and reused by `solve!`, which reports a proper retcode.
# ============================================================

# Stopping threshold. `reltol` is relative to `b` measured in the SAME norm the
# loop monitors: with a left preconditioner that is ‖M⁻¹b‖ (BiCGStab, GMRES) or
# √⟨b, M⁻¹b⟩ (CG), never the plain ‖b‖ — otherwise a scaled `Pl` shifts the
# effective tolerance by ‖b‖/‖M⁻¹b‖. Each solver computes its own measure below.
@inline _threshold(bnorm, abstol, reltol) =
    max(abstol, reltol * max(bnorm, eps(typeof(bnorm))))

# ---- preconditioning helpers ----
# LinearSolve convention: `Pl` and `Pr` are applied with `ldiv!`, and the solved
# system is `Pl⁻¹ A Pr⁻¹ y = Pl⁻¹ b`, `x = Pr⁻¹ y`. The identity default (no
# preconditioner) is a no-op, so the unpreconditioned path stays byte-for-byte
# and reduction-for-reduction identical.
# "No preconditioner": LinearSolve's default `IdentityOperator`, or `I`.
@inline _is_identity_precond(::_IdentityOp) = true
@inline _is_identity_precond(J::UniformScaling) = isone(J.λ)
@inline _is_identity_precond(_) = false

# M⁻¹ v, for either side. Out of place the result is written into `z`, or is
# `v` itself (no copy) when M is LinearSolve's default identity — callers only
# read it. In place: v ← M⁻¹ v. (`I`/`c*I` go through HaloArrays' `ldiv!`.)
@inline _precond(z, M, v) = ldiv!(z, M, v)
@inline _precond(z, ::_IdentityOp, v) = v
@inline _precond!(M, v) = ldiv!(M, v)
@inline _precond!(::_IdentityOp, v) = v

# A right preconditioner would be silently dropped by CG, so reject it explicitly.
@inline _reject_right_precond(Pr) =
    _is_identity_precond(Pr) || throw(ArgumentError(
        "HaloCG applies a LEFT preconditioner (Pl) only (A Pr⁻¹ is not symmetric); " *
        "a right preconditioner Pr::$(typeof(Pr)) is not supported. Use HaloGMRES/HaloBiCGStab."))

# ---- Conjugate Gradient — symmetric positive-definite ----
struct CGWorkspace{T}
    Ap::T; r::T; p::T; z::T
    function CGWorkspace(b)
        Ap = similar(b)
        new{typeof(Ap)}(Ap, similar(b), similar(b), similar(b))
    end
end

# Preconditioned CG (M SPD). `rz = ⟨r, M⁻¹r⟩` is the monitored residual measure:
# with the identity preconditioner `z ≡ r`, so `rz = ⟨r,r⟩`, `sqrt(rz) = ‖r‖`,
# and the two per-iteration reductions (`⟨p,Ap⟩`, `⟨r,z⟩`) reduce to exactly the
# stock CG — no extra communication when unpreconditioned.
function _krylov!(x, A, b, w::CGWorkspace, Pl, Pr; abstol, reltol, maxiter)
    _reject_right_precond(Pr)
    Ap, r, p = w.Ap, w.r, w.p
    Mb = _precond(Ap, Pl, b)                        # M⁻¹ b        (Ap is scratch here)
    thr = _threshold(sqrt(real(dot(b, Mb))), abstol, reltol)   # ‖b‖ in the CG norm
    mul!(Ap, A, x); copyto!(r, b); r .-= Ap        # r = b - A*x
    z = _precond(w.z, Pl, r)                        # z = M⁻¹ r    (r itself if Pl = I)
    copyto!(p, z)
    rz = real(dot(r, z))
    res = sqrt(rz)
    res ≤ thr && return x, 0, res, true
    iters = 0
    for k in 1:maxiter
        iters = k
        mul!(Ap, A, p)
        α = rz / real(dot(p, Ap))
        x .+= α .* p
        r .-= α .* Ap
        z = _precond(w.z, Pl, r)
        rznew = real(dot(r, z))
        res = sqrt(rznew)
        res ≤ thr && return x, iters, res, true
        p .= z .+ (rznew / rz) .* p
        rz = rznew
    end
    return x, iters, res, res ≤ thr
end

# ---- BiCGStab — general square systems (short recurrence) ----
struct BiCGStabWorkspace{T}
    r::T; tmp::T; rhat::T; p::T; v::T; s::T; t::T; shat::T
    function BiCGStabWorkspace(b)
        r = similar(b)
        new{typeof(r)}(r, similar(b), similar(b), similar(b), similar(b), similar(b), similar(b), similar(b))
    end
end

# Preconditioned BiCGStab for Pl⁻¹ A Pr⁻¹: the directions p, s are right-
# preconditioned before each operator product (p̂ = Pr⁻¹p, ŝ = Pr⁻¹s) and the
# solution is updated with p̂, ŝ; the products are left-preconditioned in place,
# so all residual norms are the Pl-preconditioned residual (the true residual
# when Pl is the identity, whatever Pr is).
function _krylov!(x, A, b, w::BiCGStabWorkspace, Pl, Pr; abstol, reltol, maxiter)
    r, tmp, rhat, p, v, s, t, shat = w.r, w.tmp, w.rhat, w.p, w.v, w.s, w.t, w.shat
    thr = _threshold(norm(_precond(tmp, Pl, b)), abstol, reltol)   # ‖Pl⁻¹ b‖
    copyto!(r, b); mul!(tmp, A, x); r .-= tmp       # r = b - A*x
    _precond!(Pl, r)                          # r ← M⁻¹ r
    copyto!(rhat, r)
    fill!(p, 0); fill!(v, 0)
    Tr = real(eltype(b))
    ρ_old = α = ω = one(Tr)
    res = norm(r)
    res ≤ thr && return x, 0, res, true
    iters = 0
    for k in 1:maxiter
        iters = k
        ρ = dot(rhat, r)
        β = (ρ / ρ_old) * (α / ω)
        p .= r .+ β .* (p .- ω .* v)                # one pass, not three
        phat = _precond(tmp, Pr, p)                 # p̂ = Pr⁻¹ p   (p itself if Pr = I)
        mul!(v, A, phat); _precond!(Pl, v)          # v ← Pl⁻¹ A p̂
        α = ρ / dot(rhat, v)
        s .= r .- α .* v
        res = norm(s)
        if res ≤ thr
            x .+= α .* phat
            return x, iters, res, true
        end
        shat_ = _precond(shat, Pr, s)               # ŝ = Pr⁻¹ s
        mul!(t, A, shat_); _precond!(Pl, t)         # t ← Pl⁻¹ A ŝ
        ω = dot(t, s) / dot(t, t)
        x .+= α .* phat .+ ω .* shat_
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
    function GMRESWorkspace(b; restart)
        K = float(eltype(b)); T = real(K); m = restart   # Hessenberg/g/sn in the element field
        V = [similar(b) for _ in 1:m+1]; H = zeros(K, m + 1, m); g = zeros(K, m + 1)
        cs = zeros(T, m); w = similar(b)
        new{typeof(V),typeof(H),typeof(g),typeof(cs),typeof(w)}(V, H, g, cs, zeros(K, m), w, similar(b))
    end
end

# Givens rotation [c s; -conj(s) c] that zeros `b`; `c` is real. Correct for both
# real and complex scalars (so GMRES converges on complex systems too).
@inline function _givens(a, b)
    iszero(b) && return (one(real(a)), zero(a))
    iszero(a) && return (zero(real(a)), one(a))
    ρ = hypot(abs(a), abs(b))
    return (abs(a) / ρ, (a / abs(a)) * conj(b) / ρ)
end

# Preconditioned restarted GMRES: build the Krylov space of Pl⁻¹ A Pr⁻¹ and
# monitor the residual `Pl⁻¹(b - Ax)` (the true residual when Pl is the identity,
# whatever Pr is). The Arnoldi vectors span the right-preconditioned space, so
# the correction Σ gᵢ vᵢ is mapped back through Pr⁻¹ before updating x.
function _krylov!(x, A, b, work::GMRESWorkspace, Pl, Pr; abstol, reltol, maxiter)
    V, H, g, cs, sn, w, Ax = work.V, work.H, work.g, work.cs, work.sn, work.w, work.Ax
    T = real(float(eltype(b))); m = length(cs)
    thr = _threshold(norm(_precond(w, Pl, b)), abstol, reltol)     # ‖Pl⁻¹ b‖
    res = norm(b); total = 0
    while total < maxiter
        mul!(Ax, A, x)
        r = V[1]; copyto!(r, b); r .-= Ax           # residual in V[1]
        _precond!(Pl, r)                       # r ← M⁻¹(b - Ax)
        β = norm(r); res = β
        res ≤ thr && break
        V[1] .*= inv(β)
        fill!(g, 0); g[1] = β
        k = 0
        inner = min(m, maxiter - total)             # never exceed the iteration budget
        for j in 1:inner
            k = j; total += 1
            vj = _precond(Ax, Pr, V[j])             # Pr⁻¹ vⱼ      (Ax is scratch here)
            mul!(w, A, vj); _precond!(Pl, w)        # w ← Pl⁻¹ A Pr⁻¹ vⱼ
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
        fill!(w, 0)
        for i in 1:k
            w .+= g[i] .* V[i]                      # correction in the Pr-space
        end
        x .+= _precond(Ax, Pr, w)                   # x += Pr⁻¹ Σ gᵢ vᵢ
        res ≤ thr && break
    end
    return x, total, res, res ≤ thr
end

# ---- MINRES — symmetric / Hermitian (in)definite (Paige–Saunders) ----
# The Lanczos tridiagonal is real symmetric even for a Hermitian operator, so all
# scalars stay real; only the basis/solution vectors carry the element type.
struct MINRESWorkspace{T}
    v::T; y::T; r1::T; r2::T; w::T; w1::T; w2::T
    function MINRESWorkspace(b)
        v = similar(b)
        new{typeof(v)}(v, similar(b), similar(b), similar(b), similar(b), similar(b), similar(b))
    end
end

# Preconditioned MINRES needs the SPD Lanczos rework, so reject rather than
# silently ignore a supplied preconditioner (use HaloCG/HaloGMRES instead).
function _krylov!(x, A, b, work::MINRESWorkspace, Pl, Pr; abstol, reltol, maxiter)
    (_is_identity_precond(Pl) && _is_identity_precond(Pr)) || throw(ArgumentError(
        "HaloMINRES does not support preconditioning; use HaloCG (SPD) or HaloGMRES with `Pl`."))
    v, y, r1, r2, w, w1, w2 = work.v, work.y, work.r1, work.r2, work.w, work.w1, work.w2
    T = real(float(eltype(b)))
    mul!(y, A, x); r2 .= b .- y                 # r0 = b - A*x0  (r1, y are free buffers)
    β = norm(r2)
    thr = _threshold(norm(b), abstol, reltol)
    res = β
    β ≤ thr && return x, 0, res, true
    fill!(w, 0); fill!(w1, 0); fill!(w2, 0)
    oldb = zero(T); dbar = zero(T); epsln = zero(T); phibar = β
    cs = -one(T); sn = zero(T)
    iters = 0
    for k in 1:maxiter
        iters = k
        v .= r2 .* inv(β)                        # vₖ = yₖ / βₖ   (yₖ lives in r2)
        mul!(y, A, v)
        k ≥ 2 && (y .-= (β / oldb) .* r1)
        α = real(dot(v, y))                      # real for a Hermitian operator
        y .-= (α / β) .* r2
        r1, r2, y = r2, y, r1                    # shift the history by rebinding, no copies
        oldb = β; β = norm(r2)
        oldeps = epsln                           # apply previous rotation to the tridiagonal
        δ    = cs * dbar + sn * α
        gbar = sn * dbar - cs * α
        epsln = sn * β
        dbar  = -cs * β
        γ = max(hypot(gbar, β), eps(T))          # next plane rotation
        cs = gbar / γ; sn = β / γ
        φ = cs * phibar; phibar = sn * phibar
        w1, w2, w = w2, w, w1                    # solution direction history, rebinding
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

# LinearSolve rejects an `AbstractMatrix` right-hand side for Krylov subspace
# methods, reading extra columns as a batch of independent vectors. A 2-D halo
# array is one N-D field, not a batch, and these solvers are coordinate-free —
# they only ever touch `b` through broadcasts and `_hdot`. Opt out of the check
# for THESE algorithms only (not every `AbstractKrylovSubspaceMethod`: a broader
# method was ambiguous with LinearSolve's own `KrylovJL` method).
const _HaloAlg = Union{HaloCGAlg, HaloBiCGStabAlg, HaloMINRESAlg, HaloGMRESAlg}
LinearSolve._check_batched_rhs_support(::_HaloAlg, ::AbstractHaloArray) = nothing

# OrdinaryDiffEq updates the per-step tolerances through `update_tolerances!`,
# which sets `cache.abstol`/`cache.reltol` (read fresh by every `solve!` below)
# and then calls this hook; the generic fallback only warns "not implemented".
LinearSolve.update_tolerances_internal!(cache, ::_HaloAlg, abstol, reltol) = nothing

LinearSolve.init_cacheval(::HaloCGAlg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose, assump) =
    CGWorkspace(b)
LinearSolve.init_cacheval(::HaloBiCGStabAlg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose, assump) =
    BiCGStabWorkspace(b)
LinearSolve.init_cacheval(::HaloMINRESAlg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose, assump) =
    MINRESWorkspace(b)
LinearSolve.init_cacheval(alg::HaloGMRESAlg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose, assump) =
    GMRESWorkspace(b; restart = alg.restart)

_retcode(converged) = converged ? SciMLBase.ReturnCode.Success : SciMLBase.ReturnCode.MaxIters

# The workspace in `cacheval` names the algorithm: `_krylov!` dispatches on it.
function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::_HaloAlg; kwargs...)
    x, iters, _res, conv = _krylov!(cache.u, cache.A, cache.b, cache.cacheval, cache.Pl, cache.Pr;
        abstol = cache.abstol, reltol = cache.reltol, maxiter = cache.maxiters)
    return SciMLBase.build_linear_solution(alg, x, nothing, cache; retcode = _retcode(conv), iters = iters)
end

end # module
