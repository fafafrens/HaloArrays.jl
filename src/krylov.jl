"""
    HaloArrays.Krylov

Matrix-free Krylov solvers that touch the unknown only through `similar`,
`copyto!`, broadcasting, `dot`, `norm`, `mul!` and `ldiv!`. They therefore run
on a halo array of any dimensionality, on a field collection, or on a plain
`Vector`, and stay correct under MPI, where `dot`/`norm` on halo arrays are
global reductions and everything else is local.

    cg!(x, A, b, ws::CGWorkspace;             Pl = nothing,               abstol, reltol, maxiter)
    minres!(x, A, b, ws::MINRESWorkspace;                                 abstol, reltol, maxiter)
    bicgstab!(x, A, b, ws::BiCGStabWorkspace; Pl = nothing, Pr = nothing, abstol, reltol, maxiter)
    gmres!(x, A, b, ws::GMRESWorkspace;       Pl = nothing, Pr = nothing, orth = :auto, abstol, reltol, maxiter)
    fgmres!(x, A, b, ws::FGMRESWorkspace;     Pl = nothing, Pr = nothing, orth = :auto, abstol, reltol, maxiter)

`x` holds the initial guess and is overwritten with the solution. `A` is
anything with `mul!(y, A, v)`. A preconditioner is anything with `ldiv!(z, M, v)`
and `ldiv!(M, v)`; `nothing` means none, and the solved system is
`Pl⁻¹ A Pr⁻¹ y = Pl⁻¹ b`, `x = Pr⁻¹ y`. The workspace is allocated once from `b`
and reused across solves. Each solver returns a [`Result`](@ref); it stops when
its residual measure falls below `max(abstol, reltol * ‖b‖)`, with `‖b‖` taken in
the same norm it monitors.

With LinearSolve loaded the same kernels are the `HaloCG()`, `HaloMINRES()`,
`HaloBiCGStab()`, `HaloGMRES()` and `HaloFGMRES()` algorithms.
"""
module Krylov

using LinearAlgebra: dot, norm, mul!, ldiv!
using ..HaloArrays: AbstractSingleHaloArray, AbstractHaloCollection, communicator, dots!

export Result,
       CGWorkspace, MINRESWorkspace, BiCGStabWorkspace, GMRESWorkspace, FGMRESWorkspace,
       cg!, minres!, bicgstab!, gmres!, fgmres!

"""
    Result(iters, residual, converged)

Outcome of a Krylov solve: iterations taken, the solver's residual measure at
exit (the norm its stopping test monitors, see each solver), and whether that
measure fell below the threshold.
"""
struct Result{T<:Real}
    iters::Int
    residual::T
    converged::Bool
end

# ---- preconditioners: `nothing` is none ---------------------------------------
# M⁻¹ v into `z`, or `v` itself when there is no preconditioner (callers only read it).
precond(z, M, v) = ldiv!(z, M, v)
precond(z, ::Nothing, v) = v
# v ← M⁻¹ v
precond!(M, v) = ldiv!(M, v)
precond!(::Nothing, v) = v

# ---- stopping test and residual -----------------------------------------------
# `reltol` is relative to `b` measured in the SAME norm the solver monitors; with
# a left preconditioner that is ‖Pl⁻¹b‖, not ‖b‖, or a scaled Pl would shift
# the effective tolerance by ‖b‖/‖Pl⁻¹b‖. Each solver passes its own measure.
threshold(bnorm, abstol, reltol) = max(abstol, reltol * max(bnorm, eps(typeof(bnorm))))
converged(res, thr) = res ≤ thr

# r = b - A x, with `tmp` for the product.
function residual!(r, A, x, b, tmp)
    mul!(tmp, A, x)
    copyto!(r, b)
    r .-= tmp
    return r
end

# ============================================================================
# Conjugate gradient — symmetric positive-definite
# ============================================================================
"""
    CGWorkspace(b)

Four vectors shaped like `b`, for [`cg!`](@ref).
"""
struct CGWorkspace{T}
    r::T; z::T; p::T; Ap::T
    function CGWorkspace(b)
        r = similar(b)
        new{typeof(r)}(r, similar(b), similar(b), similar(b))
    end
end

"""
    cg!(x, A, b, ws::CGWorkspace; Pl = nothing, abstol, reltol, maxiter) -> Result

Preconditioned conjugate gradient (Saad, *Iterative Methods for Sparse Linear
Systems*, Alg. 9.1) for `A` and `Pl` symmetric/Hermitian and definite — both
positive, or both negative (OrdinaryDiffEq's `W = J - M/γ` is negative-definite
for a diffusion operator; CG is the same iteration on `-A x = -b`). Left
preconditioning in the `Pl`-inner product; the monitored residual is
`√|⟨r, Pl⁻¹r⟩|`, which is `‖b - Ax‖` without a preconditioner. A breakdown
(`⟨p, Ap⟩ = 0`, i.e. an indefinite `A`) returns the last iterate, not converged.
One operator product, two inner products and three vector updates per iteration.
"""
function cg!(x, A, b, ws::CGWorkspace; Pl = nothing, abstol, reltol, maxiter)
    (; r, p, Ap) = ws
    Mb  = precond(ws.z, Pl, b)                        # ‖b‖ in the monitored norm
    thr = threshold(sqrt(abs(real(dot(b, Mb)))), abstol, reltol)

    residual!(r, A, x, b, Ap)                         # r  = b - A x
    z = precond(ws.z, Pl, r)                          # z  = M⁻¹ r
    copyto!(p, z)                                     # p  = z
    rz  = real(dot(r, z))
    res = sqrt(abs(rz))
    converged(res, thr) && return Result(0, res, true)

    for k in 1:maxiter
        mul!(Ap, A, p)                                # Ap = A p
        α = rz / real(dot(p, Ap))
        isfinite(α) || return Result(k - 1, res, false)   # breakdown: ⟨p, Ap⟩ = 0 (A indefinite)
        x .+= α .* p                                  # x += α p
        r .-= α .* Ap                                 # r -= α Ap
        z = precond(ws.z, Pl, r)                      # z  = M⁻¹ r
        rz, rz_old = real(dot(r, z)), rz
        res = sqrt(abs(rz))
        converged(res, thr) && return Result(k, res, true)
        p .= z .+ (rz / rz_old) .* p                  # p  = z + β p
    end
    return Result(maxiter, res, false)
end

# ============================================================================
# MINRES — symmetric / Hermitian, possibly indefinite (Paige & Saunders 1975)
# ============================================================================
"""
    MINRESWorkspace(b)

Seven vectors shaped like `b`, for [`minres!`](@ref).
"""
struct MINRESWorkspace{T}
    v::T; y::T; r1::T; r2::T; w::T; w1::T; w2::T
    function MINRESWorkspace(b)
        v = similar(b)
        new{typeof(v)}(v, similar(b), similar(b), similar(b), similar(b), similar(b), similar(b))
    end
end

"""
    minres!(x, A, b, ws::MINRESWorkspace; abstol, reltol, maxiter) -> Result

MINRES (Paige & Saunders 1975) for symmetric/Hermitian `A`, indefinite allowed.
Lanczos three-term recurrence plus a QR factorisation of the tridiagonal by
Givens rotations, so the residual norm `‖b - Ax‖` is known without forming the
residual. No preconditioning: that needs an SPD preconditioner and the
preconditioned Lanczos process. One operator product, one inner product and
one norm per iteration.
"""
function minres!(x, A, b, ws::MINRESWorkspace; abstol, reltol, maxiter)
    (; v, y, r1, r2, w, w1, w2) = ws
    T   = real(float(eltype(b)))
    thr = threshold(norm(b), abstol, reltol)

    residual!(r2, A, x, b, y)                         # r₀ = b - A x, kept in r2
    β   = norm(r2)
    res = β
    converged(res, thr) && return Result(0, res, true)
    fill!(w, 0); fill!(w1, 0); fill!(w2, 0)
    β_old = zero(T); d̄ = zero(T); ε = zero(T); φ̄ = β
    c = -one(T); s = zero(T)

    for k in 1:maxiter
        # Lanczos: vₖ = r2 / βₖ;  yₖ₊₁ = A vₖ - αₖ vₖ - βₖ vₖ₋₁  (vₖ₋₁ ∝ r1)
        v .= r2 .* inv(β)
        mul!(y, A, v)
        k ≥ 2 && (y .-= (β / β_old) .* r1)
        α = real(dot(v, y))
        y .-= (α / β) .* r2
        r1, r2, y = r2, y, r1                         # shift the history by rebinding, no copies
        β_old, β = β, norm(r2)
        # QR of the tridiagonal: apply the previous rotation, form the next one
        ε_old = ε
        δ  = c * d̄ + s * α
        γ̄  = s * d̄ - c * α
        ε  = s * β
        d̄  = -c * β
        γ  = max(hypot(γ̄, β), eps(T))
        c, s = γ̄ / γ, β / γ
        φ, φ̄ = c * φ̄, s * φ̄
        # new solution direction wₖ and the update along it
        w1, w2, w = w2, w, w1
        w .= (v .- ε_old .* w1 .- δ .* w2) .* inv(γ)
        x .+= φ .* w
        res = φ̄                                       # = ‖b - A x‖
        converged(res, thr) && return Result(k, res, true)
    end
    return Result(maxiter, res, false)
end

# ============================================================================
# BiCGStab — general square systems (van der Vorst 1992; Saad Alg. 7.7)
# ============================================================================
"""
    BiCGStabWorkspace(b)

Eight vectors shaped like `b`, for [`bicgstab!`](@ref).
"""
struct BiCGStabWorkspace{T}
    r::T; rhat::T; p::T; v::T; s::T; t::T; phat::T; shat::T
    function BiCGStabWorkspace(b)
        r = similar(b)
        new{typeof(r)}(r, similar(b), similar(b), similar(b), similar(b), similar(b), similar(b), similar(b))
    end
end

"""
    bicgstab!(x, A, b, ws::BiCGStabWorkspace; Pl = nothing, Pr = nothing, abstol, reltol, maxiter) -> Result

BiCGStab (van der Vorst 1992; Saad Alg. 7.7) for general square `A`, with split
preconditioning `Pl⁻¹ A Pr⁻¹`. Directions are right-preconditioned before each
operator product (`p̂ = Pr⁻¹p`, `ŝ = Pr⁻¹s`) and the solution is updated with
them; products are left-preconditioned in place, so the monitored residual is
`Pl⁻¹(b - Ax)`, the true residual when `Pl` is `nothing`. Two operator products
and four inner products per iteration.
"""
function bicgstab!(x, A, b, ws::BiCGStabWorkspace; Pl = nothing, Pr = nothing, abstol, reltol, maxiter)
    (; r, rhat, p, v, s, t, phat, shat) = ws
    thr = threshold(norm(precond(phat, Pl, b)), abstol, reltol)    # ‖Pl⁻¹ b‖

    residual!(r, A, x, b, phat)                       # r  = b - A x
    precond!(Pl, r)                                   # r  = Pl⁻¹ r
    copyto!(rhat, r)                                  # r̂  = r₀  (shadow residual)
    fill!(p, 0); fill!(v, 0)
    Tr = real(eltype(b))
    ρ_old = α = ω = one(Tr)
    res = norm(r)
    converged(res, thr) && return Result(0, res, true)

    for k in 1:maxiter
        ρ = dot(rhat, r)
        iszero(ρ) && return Result(k - 1, res, false) # breakdown: ⟨r̂, r⟩ = 0
        β = (ρ / ρ_old) * (α / ω)
        p .= r .+ β .* (p .- ω .* v)                  # p  = r + β (p - ω v)
        p̂ = precond(phat, Pr, p)                      # p̂  = Pr⁻¹ p
        mul!(v, A, p̂); precond!(Pl, v)                # v  = Pl⁻¹ A p̂
        α = ρ / dot(rhat, v)
        isfinite(α) || return Result(k - 1, res, false)   # breakdown: ⟨r̂, v⟩ = 0
        s .= r .- α .* v                              # s  = r - α v
        res = norm(s)
        if converged(res, thr)
            x .+= α .* p̂
            return Result(k, res, true)
        end
        ŝ = precond(shat, Pr, s)                      # ŝ  = Pr⁻¹ s
        mul!(t, A, ŝ); precond!(Pl, t)                # t  = Pl⁻¹ A ŝ
        ω = dot(t, s) / dot(t, t)
        if !isfinite(ω)                               # breakdown: t = 0. `res` is ‖s‖, the
            x .+= α .* p̂                              # residual of the half-step x + α p̂:
            return Result(k, res, false)              # return that iterate, so x and res agree
        end
        x .+= α .* p̂ .+ ω .* ŝ                        # x += α p̂ + ω ŝ
        r .= s .- ω .* t                              # r  = s - ω t
        res = norm(r)
        converged(res, thr) && return Result(k, res, true)
        ρ_old = ρ
    end
    return Result(maxiter, res, false)
end

# ============================================================================
# GMRES and flexible GMRES — general square systems
# ============================================================================
"""
    GMRESWorkspace(b; restart)

`restart + 1` basis vectors shaped like `b`, two scratch vectors, and the
`(restart+1) × restart` Hessenberg matrix with its Givens rotations, for
[`gmres!`](@ref). Memory grows linearly with `restart`.
"""
struct GMRESWorkspace{TV,TH,TG,TC,TW}
    V::TV; H::TH; g::TG; cs::TC; sn::TG; h1::TG; h2::TG; w::TW; Ax::TW
    function GMRESWorkspace(b; restart)
        restart ≥ 1 || throw(ArgumentError("GMRESWorkspace: restart must be ≥ 1, got $restart"))
        K = float(eltype(b)); T = real(K); m = restart     # Hessenberg/g/sn in the element field
        V = [similar(b) for _ in 1:m+1]; H = zeros(K, m + 1, m); g = zeros(K, m + 1)
        cs = zeros(T, m); w = similar(b)
        new{typeof(V),typeof(H),typeof(g),typeof(cs),typeof(w)}(
            V, H, g, cs, zeros(K, m), zeros(K, m), zeros(K, m), w, similar(b))
    end
end

"""
    FGMRESWorkspace(b; restart)

As [`GMRESWorkspace`](@ref) plus the `restart` preconditioned directions `Z`,
for [`fgmres!`](@ref): twice the vectors of GMRES.
"""
struct FGMRESWorkspace{TV,TH,TG,TC,TW}
    V::TV; Z::TV; H::TH; g::TG; cs::TC; sn::TG; h1::TG; h2::TG; w::TW; Ax::TW
    function FGMRESWorkspace(b; restart)
        restart ≥ 1 || throw(ArgumentError("FGMRESWorkspace: restart must be ≥ 1, got $restart"))
        K = float(eltype(b)); T = real(K); m = restart
        V = [similar(b) for _ in 1:m+1]; Z = [similar(b) for _ in 1:m]
        H = zeros(K, m + 1, m); g = zeros(K, m + 1); cs = zeros(T, m); w = similar(b)
        new{typeof(V),typeof(H),typeof(g),typeof(cs),typeof(w)}(
            V, Z, H, g, cs, zeros(K, m), zeros(K, m), zeros(K, m), w, similar(b))
    end
end

# Givens rotation [c s; -conj(s) c] that zeros `b`; `c` is real. Valid for
# complex scalars too, so GMRES converges on complex systems.
function givens(a, b)
    iszero(b) && return (one(real(a)), zero(a))
    iszero(a) && return (zero(real(a)), one(a))
    ρ = hypot(abs(a), abs(b))
    return (abs(a) / ρ, (a / abs(a)) * conj(b) / ρ)
end

# ---- orthogonalisation of the new Arnoldi vector against V[1:j] ----
# `orth = :auto` picks `:cgs2` where `dots!` turns the j inner products of a pass
# into ONE collective — a distributed halo array or collection — and `:mgs`
# otherwise, where CGS2's second pass is pure cost.
const ORTH = (:auto, :mgs, :cgs2)
function resolve_orth(orth, b)
    orth in ORTH || throw(ArgumentError("orth must be one of $ORTH, got $(repr(orth))"))
    orth === :auto || return orth
    return fused_dots(b) ? :cgs2 : :mgs
end
fused_dots(b) = false
fused_dots(b::Union{AbstractSingleHaloArray,AbstractHaloCollection}) = communicator(b) !== nothing

# Modified Gram–Schmidt: j sequential inner products, each against the
# already-updated w — robust, but j global reductions per step on MPI.
function orthogonalize!(::Val{:mgs}, ws, w, j)
    (; V, H) = ws
    for i in 1:j
        H[i, j] = dot(V[i], w)                        # conjugate inner product
        w .-= H[i, j] .* V[i]
    end
    return w
end
# Classical Gram–Schmidt, twice ("twice is enough": Giraud, Langou, Rozložník
# 2005). The j inner products of a pass are independent, so `dots!` does them
# with ONE global reduction; the second pass restores orthogonality to working
# precision. Two reductions per step instead of j, at twice the local work.
function orthogonalize!(::Val{:cgs2}, ws, w, j)
    (; V, H, h1, h2) = ws
    dots!(h1, V, w, j)                                # h1 = Vᵀ w
    for i in 1:j
        w .-= h1[i] .* V[i]
    end
    dots!(h2, V, w, j)                                # h2 = Vᵀ w  (what the first pass left)
    for i in 1:j
        w .-= h2[i] .* V[i]
    end
    for i in 1:j
        H[i, j] = h1[i] + h2[i]
    end
    return w
end

# Arnoldi step j. On entry `w` is the new operator product; orthogonalise it
# against V[1:j], form V[j+1], and extend the Givens QR of H so that |g[j+1]|
# is the residual norm of the current least-squares solution.
# Returns (residual, happy_breakdown).
function arnoldi_step!(ws, w, j, orth::Val)
    (; V, H, g, cs, sn) = ws
    T = real(eltype(g))
    orthogonalize!(orth, ws, w, j)                    # H[1:j, j], w ⟂ V[1:j]
    hnext = norm(w)
    H[j+1, j] = hnext
    happy = hnext ≤ eps(T)                            # the next basis vector can't be formed
    happy || (V[j+1] .= w .* inv(hnext))
    for i in 1:j-1                                    # previous rotations on the new column
        τ         =  cs[i] * H[i, j] + sn[i] * H[i+1, j]
        H[i+1, j] = -conj(sn[i]) * H[i, j] + cs[i] * H[i+1, j]
        H[i, j]   = τ
    end
    cs[j], sn[j] = givens(H[j, j], H[j+1, j])         # new rotation, zeroing H[j+1, j]
    H[j, j]   = cs[j] * H[j, j] + sn[j] * H[j+1, j]
    H[j+1, j] = zero(eltype(H))
    g[j+1] = -conj(sn[j]) * g[j]
    g[j]   =  cs[j] * g[j]
    return abs(g[j+1]), happy
end

# H[1:k,1:k] y = g[1:k], solved in place into g (upper triangular after the rotations).
function hessenberg_solve!(H, g, k)
    for i in k:-1:1
        for l in i+1:k
            g[i] -= H[i, l] * g[l]
        end
        g[i] /= H[i, i]
    end
    return g
end

"""
    gmres!(x, A, b, ws::GMRESWorkspace; Pl = nothing, Pr = nothing, abstol, reltol, maxiter) -> Result

Restarted GMRES (Saad & Schultz 1986; Saad Alg. 6.11) for general square `A`:
Arnoldi with modified Gram–Schmidt on `Pl⁻¹ A Pr⁻¹`, Givens QR of the Hessenberg
matrix, a restart every `restart` steps with the true residual recomputed, and
`maxiter` honoured exactly. The monitored residual is `Pl⁻¹(b - Ax)`, the true
residual when `Pl` is `nothing`. The correction `Σ yᵢ vᵢ` is mapped back through
`Pr⁻¹` once per cycle, which assumes `Pr` is one fixed linear operator; for a
preconditioner that changes between steps use [`fgmres!`](@ref).

`orth` selects the Gram–Schmidt variant: `:mgs` (modified, `j` sequential inner
products at step `j`), `:cgs2` (classical, twice: two [`dots!`](@ref) calls, so
two global reductions per step instead of `j`, at twice the local work), or
`:auto` (`:cgs2` on a distributed `HaloArray`, `:mgs` otherwise).
"""
function gmres!(x, A, b, ws::GMRESWorkspace; Pl = nothing, Pr = nothing, orth = :auto, abstol, reltol, maxiter)
    (; V, H, g, w, Ax) = ws
    m   = length(ws.cs)
    o   = Val(resolve_orth(orth, b))
    res = norm(precond(w, Pl, b))                     # ‖Pl⁻¹ b‖, the monitored norm
    thr = threshold(res, abstol, reltol)
    total = 0

    while total < maxiter
        r = V[1]
        residual!(r, A, x, b, Ax); precond!(Pl, r)   # r  = Pl⁻¹(b - A x)
        β = norm(r); res = β
        converged(res, thr) && break
        r .*= inv(β)                                  # v₁ = r / β
        fill!(g, 0); g[1] = β                         # right-hand side β e₁
        k = 0
        for j in 1:min(m, maxiter - total)            # one cycle, within the budget
            k = j; total += 1
            zⱼ = precond(Ax, Pr, V[j])                # zⱼ = Pr⁻¹ vⱼ   (Ax is scratch here)
            mul!(w, A, zⱼ); precond!(Pl, w)           # w  = Pl⁻¹ A zⱼ
            res, happy = arnoldi_step!(ws, w, j, o)
            (converged(res, thr) || happy) && break
        end
        hessenberg_solve!(H, g, k)                    # y = argmin ‖β e₁ - H̃ y‖
        fill!(w, 0)
        for i in 1:k
            w .+= g[i] .* V[i]                        # Σ yᵢ vᵢ
        end
        x .+= precond(Ax, Pr, w)                      # x += Pr⁻¹ Σ yᵢ vᵢ
        converged(res, thr) && break
    end
    return Result(total, res, converged(res, thr))
end

# Without a right preconditioner every zⱼ is vⱼ: read the correction from V and
# never touch Z (no copies, by dispatch).
zbasis(ws::FGMRESWorkspace, ::Nothing) = ws.V
zbasis(ws::FGMRESWorkspace, Pr) = ws.Z

"""
    fgmres!(x, A, b, ws::FGMRESWorkspace; Pl = nothing, Pr = nothing, abstol, reltol, maxiter) -> Result

Flexible GMRES (Saad 1993): as [`gmres!`](@ref), but every preconditioned
direction `zⱼ = Prⱼ⁻¹ vⱼ` is kept and the cycle update is `x += Σ yᵢ zᵢ`, so `Pr`
may differ at every step — a nested iterative solve with a few iterations,
smoother sweeps, a multigrid cycle. `Pl` is applied as in GMRES and must be one
fixed linear operator, since it defines the norm being minimised; the monitored
residual is `Pl⁻¹(b - Ax)`, the true residual when `Pl` is `nothing`. `orth` as
in [`gmres!`](@ref).
"""
function fgmres!(x, A, b, ws::FGMRESWorkspace; Pl = nothing, Pr = nothing, orth = :auto, abstol, reltol, maxiter)
    (; V, Z, H, g, w, Ax) = ws
    m   = length(ws.cs)
    o   = Val(resolve_orth(orth, b))
    zs  = zbasis(ws, Pr)                              # the zⱼ: Z, or V itself without Pr
    res = norm(precond(w, Pl, b))                     # ‖Pl⁻¹ b‖, the monitored norm
    thr = threshold(res, abstol, reltol)
    total = 0

    while total < maxiter
        r = V[1]
        residual!(r, A, x, b, Ax); precond!(Pl, r)   # r  = Pl⁻¹(b - A x)
        β = norm(r); res = β
        converged(res, thr) && break
        r .*= inv(β)                                  # v₁ = r / β
        fill!(g, 0); g[1] = β
        k = 0
        for j in 1:min(m, maxiter - total)
            k = j; total += 1
            zⱼ = precond(Z[j], Pr, V[j])              # zⱼ = Prⱼ⁻¹ vⱼ into Z[j] (vⱼ itself without Pr)
            mul!(w, A, zⱼ); precond!(Pl, w)           # w  = Pl⁻¹ A zⱼ
            res, happy = arnoldi_step!(ws, w, j, o)
            (converged(res, thr) || happy) && break
        end
        hessenberg_solve!(H, g, k)
        for i in 1:k
            x .+= g[i] .* zs[i]                       # x += Σ yᵢ zᵢ
        end
        converged(res, thr) && break
    end
    return Result(total, res, converged(res, thr))
end

end # module Krylov
