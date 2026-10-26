module HaloArraysLinearSolveExt

# LinearSolve glue for the solvers in `HaloArrays.Krylov`: an algorithm type per
# solver, a workspace built once at `init` and cached in `cache.cacheval`, and a
# `solve!` that runs the kernel and reports a proper retcode. The kernels
# themselves (and their docs) live in src/krylov.jl and need no LinearSolve.

using HaloArrays
using HaloArrays.Krylov
using LinearSolve
using LinearAlgebra: UniformScaling

const SciMLBase = LinearSolve.SciMLBase

# ---- algorithms: the exported constructors (src/linearsolve.jl) return these ----
struct HaloCGAlg       <: LinearSolve.AbstractKrylovSubspaceMethod end
struct HaloMINRESAlg   <: LinearSolve.AbstractKrylovSubspaceMethod end
struct HaloBiCGStabAlg <: LinearSolve.AbstractKrylovSubspaceMethod end
struct HaloGMRESAlg    <: LinearSolve.AbstractKrylovSubspaceMethod
    restart::Int
    orth::Symbol
end
struct HaloFGMRESAlg   <: LinearSolve.AbstractKrylovSubspaceMethod
    restart::Int
    orth::Symbol
end
const _HaloAlg = Union{HaloCGAlg, HaloMINRESAlg, HaloBiCGStabAlg, HaloGMRESAlg, HaloFGMRESAlg}

HaloArrays.HaloCG()       = HaloCGAlg()
HaloArrays.HaloMINRES()   = HaloMINRESAlg()
HaloArrays.HaloBiCGStab() = HaloBiCGStabAlg()
function HaloArrays.HaloGMRES(; restart = 30, orth = :auto)
    restart ≥ 1 || throw(ArgumentError("HaloGMRES: `restart` must be ≥ 1, got $restart"))
    orth in Krylov.ORTH || throw(ArgumentError("HaloGMRES: `orth` must be one of $(Krylov.ORTH), got $(repr(orth))"))
    return HaloGMRESAlg(restart, orth)
end
function HaloArrays.HaloFGMRES(; restart = 30, orth = :auto)
    restart ≥ 1 || throw(ArgumentError("HaloFGMRES: `restart` must be ≥ 1, got $restart"))
    orth in Krylov.ORTH || throw(ArgumentError("HaloFGMRES: `orth` must be one of $(Krylov.ORTH), got $(repr(orth))"))
    return HaloFGMRESAlg(restart, orth)
end

# ---- LinearSolve plumbing ----
# LinearSolve rejects an `AbstractMatrix` right-hand side for Krylov subspace
# methods, reading extra columns as a batch of independent vectors. A 2-D halo
# array is one N-D field, not a batch, and the kernels are coordinate-free. Opt
# out for THESE algorithms only (a broader method is ambiguous with KrylovJL's).
LinearSolve._check_batched_rhs_support(::_HaloAlg, ::AbstractHaloArray) = nothing
# OrdinaryDiffEq updates the per-step tolerances through `update_tolerances!`,
# which sets `cache.abstol`/`cache.reltol` (read fresh by every `solve!`) and
# then calls this hook; the generic fallback only warns "not implemented".
LinearSolve.update_tolerances_internal!(cache, ::_HaloAlg, abstol, reltol) = nothing

# ---- workspace, built once at init ----
LinearSolve.init_cacheval(::HaloCGAlg,       A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose, assump) = CGWorkspace(b)
LinearSolve.init_cacheval(::HaloMINRESAlg,   A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose, assump) = MINRESWorkspace(b)
LinearSolve.init_cacheval(::HaloBiCGStabAlg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose, assump) = BiCGStabWorkspace(b)
LinearSolve.init_cacheval(alg::HaloGMRESAlg,  A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose, assump) =
    GMRESWorkspace(b; restart = alg.restart)
LinearSolve.init_cacheval(alg::HaloFGMRESAlg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose, assump) =
    FGMRESWorkspace(b; restart = alg.restart)

# ---- solve ----
# LinearSolve's "no preconditioner" is its `IdentityOperator` (users also pass
# `I`); the kernels take `nothing`.
_precond(::LinearSolve.SciMLOperators.IdentityOperator) = nothing
_precond(J::UniformScaling) = isone(J.λ) ? nothing : J
_precond(P) = P

# Which kernel runs, and which preconditioner sides it takes — by dispatch, so
# an unsupported side is an error rather than silently dropped. CG needs a
# symmetric operator, which only the left form (in the M-inner product)
# preserves; MINRES would need the SPD Lanczos rework to precondition at all.
_run!(::HaloCGAlg,       x, A, b, ws, Pl, ::Nothing;        kw...) = cg!(x, A, b, ws; Pl, kw...)
_run!(::HaloMINRESAlg,   x, A, b, ws, ::Nothing, ::Nothing; kw...) = minres!(x, A, b, ws; kw...)
_run!(::HaloBiCGStabAlg, x, A, b, ws, Pl, Pr;               kw...) = bicgstab!(x, A, b, ws; Pl, Pr, kw...)
_run!(a::HaloGMRESAlg,   x, A, b, ws, Pl, Pr;               kw...) = gmres!(x, A, b, ws; Pl, Pr, orth = a.orth, kw...)
_run!(a::HaloFGMRESAlg,  x, A, b, ws, Pl, Pr;               kw...) = fgmres!(x, A, b, ws; Pl, Pr, orth = a.orth, kw...)
_run!(::HaloCGAlg, x, A, b, ws, Pl, Pr; kw...) = throw(ArgumentError(
    "HaloCG applies a LEFT preconditioner (Pl) only (A Pr⁻¹ is not symmetric); a right " *
    "preconditioner Pr::$(typeof(Pr)) is not supported. Use HaloGMRES/HaloBiCGStab."))
_run!(::HaloMINRESAlg, x, A, b, ws, Pl, Pr; kw...) = throw(ArgumentError(
    "HaloMINRES does not support preconditioning; use HaloCG (SPD) or HaloGMRES with `Pl`."))

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::_HaloAlg; kwargs...)
    r = _run!(alg, cache.u, cache.A, cache.b, cache.cacheval, _precond(cache.Pl), _precond(cache.Pr);
              abstol = cache.abstol, reltol = cache.reltol, maxiter = cache.maxiters)
    retcode = r.converged ? SciMLBase.ReturnCode.Success : SciMLBase.ReturnCode.MaxIters
    return SciMLBase.build_linear_solution(alg, cache.u, r.residual, cache; retcode, iters = r.iters)
end

end # module
