module HaloArraysKrylovKitExt

# KrylovKit.jl as a LinearSolve `linsolve` for halo-array states.
#
# KrylovKit is type-agnostic: it touches the unknown only through VectorInterface
# (`similar`, `copyto!`, `add!!`, `scale!!`, `inner`, `norm`), which fall back to
# the vector-space methods HaloArrays defines (`dot`/`norm` global reductions,
# interior-only broadcasts). So a halo array of any dimensionality, or a field
# collection, is a KrylovKit vector as-is — no workspace hooks, no flattening.
# LinearSolve's own `KrylovKitJL` hands KrylovKit the operator directly, which
# fails for the matrix-free `W` operator OrdinaryDiffEq builds (it is not
# callable); this algorithm applies it through `mul!` instead.

using HaloArrays
using LinearSolve
using KrylovKit
using LinearAlgebra: mul!, norm, UniformScaling

const SciMLBase = LinearSolve.SciMLBase
const _IdentityOp = LinearSolve.SciMLOperators.IdentityOperator

const _KRYLOVKIT_METHODS = (:gmres, :cg, :bicgstab)

struct HaloKrylovKitAlg{K<:NamedTuple} <: LinearSolve.AbstractKrylovSubspaceMethod
    method::Symbol
    kwargs::K
end

function HaloArrays.HaloKrylovKit(method::Symbol = :gmres; kwargs...)
    method in _KRYLOVKIT_METHODS || throw(ArgumentError(string(
        "HaloKrylovKit: unsupported method :", method, ". Supported: ",
        join(_KRYLOVKIT_METHODS, ", "), ".")))
    return HaloKrylovKitAlg(method, NamedTuple(kwargs))
end

# KrylovKit's `tol` is an ABSOLUTE bound on the residual 2-norm; LinearSolve's
# `abstol`/`reltol` combine the LinearSolve way (reltol relative to ‖b‖).
_kk_alg(alg::HaloKrylovKitAlg, tol, maxiter) =
    alg.method === :gmres    ? KrylovKit.GMRES(;    tol, maxiter, alg.kwargs...) :
    alg.method === :cg       ? KrylovKit.CG(;       tol, maxiter, alg.kwargs...) :
                               KrylovKit.BiCGStab(; tol, maxiter, alg.kwargs...)

@inline _is_identity_precond(::_IdentityOp) = true
@inline _is_identity_precond(::UniformScaling) = true
@inline _is_identity_precond(_) = false

LinearSolve.default_alias_A(::HaloKrylovKitAlg, ::Any, ::Any) = true
LinearSolve.default_alias_b(::HaloKrylovKitAlg, ::Any, ::Any) = true
LinearSolve.init_cacheval(::HaloKrylovKitAlg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose, assump) = nothing
LinearSolve.update_tolerances_internal!(cache, ::HaloKrylovKitAlg, abstol, reltol) = nothing
# A halo array is one N-D field, not a batch of columns (see the LinearSolve ext).
LinearSolve._check_batched_rhs_support(::HaloKrylovKitAlg, ::AbstractHaloArray) = nothing

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::HaloKrylovKitAlg; kwargs...)
    (_is_identity_precond(cache.Pl) && _is_identity_precond(cache.Pr)) || throw(ArgumentError(
        "HaloKrylovKit: KrylovKit has no preconditioning; Pl/Pr are not supported. " *
        "Use HaloCG / HaloGMRES / HaloBiCGStab / HaloMINRES for a left preconditioner."))
    A = cache.A; b = cache.b
    tol = max(cache.abstol, cache.reltol * norm(b))
    x, info = KrylovKit.linsolve(v -> mul!(similar(v), A, v), b, cache.u, _kk_alg(alg, tol, cache.maxiters))
    x === cache.u || copyto!(cache.u, x)
    retcode = info.converged == 1 ? SciMLBase.ReturnCode.Success : SciMLBase.ReturnCode.MaxIters
    return SciMLBase.build_linear_solution(alg, cache.u, info.normres, cache;
                                           retcode, iters = info.numiter)
end

end
