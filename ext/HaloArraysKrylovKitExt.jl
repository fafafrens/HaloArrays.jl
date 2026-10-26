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
using KrylovKit: VectorInterface
using LinearAlgebra: mul!, norm, dot, rmul!, axpby!, UniformScaling

const SciMLBase = LinearSolve.SciMLBase
const _IdentityOp = LinearSolve.SciMLOperators.IdentityOperator

# ---- VectorInterface on halo arrays ------------------------------------------
# VectorInterface's generic `AbstractArray` fallbacks go through `map`/`eachindex`
# (`scale`, `add`, `zerovector!`) and a two-argument `mapreduce` (`inner`). On a
# tiled ThreadedHaloArray `eachindex` has no tile-less interior, so `map` throws;
# everywhere else those paths scalar-index the interior and allocate O(N) per
# inner product. Route them to the halo array's own vector-space kernels instead:
# interior-only broadcasts, the fused BLAS-1 updates, and the global `dot`/`norm`.
const VI = VectorInterface
VI.zerovector(x::AbstractHaloArray, ::Type{S}) where {S<:Number} = fill!(similar(x, S), zero(S))
VI.zerovector!(x::AbstractHaloArray) = fill!(x, zero(eltype(x)))
VI.scale(x::AbstractHaloArray, α::Number) = x .* α
VI.scale!(x::AbstractHaloArray, α::Number) = α === VI.One() ? x : rmul!(x, α)
VI.scale!(y::AbstractHaloArray, x::AbstractHaloArray, α::Number) = (y .= x .* α; y)
VI.scale!!(x::AbstractHaloArray, α::Number) =
    VI.promote_scale(x, α) <: VI.scalartype(x) ? VI.scale!(x, α) : VI.scale(x, α)
VI.scale!!(y::AbstractHaloArray, x::AbstractHaloArray, α::Number) =
    VI.promote_scale(y, x, α) <: VI.scalartype(y) ? VI.scale!(y, x, α) : VI.scale(x, α)
VI.add(y::AbstractHaloArray, x::AbstractHaloArray, α::Number, β::Number) = α .* x .+ β .* y
function VI.add!(y::AbstractHaloArray, x::AbstractHaloArray, α::Number, β::Number)
    T = VI.scalartype(y)
    return axpby!(convert(T, α), x, convert(T, β), y)
end
VI.add!!(y::AbstractHaloArray, x::AbstractHaloArray, α::Number, β::Number) =
    VI.promote_add(y, x, α, β) <: VI.scalartype(y) ? VI.add!(y, x, α, β) : VI.add(y, x, α, β)
VI.inner(x::AbstractHaloArray, y::AbstractHaloArray) = dot(x, y)

# ---- the LinearSolve algorithm -------------------------------------------------
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
# KrylovKit's `maxiter` counts iterations for CG/BiCGStab but restart CYCLES for
# GMRES (each up to `krylovdim` operator products), so LinearSolve's `maxiters`
# — a count of operator products — is converted to cycles there.
_kk_alg(alg::HaloKrylovKitAlg, tol, maxiters) =
    alg.method === :gmres    ? KrylovKit.GMRES(;    tol, maxiter = cld(maxiters, _krylovdim(alg)), alg.kwargs...) :
    alg.method === :cg       ? KrylovKit.CG(;       tol, maxiter = maxiters, alg.kwargs...) :
                               KrylovKit.BiCGStab(; tol, maxiter = maxiters, alg.kwargs...)
_krylovdim(alg::HaloKrylovKitAlg) = get(alg.kwargs, :krylovdim, KrylovKit.KrylovDefaults.krylovdim[])

@inline _is_identity_precond(::_IdentityOp) = true
@inline _is_identity_precond(J::UniformScaling) = isone(J.λ)
@inline _is_identity_precond(_) = false

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
    # `numops` (operator applications) is the count comparable to `maxiters` and
    # to the Halo solvers' `iters`; KrylovKit's `numiter` is restart cycles for GMRES.
    return SciMLBase.build_linear_solution(alg, cache.u, info.normres, cache;
                                           retcode, iters = info.numops)
end

end
