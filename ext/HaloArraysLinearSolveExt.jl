module HaloArraysLinearSolveExt

using HaloArrays
using LinearSolve
using Krylov

# `HaloKrylov(:method)` → a `LinearSolveFunction` that runs Krylov.jl's `method`
# with a halo array as the solution vector. The work vectors are allocated by
# `KrylovConstructor` (via `similar`), so a geometry-carrying halo array works
# directly; the solve is matrix-free (only `A`'s mat-vec is used) and, on a
# distributed `HaloArray`, collective (its `dot`/`norm` Allreduce).
#
# LinearSolveFunction passes `(A, b, u, p, isfresh, Pl, Pr, cacheval; kwargs...)`;
# we (re)build the workspace from `b`, solve in place, and copy the solution back
# into `u`. The Krylov workspace is cached in `cacheval` and reused while it is
# not stale (`isfresh == false`) and its geometry still matches `b`.
function HaloArrays.HaloKrylov(method::Symbol; kwargs...)
    solve_kwargs = kwargs
    bridge = function (A, b, u, _p, isfresh, _Pl, _Pr, cacheval; _ignore...)
        ws = _halokrylov_workspace(Val(method), b, isfresh, cacheval)
        Krylov.krylov_solve!(ws, A, b; solve_kwargs...)
        copyto!(u, Krylov.solution(ws))
        return u
    end
    return LinearSolveFunction(bridge)
end

# Reuse the cached workspace when it is fresh and sized for `b`; otherwise build
# a new one from a KrylovConstructor.
@inline function _halokrylov_workspace(::Val{method}, b, isfresh, cacheval) where {method}
    if !isfresh && cacheval isa Krylov.KrylovWorkspace && length(Krylov.solution(cacheval)) == length(b)
        return cacheval
    end
    return Krylov.krylov_workspace(Val(method), Krylov.KrylovConstructor(b))
end

end # module
