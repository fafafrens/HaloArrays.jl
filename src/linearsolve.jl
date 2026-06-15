# Entry point for the LinearSolve/Krylov extension. The method itself lives in
# `ext/HaloArraysLinearSolveExt.jl` and is only defined once both `LinearSolve`
# and `Krylov` are loaded (a package extension), so the heavy solver stack stays
# an optional dependency.

"""
    HaloKrylov(method::Symbol; kwargs...) -> LinearSolve algorithm

A LinearSolve-compatible iterative solver that runs any Krylov.jl `method`
directly on a halo array as the solution vector — matrix-free, no marshalling to
a plain `Vector`. Pass it as the `linsolve` of an implicit SciML integrator or a
`LinearProblem`:

```julia
using HaloArrays, LinearSolve, Krylov, OrdinaryDiffEq
solve(prob, FBDF(linsolve = HaloKrylov(:gmres), concrete_jac = false))
```

It allocates Krylov's work vectors with `Krylov.KrylovConstructor` (i.e. via
`similar`), which is what lets a geometry-carrying halo array be the solver
vector — unlike LinearSolve's stock `KrylovJL_*` wrappers, which allocate with
`S(undef, n)` and have no such constructor for a halo array.

`method` is any Krylov.jl solver symbol. The square-system, matrix-free methods
that allocate all their work vectors up front (so they need only the operator's
mat-vec) work directly on a halo array:

    :gmres :dqgmres :diom :fom :cg :car :bicgstab :cgs
    :minres :minres_qlp :symmlq :cg_lanczos :minares

Not supported: `:fgmres`, `:qmr`, `:bilq` allocate extra vectors mid-solve via
`S(undef, n)`, which a geometry-carrying halo array has no constructor for (the
same limitation as `KrylovJL_*`); and `:cr` requires an SPD operator (it rejects
an indefinite Newton `W`). Transpose/rectangular methods (`:lsqr`, `:lsmr`, …)
additionally require an adjoint or second operator. `kwargs` are forwarded to the
Krylov solver (e.g. `atol`, `rtol`, `itmax`).

!!! note
    Defined only when both `LinearSolve` and `Krylov` are loaded (a package
    extension). Without them, this throws a `MethodError`.

Requires the global reductions and BLAS-1 ops in [vector_space.jl](@ref) — `dot`,
`norm`, `axpy!`, … — which make the solve correct (and collective under MPI).
"""
function HaloKrylov end
