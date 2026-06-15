# Entry point for the LinearSolve/Krylov extension. The method itself lives in
# `ext/HaloArraysLinearSolveExt.jl` and is only defined once both `LinearSolve`
# and `Krylov` are loaded (a package extension), so the heavy solver stack stays
# an optional dependency.

"""
    HaloKrylov(method::Symbol; kwargs...) -> KrylovJL

A LinearSolve-compatible iterative solver that runs a Krylov.jl `method` directly
on a halo array as the solution vector — matrix-free, no marshalling to a plain
`Vector`. A thin, symbol-keyed alias for `LinearSolve.KrylovJL`, so it plugs in as
the `linsolve` of an implicit SciML integrator or a `LinearProblem`:

```julia
using HaloArrays, LinearSolve, Krylov, OrdinaryDiffEq
solve(prob, FBDF(linsolve = HaloKrylov(:gmres), concrete_jac = false))
```

The enabling piece is a method this extension adds to `LinearSolve.init_cacheval`
for halo-array operands: it builds Krylov's workspace with
`Krylov.KrylovConstructor` (i.e. via `similar`) instead of `S(undef, n)`, which is
what a geometry-carrying halo array has no constructor for. Because it hooks
LinearSolve's own cache path, **the stock `KrylovJL_*` algorithms work on a halo
array too** — e.g. `linsolve = KrylovJL_GMRES()` — and the Krylov workspace is
built once and **cached/reused** across solves by LinearSolve's `solve!`.
`HaloKrylov(:method)` is just the convenient way to reach methods without a named
`KrylovJL_*` wrapper.

Supported `method`s (matrix-free, all work vectors allocated up front):

    :gmres :dqgmres :diom :fom :cg :bicgstab :cgs
    :minres :minres_qlp :symmlq :minares

Unsupported, with a helpful error: `:fgmres`, `:qmr`, `:bilq` allocate extra
vectors mid-solve via `S(undef, n)` (the geometry-array wall); `:car` and
`:cg_lanczos` aren't wrapped by LinearSolve's `KrylovJL`. `kwargs` are forwarded
to `KrylovJL` / the Krylov solver (e.g. `atol`, `rtol`, `itmax`, `gmres_restart`).

!!! note
    Defined only when both `LinearSolve` and `Krylov` are loaded (a package
    extension); without them this throws a `MethodError`. The solve relies on the
    global reductions and BLAS-1 ops in `vector_space.jl` (`dot`, `norm`,
    `axpy!`, …), which also make it correct and collective under MPI.
"""
function HaloKrylov end
