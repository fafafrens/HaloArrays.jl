# Linear & implicit solves

A halo array can be the **state vector of a matrix-free solve** — an implicit ODE
step, or a `LinearProblem` — without ever marshalling its interior to a flat
`Vector`. The operator is applied through `mul!`, and convergence is judged by
`dot`/`norm`, which HaloArrays defines as **global reductions** — so the same
solve is automatically correct across MPI ranks and threaded tiles.

These solvers live in a package **extension**, active only when `LinearSolve` and
`Krylov` are loaded:

```julia
using HaloArrays, LinearSolve, Krylov
```

## Which solver to use

The right choice depends on the **dimensionality** of the state, because
Krylov.jl / LinearSolve's stock solvers model the unknown as a flat
`AbstractVector`:

| state | solver | notes |
|---|---|---|
| **1-D** halo array | [`HaloKrylov`](@ref)`(:method)`, or stock `KrylovJL_*` | a 1-D halo array *is* an `AbstractVector`; mature Krylov.jl methods, cached |
| **2-D / 3-D** halo array | [`HaloCG`](@ref) / [`HaloBiCGStab`](@ref) / [`HaloGMRES`](@ref) | coordinate-free; the only option an N-D halo array can use |

[`HaloKrylov(:method)`](@ref) is a thin `KrylovJL` alias; the extension teaches
`KrylovJL`'s workspace to allocate with `similar` (via `KrylovConstructor`)
instead of `S(undef, n)`, so the **stock** `KrylovJL_GMRES()` etc. work on a 1-D
halo array too — and are cached by LinearSolve. But Krylov.jl requires
`b::AbstractVector`, so an `N≥2` halo array can't use it.

[`HaloCG`](@ref)/[`HaloBiCGStab`](@ref)/[`HaloGMRES`](@ref) are **coordinate-free**
(they touch the unknown only through `similar`/`copy`/broadcast/`dot`/`norm`/`mul!`),
so they take a halo array of *any* dimensionality.

## Implicit ODEs (Jacobian-free Newton–Krylov)

To integrate a stiff problem with the halo array as the SciML state, pass an
**iterative** `linsolve` together with `concrete_jac = false`:

```julia
using OrdinaryDiffEq
solve(prob, FBDF(linsolve = HaloKrylov(:gmres), concrete_jac = false))   # 1-D state
solve(prob, FBDF(linsolve = HaloGMRES(),        concrete_jac = false))   # N-D state
```

This works on **distributed states** too: OrdinaryDiffEq automatically wraps the
iterative solver with error-weight preconditioners (`Diagonal(weight)` where
`weight` is a halo array), and HaloArrays supplies elementwise `mul!`/`ldiv!`
methods for them, so the whole implicit solve — Newton, GMRES, preconditioning,
and the convergence-controlling `norm`/`dot` reductions — stays collective and
correct across MPI ranks.

!!! warning "You must select an iterative linsolve"
    With the *default* `linsolve` (a factorization), the integrator builds a
    *dense* Jacobian matrix — which can't be formed column-by-column from a
    halo-array state, and errors. Only an iterative `linsolve` switches the
    integrator to the matrix-free (Jacobian-vector-product) path. This is what
    makes the halo array usable as the implicit state.

The Newton solve then *just works*: it is composed entirely of the RHS, the
vector-space ops, and this linear solve — and its convergence norm is global, so
every MPI rank makes the same accept/converge decision (no deadlock). See
`examples/finite_volume/stiff_reaction_diffusion_implicit_1d.jl` and its
`_mpi_1d` companion.

## A `LinearProblem` directly

```julia
A = FunctionOperator(neg_laplacian!, similar(b), similar(b); islinear = true, ...)
sol = solve(LinearProblem(A, b; u0 = zero(b)), HaloCG())
```

!!! tip "Pass `u0` explicitly"
    For a `LinearProblem`, give `u0` (e.g. `zero(b)`). LinearSolve's default
    initial guess flattens `b` to a 1-D vector of length `size(A, 2)`, which a
    geometry-carrying halo array can't be.

See `examples/poisson/operator.jl` for a 2-D Dirichlet Poisson solved this way,
and `examples/poisson/mpi.jl` for the distributed version.
