# Linear & implicit solves

A halo array can be the **state vector of a matrix-free solve** — an implicit ODE
step, or a `LinearProblem` — without ever marshalling its interior to a flat
`Vector`. The operator is applied through `mul!`, and convergence is judged by
`dot`/`norm`, which HaloArrays defines as **global reductions** — so the same
solve is automatically correct across MPI ranks and threaded tiles.

These solvers live in package **extensions**: the coordinate-free
[`HaloCG`](@ref)/[`HaloMINRES`](@ref)/[`HaloBiCGStab`](@ref)/[`HaloGMRES`](@ref)
activate when `LinearSolve` is loaded, and [`HaloKrylovKit`](@ref) when
`LinearSolve` and `KrylovKit` are:

```julia
using HaloArrays, LinearSolve            # HaloCG, HaloMINRES, HaloBiCGStab, HaloGMRES
using KrylovKit                          # + HaloKrylovKit(:gmres | :cg | :bicgstab)
```

## Which solver to use

Both families take a halo array of **any dimensionality**, or a field
collection, as the unknown; both are matrix-free and MPI-collective.

| solver | for | notes |
|---|---|---|
| [`HaloCG`](@ref) | symmetric positive-definite `A` | cheapest; left preconditioner `Pl`; workspace cached |
| [`HaloMINRES`](@ref) | symmetric, possibly indefinite `A` | where CG would break down; no preconditioner |
| [`HaloBiCGStab`](@ref) | general `A` | fixed memory, two products per iteration; `Pl` and `Pr` |
| [`HaloGMRES`](@ref)`(; restart)` | general `A` | most robust; memory grows with `restart`; `Pl` and `Pr` |
| [`HaloKrylovKit`](@ref)`(:gmres | :cg | :bicgstab)` | same | KrylovKit.jl's implementations; no preconditioning, basis allocated per solve |

LinearSolve's stock `KrylovJL_*` (Krylov.jl) and `SimpleGMRES` are **not**
supported: they model the unknown as a flat `AbstractVector`, which a
geometry-carrying halo array is not.

## Implicit ODEs (Jacobian-free Newton–Krylov)

To integrate a stiff problem with the halo array as the SciML state, pass an
**iterative** `linsolve` together with `concrete_jac = false`:

```julia
using OrdinaryDiffEq
solve(prob, FBDF(linsolve = HaloGMRES(),             concrete_jac = false))
solve(prob, FBDF(linsolve = HaloKrylovKit(:gmres),   concrete_jac = false))   # needs KrylovKit
```

An N-D or multi-field state passes through untouched: `vec` is the identity on
any halo array (single array or field collection), so the integrator's internal
flattening around its matrix-free `W` operator never wraps the state in a
`ReshapedArray`, and `rhs!` keeps seeing its named fields.

To supply your own Jacobian-vector product instead of the automatic one, pass a
matrix-free operator as `jac_prototype` (the `jvp` keyword of `ODEFunction` is
not consulted on this path):

```julia
using LinearSolve.SciMLOperators: FunctionOperator
jvp!(Jv, v, u, p, t) = ...            # Jv .= J(u) * v; call synchronize_halo!(v) as rhs! does on u
J = FunctionOperator(jvp!, similar(u0), similar(u0); u = u0, p = p, t = 0.0, isinplace = true)
prob = ODEProblem(ODEFunction{true}(rhs!; jac_prototype = J), u0, tspan, p)
```

The integrator keeps `u`, `p`, `t` in the operator up to date at every Newton
iterate, and the halo array (or collection) reaches `jvp!` unflattened. This
replaces the `rhs!` evaluations the automatic JVP would make.

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

See `examples/poisson/operator.jl` for a 2-D Dirichlet Poisson solved this way.
Its `poisson/mpi.jl` companion applies the same matrix-free stencil with
hand-written coordinate-free Krylov solvers across MPI ranks. For repeated
`LinearProblem` solves, `examples/schrodinger/crank_nicolson_2d.jl` reuses one
`LinearSolve` cache and one `HaloGMRES` workspace across every Crank–Nicolson
time step.
