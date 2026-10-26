# Entry points for the LinearSolve extensions. The methods live in
# `ext/HaloArraysLinearSolveExt.jl` (coordinate-free solvers, needs `LinearSolve`)
# and `ext/HaloArraysKrylovKitExt.jl` (KrylovKit as a `linsolve`, needs
# `LinearSolve` and `KrylovKit`), so the solver stack stays optional.

"""
    HaloKrylovKit(method::Symbol = :gmres; kwargs...)

A LinearSolve algorithm that runs a KrylovKit.jl `method` (`:gmres`, `:cg`,
`:bicgstab`) with a halo array — of any dimensionality, or a field collection —
as the unknown, matrix-free. KrylovKit touches the unknown only through
VectorInterface (`similar`, `add!!`, `scale!!`, `inner`, `norm`), which resolve
to the halo array's own vector-space methods, so `dot`/`norm` stay global
reductions and the solve is collective under MPI. Pass it as the `linsolve` of
an implicit SciML integrator or of a `LinearProblem`:

```julia
using HaloArrays, LinearSolve, KrylovKit, OrdinaryDiffEq
solve(prob, FBDF(linsolve = HaloKrylovKit(:gmres; krylovdim = 30), concrete_jac = false))
```

`kwargs` are forwarded to the KrylovKit algorithm (`krylovdim`, `orth`,
`verbosity`); `tol` and `maxiter` come from LinearSolve's `abstol`/`reltol`/
`maxiters` (KrylovKit's absolute `tol` is set to `max(abstol, reltol * norm(b))`).
The operator is applied through `mul!`, so OrdinaryDiffEq's matrix-free `W`
operator works. KrylovKit has no preconditioning: a non-identity `Pl`/`Pr` throws;
use [`HaloCG`](@ref)/[`HaloGMRES`](@ref)/… for a left preconditioner. KrylovKit
allocates its Krylov basis per solve (no workspace caching).

!!! note
    Defined only when both `LinearSolve` and `KrylovKit` are loaded (a package
    extension); without them this throws a `MethodError`.
"""
function HaloKrylovKit end

"""
    HaloCG()
    HaloBiCGStab()
    HaloMINRES()
    HaloGMRES(; restart = 30)

LinearSolve algorithms for solving a linear system with a **halo array of any
dimensionality** as the unknown — matrix-free, no marshalling to a flat vector.
Pass as the `linsolve` of a `LinearProblem` or an implicit SciML integrator:

```julia
using HaloArrays, LinearSolve
sol = solve(LinearProblem(A, b; u0 = zero(b)), HaloCG())   # b an N-D halo array
```

Unlike LinearSolve's stock `KrylovJL_*` (which require a 1-D `b::AbstractVector`
and so cannot take an N-D halo array), these are **coordinate-free**: they touch the
unknown only through `similar`/`copy`/broadcast/`dot`/`norm`/`mul!`, so they run
on a 2-D or 3-D halo array, and stay correct under MPI (`dot`/`norm` are global
reductions). The Krylov workspace is built once and cached by LinearSolve.

- `HaloCG` — Conjugate Gradient, for symmetric/Hermitian positive-definite `A`.
- `HaloMINRES` — MINRES, for symmetric/Hermitian `A` (including **indefinite**,
  where `HaloCG` would break down).
- `HaloBiCGStab` — BiCGStab, for general square `A`.
- `HaloGMRES` — restarted GMRES (`restart` basis vectors), for general square `A`.

Preconditioning follows the LinearSolve convention (`Pl`, `Pr` applied with
`ldiv!`; the solved system is `Pl⁻¹ A Pr⁻¹ y = Pl⁻¹ b`, `x = Pr⁻¹ y`):
`HaloGMRES`/`HaloBiCGStab` take both sides, `HaloCG` takes `Pl` only (the
right form is not symmetric), `HaloMINRES` takes neither. An unsupported
preconditioner throws rather than being silently ignored. With `Pr` only, the
monitored residual is the true residual `b - Ax`.

For a `LinearProblem`, pass `u0` explicitly (e.g. `zero(b)`) — LinearSolve's
default initial guess flattens `b` to a 1-D vector, which a halo array can't be.

!!! note
    Defined only when `LinearSolve` is loaded (a package extension); without it
    this throws a `MethodError`.
"""
function HaloCG end
function HaloBiCGStab end
function HaloMINRES end
function HaloGMRES end
@doc (@doc HaloCG) HaloBiCGStab
@doc (@doc HaloCG) HaloMINRES
@doc (@doc HaloCG) HaloGMRES
