# Examples

Runnable examples for local, threaded, MPI-backed, GPU, and DiffEq workflows,
grouped by topic:

```
tutorials/      progressive, self-contained walkthroughs (start here)
heat/           heat diffusion — the simplest stencil
finite_volume/  conservative finite volume (Burgers, advection)
hydro/          2-D ideal hydrodynamics
lattice/        lattice field theory Monte Carlo (scalar φ⁴, SU(2) Wilson)
poisson/        matrix-free Krylov solves of a Poisson problem
```

## Setup

The package test project is enough for the local and MPI heat examples:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

DiffEq, GPU, and plotting examples use optional packages. From the repository
root, create the examples environment once:

```bash
julia --project=examples -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
```

The generated `examples/Manifest.toml` is intentionally ignored by git.

## Tutorials

Step-by-step, heavily commented walkthroughs. Each is self-contained.

```bash
julia --project=. examples/tutorials/local.jl
julia --project=. -t 4 examples/tutorials/threaded.jl
julia --project=. -t 4 examples/tutorials/broadcast.jl
mpiexec -n 4 julia --project=. examples/tutorials/mpi.jl
julia --project=examples -t 4 examples/tutorials/gpu.jl       # needs Metal.jl (macOS)
julia --project=examples examples/tutorials/diffeq.jl
```

## Heat diffusion (`heat/`)

These use the shared finite-difference kernels in `heat/common.jl`. The single
local and MPI entry points each run the 1-D, 2-D, and 3-D cases:

```bash
julia --project=. examples/heat/local.jl
mpiexec -n 4 julia --project=. examples/heat/mpi.jl
```

`heat/local_vs_threaded.jl` solves the same 2-D problem on a `LocalHaloArray` and
a `ThreadedHaloArray`, both by hand (explicit Euler) and via OrdinaryDiffEq:

```bash
julia --project=examples -t 4 examples/heat/local_vs_threaded.jl
```

`heat/cpu_vs_gpu_2d.jl` solves the same 2-D periodic problem with a CPU-backed and
a Metal-backed `LocalHaloArray` (KernelAbstractions, colored face regions, the
interior-cell region API). Requires an Apple GPU supported by Metal.jl:

```bash
julia --project=examples examples/heat/cpu_vs_gpu_2d.jl
```

To save a gathered MPI snapshot to HDF5:

```bash
mpiexec -n 4 julia --project=. -e 'include("examples/heat/mpi.jl"); run_mpi_heat_2d(save_hdf5=true)'
```

## Finite volume (`finite_volume/`)

1-D inviscid Burgers with a conservative update and Rusanov fluxes, plus periodic
linear advection with an upwind flux. The DiffEq variants share the semi-discrete
RHS in `finite_volume/common.jl`:

```bash
julia --project=. examples/finite_volume/burgers_1d.jl
mpiexec -n 4 julia --project=. examples/finite_volume/burgers_mpi_1d.jl
julia --project=examples examples/finite_volume/burgers_diffeq_1d.jl
julia --project=examples examples/finite_volume/advection_diffeq_1d.jl
```

`stiff_reaction_diffusion_implicit_1d.jl` shows an **implicit** SciML solve with
autodiff Jacobians using the `HaloArray` *as the ODE state* — matrix-free
(`concrete_jac=false`) via `SimpleGMRES()`. The key point: pick a `similar`-based
linear solver (`SimpleGMRES`, or `IterativeSolversJL_CG` for symmetric systems),
not the `KrylovJL_*` wrappers, which allocate work vectors as `S(undef, n)` and
can't construct a geometry-carrying `HaloArray`.

```bash
julia --project=examples examples/finite_volume/stiff_reaction_diffusion_implicit_1d.jl
```

`acoustics_characteristic_1d.jl` demonstrates a **coupled** boundary condition:
1-D linear acoustics on two fields `(p, u)` with a characteristic non-reflecting
outflow that mixes both fields at the edge (via `AbstractCoupledBoundaryCondition`
+ `apply_coupled_bc!`). A right-moving pulse exits the domain with negligible
reflection.

```bash
julia --project=. examples/finite_volume/acoustics_characteristic_1d.jl
```

### Special relativistic hydrodynamics

Valencia-form special relativistic Euler with a Rusanov flux and SSP-RK2. The
`relativistic_common.jl` file holds the shared mass-based (D, S, τ) kernels; the
other variants are self-contained. Conserved→primitive recovery is the crux:
closed-form for μ=0, a 1-D Newton on pressure for the mass-based scheme, and a
2-D Newton in (T, μ) when a conserved charge is present.

```bash
# 1-D
julia --project=. examples/finite_volume/relativistic_hydro_1d.jl            # mass-based (D,S,τ), coupled outflow
julia --project=. examples/finite_volume/relativistic_hydro_mu0_1d.jl        # conformal, charge-free (closed form)
julia --project=. examples/finite_volume/relativistic_hydro_Tmu_1d.jl        # conserved charge, (T,μ) primitives
julia --project=. examples/finite_volume/relativistic_hydro_cylindrical_1d.jl

# 2-D circular blast wave (directional fluxes + Mignone–Bodo wave speeds)
julia --project=. examples/finite_volume/relativistic_hydro_mu0_2d.jl        # conformal, μ=0  (Mx,My,E)
julia --project=. examples/finite_volume/relativistic_hydro_Tmu_2d.jl        # with charge     (N,Mx,My,E)

# 3-D spherical blast wave
julia --project=. examples/finite_volume/relativistic_hydro_mu0_3d.jl        # conformal, μ=0  (Mx,My,Mz,E)
julia --project=. examples/finite_volume/relativistic_hydro_Tmu_3d.jl        # with charge     (N,Mx,My,Mz,E)
```

The multi-D runs evolve a centred over-pressure disk/sphere into a circular/
spherical blast with one `accumulate_flux_divergence!` sweep per axis; energy
(and, for the charge version, the charge) is conserved to ~machine precision and
the result is symmetric across axes.

## Ideal hydrodynamics (`hydro/`)

2-D non-relativistic ideal hydro in conservative variables, periodic halos,
first-order Rusanov fluxes, stepped by OrdinaryDiffEq:

```bash
julia --project=examples examples/hydro/local_2d.jl
JULIA_NUM_THREADS=4 julia --project=examples examples/hydro/threaded_2d.jl
mpiexec -n 4 julia --project=examples examples/hydro/mpi_2d.jl
julia --project=examples examples/hydro/plot_2d.jl   # writes an SVG comparing initial/final fields
```

The threaded run picks a 2-D tile decomposition whose tile count equals
`Threads.nthreads()`; a manual `tile_dims` must satisfy
`prod(tile_dims) == Threads.nthreads()`.

## Lattice field theory (`lattice/`)

A 2-D free scalar field heat-bath update with checkerboard sweeps and periodic
halos. The local/threaded version also shows named (`MultiHaloArray`) and indexed
(`ArrayOfHaloArray`) field collections:

```bash
julia --project=. examples/lattice/scalar_local_threaded_2d.jl
mpiexec -n 4 julia --project=. examples/lattice/scalar_mpi_2d.jl
```

Metal Monte Carlo using `LocalHaloArray` storage on the GPU with colored cell
regions. The Philox variants use stateless per-site RNG, avoiding per-sweep
random-array allocation. These require an Apple GPU supported by Metal.jl:

```bash
julia --project=examples examples/lattice/phi4_metal_2d.jl
julia --project=examples examples/lattice/phi4_metal_philox_2d.jl
julia --project=examples examples/lattice/su2_wilson_metal_2d.jl
```

2-D pure SU(2) Wilson plaquette Metropolis on the CPU with both `LocalHaloArray`
and `ThreadedHaloArray`. Gauge links are an `ArrayOfHaloArray` of field shape
`(4, 2)` (quaternion component × link direction). Choose a lattice size divisible
by the number of threads:

```bash
julia --project=. examples/lattice/su2_wilson_local_threaded_2d.jl
JULIA_NUM_THREADS=4 julia --project=. examples/lattice/su2_wilson_local_threaded_2d.jl
```

## Matrix-free Poisson (`poisson/`)

Wraps the `-∇²` stencil as a `SciMLOperators.FunctionOperator` and solves a
Dirichlet Poisson problem against a manufactured solution, verifying O(h²)
convergence — three ways (CG, BiCGStab, GMRES) using the coordinate-free Krylov
solvers in `poisson/krylov_solvers.jl`, which run directly on a halo array
(`mul!`, `dot`, `norm`, broadcast). Because `dot`/`norm` are global reductions,
the identical solvers and operator give a correct *distributed* solve under MPI:

```bash
julia --project=examples examples/poisson/operator.jl
mpiexec -n 4 julia --project=examples examples/poisson/mpi.jl
```

A 4-rank (2×2) run reproduces the serial n=64 result exactly (CG and GMRES match
iteration-for-iteration; all three reach error 1.5e-5). `poisson/mpi.jl` uses an
operator written with separate `hx`, `hy`, so it stays correct for any rank
decomposition (e.g. a 2×1 split → 64×32 global grid, which a single-`h` operator
would get wrong).

`poisson/krylov_solvers.jl` is a small, reusable, coordinate-free implementation
of CG / BiCGStab / GMRES (reimplemented from the standard algorithms, not copied
from Krylov.jl). It works on any vector type with `similar`, `copy`,
broadcasting, `dot`, `norm`, and `mul!` — plain arrays or any HaloArray backend.

## Notes

- If the machine has fewer cores than MPI ranks, add `--oversubscribe` to
  `mpiexec`.
- The examples avoid plotting packages by default so they run headless.
