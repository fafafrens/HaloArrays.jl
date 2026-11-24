# HaloArrays.jl

Julia package (Julia ≥ 1.10, repo `fafafrens/HaloArrays.jl`): arrays with
ghost (halo) cells and one API for serial, threaded-tile and MPI stencil/PDE
codes. A stencil written against `interior_view` / `synchronize_halo!` runs
unchanged on every backend; only the constructor changes. Storage can be a GPU
array (KernelAbstractions + Adapt). Halo arrays are also vector-space objects, so
they are the state of OrdinaryDiffEq integrators and of matrix-free Krylov solves.

## Layout

```
src/HaloArrays.jl            module, include order, the full export list
src/abstract_haloarray.jl    abstract types, backends, shared indexing/show
src/local_haloarray.jl       LocalHaloArray   (one block, one process)
src/threaded_haloarray.jl    ThreadedHaloArray (shared-memory tiles)
src/haloarray.jl             HaloArray        (MPI, on a CartesianTopology)
src/mpi_support.jl           MPI exchange state, per-face pack/post/unpack/finish
src/cartesian_topology.jl    CartesianTopology / ThreadedCartesianTopology
src/thread_backend.jl        ThreadsBackend / SerialBackend / Polyester / OhMyThreads
src/field_collection.jl      FieldCollection: shared machinery of the collections
src/multihaloarray.jl        MultiHaloArray   (named fields)
src/ArrayOfHaloArray.jl      ArrayOfHaloArray (index-addressed fields)
src/field_access.jl          siteview / sitevector / field_storages
src/maybehaloarray.jl        MaybeHaloArray (root-only results), is_active
src/boundary.jl              boundary conditions as per-cell rules bc(u, k)
src/halo_exchange.jl         synchronize_halo!, halo_exchange!, split start/finish
src/cell_ranges.jl, face_ranges.jl                    CellRanges, FaceRanges
src/cell_kernel_windows.jl, face_kernel_windows.jl    windows/checkerboards for kernels
src/interior_broadcast*.jl   broadcast over the interior
src/reduction.jl             reductions, DimReductionPlan, reduce!
src/vector_space.jl          BLAS-1 (`_interior_map!`), dot/norm, dots!
src/krylov.jl                HaloArrays.Krylov: cg!/minres!/bicgstab!/gmres!/fgmres!
src/linearsolve.jl           HaloCG/HaloGMRES/... algorithm types (methods in ext)
src/grid.jl                  coordinate systems, axes, CellGeometry
src/gather.jl, hdf5_api.jl, adapt.jl
ext/                         DiffEqBase, HDF5, KernelAbstractions(+GPUArraysCore),
                             LinearSolve, KrylovKit, OhMyThreads, Polyester
test/                        runtests.jl + one test_*.jl per area (test_mpi_* need ranks)
benchmark/                   own Project.toml; see benchmark/README.md
docs/                        Documenter site (guide, indexing, loops, habits, solvers, api/)
examples/                    own Project.toml; heat, finite_volume, hydro, lattice,
                             poisson, schrodinger, dg, tutorials
CHANGELOG.md                 Keep-a-Changelog; every user-visible change gets an entry
```

`archive/old/` is ignored (`.ignore`). `LocalPreferences.toml` and every
`Manifest.toml` are gitignored: they hold one machine's MPI/HDF5 choice and
resolved versions.

## Commands

```bash
# unit tests (Aqua, JLArrays, KernelAbstractions, LinearSolve tests only run under Pkg.test)
julia --project=. -e 'using Pkg; Pkg.test()'
JULIA_NUM_THREADS=4 julia --project=. -e 'using Pkg; Pkg.test()'

# MPI tests: 2 and 4 ranks
HALOARRAYS_RUN_UNIT_TESTS=false mpiexec -n 4 julia --project=. test/runtests.jl
# if the system mpiexec does not match MPI.jl's library, use MPI.jl's:
julia --project=. -e 'using MPI; run(`$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()) --project=. test/runtests.jl`)'

# example smoke tests (opt-in, slow; CI shards them 1/3..3/3)
HALOARRAYS_RUN_UNIT_TESTS=false HALOARRAYS_RUN_EXAMPLE_TESTS=true julia --project=. -e 'using Pkg; Pkg.test()'

# one test file
julia --project=. -e 'using Test, MPI, HaloArrays; MPI.Init(); include("test/test_boundary.jl")'

# benchmarks (setup once, then run)
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=benchmark -t 8 benchmark/stencil.jl
julia --project=benchmark benchmark/run_mpi.jl 4          # HALO_BENCH_QUICK=1 for a smoke run

# docs
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl                          # output in docs/build
```

Test switches: `HALOARRAYS_RUN_UNIT_TESTS`, `HALOARRAYS_RUN_MPI_TESTS` (on by
default when ranks > 1), `HALOARRAYS_RUN_EXAMPLE_TESTS`, `HALOARRAYS_EXAMPLE_SHARD`.

CI (`.github/workflows/ci.yml`): unit on Julia 1.10 and 1, threaded with 2 and 4
threads, example smoke tests in 3 shards, MPI with 2 and 4 ranks. It triggers on
`pull_request` opened/synchronize/reopened against `main` only; a draft retargeted
to main or marked ready does not start it: `gh workflow run ci.yml --ref <branch>`.

The docs build fails if an exported symbol has no docstring (`checkdocs = :exports`)
and `test_public_api.jl` checks the export list, so a new export needs a docstring,
a place on a `docs/src/api/*.md` page, and a test.

## Benchmark notes

Reference numbers and every command are in `benchmark/README.md`. Things that
bite when running them:
- `Pkg.develop(path=".")` under Julia 1.11+ appends a `[sources]` table to
  `benchmark/Project.toml` (and `Pkg.update()` does the same to
  `examples/Project.toml`, reordering it too); do not commit that
  (`git checkout <env>/Project.toml`). `Pkg.status` then warns that the project
  changed since the manifest was resolved; that is harmless.
- The `benchmark` and `examples` manifests are gitignored and go stale; if a
  script fails to load a package, `Pkg.update()` in that environment.
- Launch MPI runs through `MPI.mpiexec()` of the same project, as `run_mpi.jl`
  does. Without a `LocalPreferences.toml` an environment uses the MPI from
  `MPICH_jll`, whatever the root project is pinned to.
- `benchmarktools_times!` in `benchmark/common.jl` must pass a finite `seconds`:
  BenchmarkTools 1.8 converts `seconds / sample_time` to an `Int`, so `Inf`
  throws `InexactError`.
- `stencil.jl` tiles as `(1, nthreads)` and silently skips the threaded case when
  the thread count does not divide the size.
- Results are single runs unless you repeat them; small GPU sizes vary by tens
  of percent from run to run.

## API

### Arrays

```julia
u = LocalHaloArray(Float64, (64, 64), 1; boundary_condition=:periodic)
# size is per tile: 2×2 tiles of 64×64 give a 128×128 domain
u = ThreadedHaloArray(Float64, (64, 64), 1; dims=(2, 2), boundary_condition=:periodic,
                      thread_backend=ThreadsBackend())
# size is per rank: the domain is the size times the process grid
topo = CartesianTopology(comm, (0, 0); periodic=(true, true))
u = HaloArray(Float64, (128, 128), 1, topo; boundary_condition=:periodic)

# several fields on one grid; first argument is the backend's array type
s = MultiHaloArray(LocalHaloArray, Float64, (32, 32), 1;
                   fields=(:E, :Mx, :My, :D), boundary_condition=:periodic)
s = MultiHaloArray((; rho, mom))                      # from existing arrays
U = ArrayOfHaloArray(LocalHaloArray, Float64, (2, 2), (16, 16, 16), 1;
                     boundary_condition=:periodic)   # 2×2 fields per cell
```

Every constructor takes `storage=` (`zeros` by default, e.g. `CUDA.zeros`);
`Adapt.adapt(CuArray, u)` moves an existing array with its boundary conditions.
Positional arguments are always `(T, per-block size, halo width[, topology])`.

### Boundary conditions

`boundary_condition=` is one condition for all faces, or a tuple per dimension of
`(low, high)`: `((Reflecting(), Repeating()), (:periodic, :periodic))`.
Built-ins: `Periodic`, `Mirror(sign)` (`Reflecting()`, `Antireflecting()`),
`Repeating`, `Extrapolation(order)`, `NoBoundaryCondition`.

A condition is a per-cell rule `bc(u, k) -> value of one ghost cell`: `u` is a
`FaceCells` accessor (`u(j)` interior cell `j` in from the wall, `far(u)` the same
from the opposite wall, `global_index(u)` the ghost's global index), `k` the
ghost's distance from the wall. A custom condition is a callable type or a bare
function (wrapped as `FunctionBC`): `(u, k) -> 2g - u(0)` is a Dirichlet wall.
The same method runs on CPU, in a GPU kernel, per tile and on MPI physical faces.

Coupled conditions (ghosts of one field depend on the others): mark the faces
`NoBoundaryCondition`, subtype `AbstractCoupledBoundaryCondition`, implement
`HaloArrays.apply_coupled_bc!(bc, state, ::Side{S}, ::Dim{D}, tile)` using
`eachfield`, `edge_view`, `ghost_view`, and call `apply_coupled_bc!(bc, state)`
after `synchronize_halo!(state)`.

### Ghost cells

```julia
synchronize_halo!(u)            # exchange + boundary conditions: the only place ghosts are filled
start_halo_exchange!(u); work!(u); finish_halo_exchange!(u); boundary_condition!(u)   # split form
```

### Sizes, indices, access

| call | meaning |
|---|---|
| `size(u)`, `axes(u)` | whole domain (global) |
| `interior_size(u)`, `interior_view(u)` | cells this block owns, ghost-free |
| `storage_size(u)`, `parent(u)` | padded backing array |
| `interior_range(u)` | owned cells in storage coordinates |
| `halo_width(u)`, `unit_vector(u, d)` | stencil offsets |
| `tile_count(u)`, `tile_parent(u, t)`, `CellRanges(u, t)` | per-tile storage and ranges |
| `tile_foreach(f, u)`, `tile_mapreduce` | run over tiles on the array's thread backend |
| `fill_from_global_indices!(f, u)` | initialise from a function of the global index |
| `interior_to_global_index` | conversion |
| `field_storages(s)`, `siteview`, `sitevector` | multi-field access at a cell |
| `halo_backend(u)`, `communicator(u)`, `is_root(u)` | dispatch and MPI hooks |

`u[i, j]` takes global coordinates, checks ownership and is local-only on MPI:
setup and printing only. Linear indexing and slicing of a halo array are refused.

The portable loop (one function for all three backends):

```julia
function laplacian!(du, u, inv_h2)
    tile_foreach(u) do t
        s, d = tile_parent(u, t), tile_parent(du, t)
        e1, e2 = unit_vector(u, 1), unit_vector(u, 2)
        @inbounds for I in interior_cells(CellRanges(u, t))
            d[I] = (s[I + e1] + s[I - e1] + s[I + e2] + s[I - e2] - 4s[I]) * inv_h2
        end
    end
    return du
end
```

Finite volume: `FaceRanges` + `accumulate_flux_divergence!`. GPU and coloured
sweeps: `CellWindow`, `CellCheckerboard`, `FaceWindow`, `FaceCheckerboard`,
`LinearCellWindow` / `linear_cell_window` (see `docs/src/loops.md`).
Geometry: `cell_geometry`, `Cartesian`/`Polar`/`Cylindrical`/`Spherical`,
`UniformAxis`/`EdgeAxis`, `cell_center`, `cell_volume`, `face_area`.

### Bulk operations, reductions, output

Broadcast (`u .= u .+ dt .* du`), `fill!`, `copyto!`, `axpy!`, `axpby!`, `rmul!`,
`swap!` act on the interior. `sum`, `maximum`, `mapreduce`, `dot`, `norm`, `dots!`
are global: one `Allreduce` on MPI, a tile reduce when threaded.

```julia
plan = DimReductionPlan(u, 2)                # once
profile = reduce!(plan, identity, +, u)      # MaybeHaloArray; is_active(profile) on the root
A = gather_haloarray(u)                      # global interior as an Array (root only on MPI)
append_haloarray!(file_or_name, "u", u)      # HDF5 time series (needs HDF5 loaded)
```

### Solvers

```julia
using LinearSolve                             # HaloCG, HaloMINRES, HaloBiCGStab, HaloGMRES, HaloFGMRES
using KrylovKit                               # + HaloKrylovKit(:gmres | :cg | :bicgstab)
sol = solve(LinearProblem(A, b; u0 = zero(b)), HaloCG())        # always pass u0
solve(prob, FBDF(linsolve = HaloGMRES(), concrete_jac = false)) # implicit ODE: iterative linsolve required

using HaloArrays.Krylov                       # no LinearSolve needed
ws = CGWorkspace(b)
r  = cg!(x, A, b, ws; Pl = nothing, abstol = 0.0, reltol = 1e-8, maxiter = 200)
r.converged, r.iters, r.residual
```

`A` needs `mul!`, a preconditioner `ldiv!`. Krylov.jl's `KrylovJL_*` and
`SimpleGMRES` are not supported (they want a flat `AbstractVector`).

Full manual: `docs/src/guide.md`, `indexing.md`, `loops.md`, `habits.md`,
`solvers.md`, and `docs/src/api/essentials.md` for the short list.

## Conventions

Code style:
- Multiple dispatch, not `isa` ladders or type-testing ternaries. Before adding a
  check, look for a type to dispatch on (a workspace, a preconditioner, `::Nothing`,
  `halo_backend(u)`).
- Fail by construction (a tuple broadcast that errors on a length mismatch) rather
  than added runtime checks.
- Numerical kernels read like the published algorithm: one line per step, the
  textbook's names in margin comments, shared pieces as small named helpers.
- Workspace constructors are inner constructors. `zero(eltype(u))`, never a literal `0`.
- Name helpers by their contract (in-place vs into-`z`), not by a side (left/right).
- Keep LinearSolve/SciML types out of the core kernels; they belong in `ext/`.
- One seam per concern: reductions go through `_global(r, op, communicator(u))`,
  BLAS-1 through `_interior_map!`, single-block code is written once on
  `_SingleBlock` (`LocalHaloArray` and `HaloArray`). Do not add per-backend copies.
- Hot paths are allocation-free; a collection loop must specialise on its function
  argument and unroll over the field tuple (the 0.16.1 fix).
- Aqua's ambiguity check runs in the suite: for vector-like arguments prefer
  `StridedVector` over `AbstractVector` to avoid clashing with StaticArrays methods.
- Deprecated spellings are removed, not kept; breaking changes are marked
  **Breaking** in the changelog and bump the minor version.

Workflow:
- One concern per PR, branched from `main`. Stacked PRs are fine, but `main` takes
  squash merges, so after a merge rebase the next branch with
  `git rebase --onto origin/main` dropping the squashed commits.
- Before pushing: full `Pkg.test`, then MPI with 2 and 4 ranks, locally.
- After pushing, confirm CI actually started (see above).
- Merges are the maintainer's call: squash merge, once CI is green.
- Every PR updates `CHANGELOG.md`; a release PR bumps `Project.toml` and dates the
  changelog section.
