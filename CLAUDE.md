# HaloArrays.jl

Julia package (v0.16.1, Julia ≥ 1.10, repo `fafafrens/HaloArrays.jl`): arrays with
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

`archive/old/` is ignored (`.ignore`). `LocalPreferences.toml` pins this machine's
Homebrew OpenMPI and HDF5; do not commit changes to it for other machines.

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

## Benchmark baseline

Quick-start harnesses on this machine (Apple M4 Max, 12 P + 4 E cores, Julia
1.13.3, v0.16.1, 2026-11-24; single runs, machine not idle). Full tables in
`benchmark/README.md`; use these to spot a regression, not to claim scaling.

- `stencil.jl` at 2048², Mcell/s: Local 1541; Threaded 1379 (1 tile), 2483 (2),
  4861 (4), 9286 (8), 9149 (16). At 1024²: Local 1678, 8 tiles 8155.
- `run_mpi.jl`, blocking exchange of a 64³ block: 58 µs (2 ranks), 74 µs (4),
  135 µs (8); 128³: 185 / 288 / 529 µs. The split-exchange `hidden` gain is noise here.

CLI micro-suite, same machine and date (medians; every case has a reference
line in `benchmark/README.md`):

- MPI, 4 ranks, 128²: blocking exchange 34 µs, split 38 µs; `synchronize_halo!`
  36 to 38 µs, 0 B; `mapreduce` 51 µs. Gather 64²: 183 µs; HDF5 append 2.6 ms.
- `synchronize_halo!` at 128², any boundary mode, 0 B: local 0.13 to 0.17 µs,
  threaded 2×2 tiles 0.46 µs.
- Thread backends (4 threads, 256²): task-based `threads`/`ohmythreads` cost a
  fixed 13 to 15 µs and 1 to 3 KB per call; `serial` and `polyester` are 0.3 to
  9 µs. Only `mapreduce` gains from tasks (serial 39 µs, polyester 11 µs).
- Threaded halo sync vs serial: the spawn cost is now 15 to 20 µs (the README's
  older table says 130 to 180), so threading wins from about 20 µs of exchange
  work: 2048² halo 3 with 64 tiles 59 → 26 µs, 128³ halo 2 98 → 39 µs.
- Full steps: ideal hydro 128² local 23 ms, threaded 11 ms (15 ms before the
  one-sweep collection change below), MPI 4 ranks 10 ms;
  relativistic 1024² serial 160 ms/step, 8 threads 6.8×, 8 ranks 6.0×;
  cylindrical 5.8×; heat 512² local 1.56 ms, 2 tiles 1.12 ms, 2 ranks 0.89 ms.
- Metal: `CellCheckerboard` compressed launch 1.3× (128²) to 3.4× (1024²) over
  the naive launch; linear vs Cartesian full pass 1.4 to 1.5× in 3D/4D and a
  noisy 1.4 to 2.0× in 2D (the older README table says up to 3.5×);
  single-launch Jacobi gains 1.2 to 1.9× up to 512² (noisy) and nothing above.
  CPU Jacobi vs checkerboard 1.9 to 2.8×. Run twice (scratch env, then
  `--project=examples`): agreement within a few percent from 512² up, small
  GPU sizes vary by tens of percent.

Things that bite when running them:
- `benchmarktools_times!` in `benchmark/common.jl` passes a finite `seconds=1.0e6`:
  BenchmarkTools 1.8 converts `seconds / sample_time` to an `Int`, so `Inf`
  throws `InexactError`. Keep it finite.
- The `examples` environment was re-resolved on 2026-11-24 (Julia 1.13,
  HaloArrays 0.16.1 at `..`, Metal 1.11, KernelAbstractions 0.9.48, LinearSolve 5,
  OrdinaryDiffEq 7.8). Its `Manifest.toml` is gitignored; if it goes stale again,
  `Pkg.update()` there, then `git checkout examples/Project.toml`, because Julia
  1.13 reorders the file and appends `[sources]`. `Pkg.status` then warns that
  the project changed since the manifest was resolved; that is harmless.
- Launch MPI runs through `MPI.mpiexec()` of the same project, as `run_mpi.jl` does.
- `Pkg.develop(path=".")` under Julia 1.13 appends a `[sources]` table to
  `benchmark/Project.toml`; do not commit it (`git checkout benchmark/Project.toml`).
- The benchmark environment has no `LocalPreferences.toml`, so it runs MPICH from
  `MPICH_jll`, not the Homebrew OpenMPI the root project is pinned to.
- `stencil.jl` tiles as `(1, nthreads)` and silently skips the threaded case when
  the thread count does not divide the size (12 threads on 256/1024/2048).
- Use at most 8 to 12 threads: beyond the 12 performance cores the efficiency
  cores drag the result down (16 threads at 256²: 502 Mcell/s).

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
- Force-pushes are Eduardo's to run (`git push --force-with-lease`). Merge
  (`gh pr merge --squash`) only when CI is green and he has said to merge.
- Every PR updates `CHANGELOG.md`; a release PR bumps `Project.toml` and dates the
  changelog section.

## Current state (2026-11-24)

`main` is at 0.16.1 (last merge #105, a CompatHelper bump). Three PRs are open,
see below. Recent history:

- #104 (0.16.1): halo refresh of collections is allocation-free however it was first compiled.
- #100, #103: manual pages *Index spaces*, *Cell and face loops*, *Good habits*.
- #99, #101 (0.16.0): consolidation. Dispatch holes closed, one communicator seam
  for reductions, one request per face in the MPI exchange, BLAS-1 as one kernel,
  one constructor per collection family with `storage=` everywhere, deprecated
  spellings removed.
- #98: boundary conditions as per-cell rules `bc(u, k)`. **Breaking:** `FunctionBC`
  takes the rule, not the old slab function.
- #97: KernelAbstractions extension; on GPU storage all rule faces of a block are
  filled in one launch.
- #93 to #96 (0.14, 0.15): `Mirror(sign)`, `Extrapolation(order)`, linear-index cell windows.
- #85 to #92 (0.13): KrylovKit linsolve, right preconditioning, FGMRES,
  `HaloArrays.Krylov` submodule, `dots!` + CGS2.

Decisions that stand:
- `synchronize_halo!` is the only place ghosts are filled; the split pair is the
  exchange alone and needs `boundary_condition!` afterwards.
- Positional boundary-condition constructor forms are gone; use `boundary_condition=`.
- `dot`/`norm` on a multi-field state do one `Allreduce`, not one per field.
- `lmul!(s, x)` multiplies on the left (matters for non-commutative cell types).
- MPI `eachindex` is the owned global block.

## Open PRs (opened 2026-11-24)

Three branches off `main`, one concern each:

1. `collection-tile-sweep`: one tile sweep per bulk write on a threaded
   collection. `_foreach_tile`/`_mapreduce_tile` and the public
   `tile_foreach`/`tile_mapreduce` accept a collection; `fill!` is one method on
   `HaloArrayField` with a per-tile `_fill_tile!`; collection broadcast dispatches
   on `halo_backend(dest)` (`_run_fields!`, `_tile_fields!` in
   `src/interior_broadcast_marray.jl`); the hydro example uses `tile_foreach(u)`.
   Tests in `test/test_thread_backend.jl` (a `CountingBackend` asserts one sweep).
2. `benchmark-timer-fix`: finite `seconds` for the BenchmarkTools timer in
   `benchmark/common.jl`.
3. `benchmark-refs-claude-md`: this file, the reference numbers in
   `benchmark/README.md`, the Performance section in `README.md`. Its hydro
   numbers assume branch 1, so merge that one first.

Decided on 2026-11-24, not to do for now:
- A generic field walker (`_each_leaf`) porting every collection operation to
  one sweep: judged too large and unproven. Port a single operation, with its
  own measurement, only when a benchmark shows it matters.
- A size threshold so small tiles skip the task spawn on `ThreadsBackend`:
  measured (the task backends have a ~13 µs floor; `fill!`/broadcast cross over
  between 256² and 1024² on 4 tiles, `mapreduce` between 64² and 256²) but left
  unimplemented. `PolyesterBackend` has no such floor and is the answer for
  small tiles today.

Still per field on a threaded collection: the reductions (`_local_sum`,
`_local_dot`, `mapreduce`, `any`/`all`), `copyto!` between collections,
`swap!`/`rotate!`/`reflect!`. The other examples (`finite_volume`, `lattice`)
still call `OhMyThreads.tforeach` directly.

## Open tasks

Carried from the notes of 2026-11-05 and not re-verified against 0.16.1; check
each against the code before starting on it.

Left over from the #99 review:
- Tiled GPU `gather` still scalar-indexes.
- Coupled boundary conditions as rules over a tuple of fields.
- Known hole: collection `dot`/`dots!` field-count mismatch is not caught for a
  1-field collection.

Solver roadmap, in priority order:
1. Chebyshev polynomial preconditioner (pairs with FGMRES).
2. DCGS2 delayed reorthogonalisation.
3. Fused threaded `dots!` sweep.
4. One-sided `Pr` for CG.
5. Preconditioned MINRES.
6. `precs` hook for OrdinaryDiffEq.
