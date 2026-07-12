# Changelog

All notable changes to HaloArrays.jl are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`DimReductionPlan` / `reduce!` / `free!`** — a reusable dimensional
  reduction for distributed `HaloArray`s. `mapreduce_haloarray_dims` used to
  pay two `MPI.Comm_split` collectives plus a `Cart_create` on *every* call
  and leak the communicators embedded in the returned topology (repeated
  calls — e.g. saving a profile each step — eventually exhaust MPI context
  ids). The plan builds the slice and root communicators once with
  `MPI.Cart_sub` (the purpose-built sub-grid call, replacing the hand-rolled
  color/key splits), preallocates the reduced output array, and each
  `reduce!(plan, f, op, u)` then costs a single `MPI.Reduce` — build it once
  outside a hot loop, `free!(plan)` when done. The plan is geometry-only, so
  one plan serves any `f`/`op` over arrays sharing the topology and interior
  size. `benchmark/reduction_plan.jl` measures plan reuse against the one-shot
  path (3.5–3.8× per call at 4 ranks).
- **`sum(u; dims=…)` (and `prod`/`maximum`/`minimum`/`mapreduce`) now work on a
  distributed `HaloArray`** instead of throwing: the `dims=` keyword runs a
  transient `DimReductionPlan` — built, used, and released within the call —
  and returns a fresh reduced array each time, with the reduced dimensions
  dropped, on the coordinate-0 slice of the topology (a `MaybeHaloArray`),
  matching `mapreduce_haloarray_dims` semantics rather than Base's
  kept-singleton-dims shape. **The result owns its sub-communicator**:
  `free!(result)` releases it (optional; reclaimed at `MPI.Finalize`), keeping
  communicator use bounded when reducing in a loop. `mapfoldl`/`mapfoldr` with
  `dims=` still throw (a cross-rank slice reduction reorders the fold), and
  `init=` is rejected (it would be folded in once per rank).
- **`dims=` reductions are backend-preserving**: `LocalHaloArray` returns a
  reduced `LocalHaloArray`; `ThreadedHaloArray` returns a reduced
  `ThreadedHaloArray` whose tile layout is the original layout with the
  reduced dimensions dropped (same thread backend — and the assembly runs in
  parallel over the reduced tiles through it, race-free since each task owns
  one output tile); collections
  (`MultiHaloArray`, `ArrayOfHaloArray`, any backend) reduce every field and
  rebuild the same collection kind. Only the distributed backend wraps in
  `MaybeHaloArray` — the one case where the result may be absent on a rank.
  `is_active`/`interior_view`/`free!` behave uniformly on all of them
  (`interior_view` now passes through `MaybeHaloArray`, active-guarded;
  `free!` is a safe no-op on serial results), so backend-generic code needs
  no branches. `mapreduce_haloarray_dims` gained the matching methods (and
  `mapreduce_mhaloarray_dims` now covers both collection kinds through it).
  Breaking detail: `mapreduce(f, op, u::LocalHaloArray; dims)` previously
  leaked Base's semantics (a plain `Array` with kept singleton dims).
- **`examples/poisson/cg_fused.jl`** — the performance counterpoint to the
  coordinate-free Krylov solvers: the same CG with its six per-iteration array
  sweeps fused into three (`p·Ap` accumulated inside the stencil sweep; the
  `x`/`r` updates and `‖r‖²` in one pass per tile). Fewer sweeps and half the
  task barriers make the threaded backend the fastest configuration (1.3–1.4×
  reproducibly on a laptop at 1024², more when thread placement punishes the
  unfused version); one `Allreduce` hook keeps it MPI-correct, and the script
  self-checks against the textbook `cg!`. Runs in the CI smoke tests.

- **`DimReductionPlan` is backend-generic**: `DimReductionPlan(u, dims)` +
  `reduce!` + `free!` now compile and run unchanged on `LocalHaloArray` and
  `ThreadedHaloArray` too (a lightweight serial plan holding the preallocated
  reduced output; `free!` is a no-op and the plan stays usable), so hot-loop
  code that hoists a plan is write-once across backends. A plan's output
  element type is fixed at construction — `eltype(u)` unless overridden with
  the new `output_eltype` keyword — and promoting reductions against it throw
  a descriptive `ArgumentError` instead of an `InexactError`/MPI type
  mismatch. The one-shot forms are now literally transient plans (built with
  the `Base.promote_op`-predicted element type, one `reduce!`, released), so
  they promote like Base on **every** backend: `sum(::Bool array; dims=…)`
  counts in `Int` on Local, Threaded, and MPI alike. `reduce!` also
  normalizes `Base.add_sum`/`mul_prod` itself, so driving a plan with Base's
  internal reducers works on non-Intel MPI just like the keyword forms.

### Changed
- **`mapreduce_haloarray_dims` is reimplemented over the transient
  `DimReductionPlan`** (identical results and return type): communicator
  construction drops from two `Comm_split`s plus a `Cart_create` to two
  `Cart_sub`s, the reduce-side communicator is freed within the call instead
  of leaking, and the one remaining communicator is owned by the returned
  array (`free!`-able, see above). The internal `subcomm_for_slices`,
  `root_topology_multi`, and `coords_to_color_multi` helpers this replaced
  are removed.
- The scalar (no-`dims`) `mapreduce` path normalizes `Base.add_sum`/
  `Base.mul_prod` to `+`/`*` before building the `MPI.Op`, so the builtin
  `MPI_SUM`/`MPI_PROD` apply — required on non-Intel architectures, where
  MPI.jl cannot register custom reduction operators (previously
  `sum(u; dims=:)` errored there).
- `CartesianTopology` prints compactly (`dims`/`coords`/`periodic`) instead of
  dumping raw communicator handles and neighbor tables into every `HaloArray`
  display.
- **GPU examples synchronize once per sweep/step instead of after every kernel
  launch** (`heat/cpu_vs_gpu_2d.jl`, `tutorials/gpu.jl`, both `phi4_metal` and
  `su2_wilson_metal`): launches on one backend queue execute in order, so only
  the host-read boundary needs a sync — measured 2× on an M2 GPU, bit-identical
  results. The GPU tutorial now teaches the rule.
- `benchmark/stencil.jl` tiles along the last dimension (`dims=(1,nt)`), the
  layout the `ThreadedHaloArray` docstring recommends (its threaded halo
  refresh measures ~3.7× cheaper than the first-dimension split).
- **The two benchmark directories are unified into `benchmark/`** (the Julia
  convention): the quick-start throughput harnesses and the former
  `benchmarks/` CLI micro-suite now live together under one environment and
  one README. The micro-suite was brought up to the 0.3.0 API (`versors` →
  `unit_vector`; `gather_hdf5.jl` now loads the HDF5 weak dependency it
  needs), and every script was smoke-run (serial, threaded, 2-rank MPI, and
  Metal).
- **New `benchmark/checkerboard_inout.jl`** compares a checkerboard stencil
  sweep in-place vs out-of-place vs a single-pass Jacobi, on both CPU and Metal:
  the single-pass update is ~2–3× faster than the two-pass red-black sweep on
  both backends (one launch vs two on the GPU; contiguous SIMD vs stride-2
  access on the CPU), quantifying the cost of the coloring that in-place
  updates require.
- **Docs: a "Choosing between OhMyThreads and Polyester" section** in the guide,
  with the measured guidance (default OhMyThreads for coarse per-tile work;
  `PolyesterBackend` for many thin per-tile ops, where its `@batch` pool avoids
  the task-spawn cost — measurably faster and near-allocation-free).

### Changed (breaking)
- **Collection `dims=` reductions use collection-global coordinates and can
  reduce the field axis.** A `MultiHaloArray`/`ArrayOfHaloArray` presents as an
  array with axes `(field…, spatial…)`, but `sum(c; dims=d)` previously
  forwarded `d` to each field's *spatial* reduction — so the field axis was
  unreachable and `dims` was off by the field-axis count versus `size(c)`.
  Now `dims` is interpreted in the collection's own coordinates: field axes
  (`1:F`) reduce **locally** (an elementwise fold across fields — no
  communication, no plan, a bare result), collapsing all fields into one
  `HaloArray` (`MultiHaloArray` drops the names) or a partial set into a
  smaller collection; spatial axes (`F+1:D`) reduce per field as before.
  `MaybeHaloArray` wraps the result (outermost) only when a spatial axis was
  reduced on MPI. Migration: shift spatial `dims` up by the number of field
  axes — e.g. for 2-D fields `sum(c; dims=2)` (old spatial-1) becomes
  `sum(c; dims=3)`. `mapreduce_mhaloarray_dims` and the collection form of
  `mapreduce_haloarray_dims` follow the same coordinates.
- **`DimReductionPlan` extends to collections.** `DimReductionPlan(c, dims)`
  returns a plan that classifies the axes once and holds one reused per-field
  array plan for the spatial axes, so a hoisted collection reduction rebuilds
  no MPI communicators; the collection one-shot (`sum(c; dims=…)`) is a
  transient such plan, mirroring the array path.

### Fixed
- **`mapfoldl`/`mapfoldr` with `dims=` throw a clean error on every backend**:
  the guard covered only `ThreadedHaloArray`, so on a `LocalHaloArray` the
  call fell through to Base's dims-less `mapfoldl`, producing an obscure
  "no method matching mapfoldl(…; dims)" `MethodError` instead of the intended
  "folds with `dims=` are not supported" `ArgumentError`.
- **`interior_range(m::MaybeHaloArray)` is active-guarded** like
  `interior_view`: an inactive reduction result used to return valid-looking
  ranges into its placeholder data (silently reading zeros) while
  `interior_view` correctly errored — the two accessors now agree.
- **`free!` on a primary `HaloArray` gives a descriptive error** instead of a
  `MethodError`: `free!` releases only the sub-communicator of a reduction
  result (a `MaybeHaloArray`); calling it on a bare, unwrapped `HaloArray`
  (which owns its topology) now explains that rather than failing on dispatch.
- **`dims=` reductions keep GPU-backed arrays on the device**: the reduced
  output (and, on MPI, its exchange buffers) is allocated with `similar` on
  the source's parent instead of the CPU `zeros` constructors, and the
  result assembly uses broadcast assignments instead of strided `copyto!`.
  Previously `sum(adapt(JLArray, u); dims=2)` threw "Scalar indexing is
  disallowed" and any GPU dims-reduction would have landed on the host.
- **`ThreadedHaloArray` dims-reductions returned silently wrong values**: the
  generic tile driver combined the per-tile reduced arrays with `op` across
  *all* tiles — elementwise-mixing tiles that lie along kept dimensions (e.g.
  column sums over a `(2,1)` tiling summed the two tile halves together). The
  tiled backend now reduces each tile and assembles: tiles along removed
  dimensions combine with `op`, tiles along kept dimensions land at their
  global offset. `mapfoldl`/`mapfoldr` with `dims=` on a tiled array throw
  instead of going through the same broken combine.
- **Three examples never ran their simulation**: `relativistic_hydro_mu0_2d`,
  `mu0_3d` and `Tmu_3d` had the driver call commented out, so the CI smoke
  tests only checked that they parse. The drivers auto-run again.
- The examples environment could not resolve against HaloArrays 0.3.0
  (`examples/Project.toml` required DiffEqBase 7; the package compat says 6).

## [0.3.0]

### Changed (breaking)
- **One initializer, one callback.** `fill_from_local_indices!` was removed: a
  local-index fill makes the global field depend on the domain decomposition,
  contradicting the backend-agnostic promise (`interior_view(u) .= …` covers the
  rare legitimate use). `fill_from_global_indices!` is the single initializer;
  its callback receives the **index tuple** `f(I)` (the docstring previously
  showed a splatted form that never worked) and it returns `u`.
- **Uniform returns.** Every public mutating driver now returns its array on
  every backend — `halo_exchange!`, `start_/finish_halo_exchange!`,
  `boundary_condition!` (whole-array, per-face, collections, threaded) and the
  `_threads!` variants. The MPI methods previously returned `nothing`, breaking
  backend-agnostic chaining.
- **`unit_vector` is the single name for Cartesian unit steps** — new methods on
  halo arrays and `Val(N)` (`unit_vector(u[, dim])`) absorb the internal
  `face_offset` (deleted) and the private `versors` the examples used to reach for.
- **View helpers renamed and reordered**: `get_send_view(s, d, u[, tile])` →
  `edge_view(u, s, d[, tile])` and `get_recv_view(…)` → `ghost_view(u, s, d[, tile])`
  — array first like every other helper, tile last, and names that are correct in
  both of their roles (boundary conditions *and* the MPI exchange, which sends the
  edge and receives into the ghost). `tile = nothing` means "whole array", so
  backend-generic code can pass a tile handle straight through.
- **`get_comm` → `communicator`, `isactive` → `is_active`** — the last `get_`
  holdouts and the one predicate that didn't follow the package's underscored
  naming.
- **Coupled boundary conditions: one method, every backend.** The canonical
  signature is now `apply_coupled_bc!(bc, state, side, dim, tile)` with
  `tile === nothing` on Local/MPI fields and the boundary tile id on threaded
  fields — mirroring `FunctionBC`'s backend-uniform design. The legacy split
  4-arg / per-tile 5-arg methods still dispatch.
- **Every per-tile operation is written once over the tile drivers.** Two tiny
  drivers over the one-tile decomposition (single-block arrays are a one-tile
  decomposition; `ThreadedHaloArray` splits its tiles across the thread
  backend) — `_foreach_tile` and its reduce sibling `_mapreduce_tile` — replace
  the per-backend method pairs: `fill!`, `copyto!`, `fill_from_global_indices!`,
  the BLAS-1 family (`rmul!`/`lmul!`/`axpy!`/`axpby!` and
  `swap!`/`rotate!`/`reflect!`), and the reductions' local parts
  (`mapreduce`/`mapfoldl`/`mapfoldr`, `any`/`all`, `sum`, `norm`, `dot`). The
  MPI `HaloArray` reductions are now literally `Allreduce(local part)`, so each
  reduction's local math exists in exactly one place. A/B benchmarked: values
  bit-identical, Local path 0-alloc and time-identical, threaded within noise.
- **`copyto!` between halo arrays now validates shape on every backend** — one
  uniform guard (global size, tile layout, per-tile padded storage). This
  subsumes the old threaded-only checks and newly rejects copies between arrays
  with **different halo widths**, which the single-block path previously
  performed as a silently misaligned raw-storage copy.
- **Boundary-condition ghost-fill kernels pair `ghost_view` with `edge_view`.**
  Each is a single fused broadcast: Reflecting/Antireflecting mirror via a
  reversed-range view (the reversal is side-independent in slab-local
  coordinates), Repeating broadcast-expands the wall slice across the ghost
  thickness, and local Periodic is one copy from the opposite side's edge —
  visibly the same operation as the halo exchange (`_periodic_into!` deleted).
  On GPU parents each face is one kernel launch instead of one per halo layer.

### Deprecated
- `get_send_view`, `get_recv_view` (all arities, old argument order), `get_comm`,
  and `isactive` remain as `@deprecate` shims; they will be removed in 0.4.

### Fixed
- **Implicit OrdinaryDiffEq solves on distributed states.** OrdinaryDiffEq wraps
  every iterative linear solver with error-weight preconditioners
  `Diagonal(weight)` where `weight` is a halo array; LinearAlgebra's generic
  diagonal kernels apply them by scalar-indexing *global* indices — fine on one
  rank by accident, an error on 2+. New elementwise `mul!`/`ldiv!` methods for
  `Diagonal`-of-halo-array route through the interior broadcast (no
  communication, every backend).
- **`iterate(::ThreadedHaloArray)` returned the indices, not the values**, so
  `collect`, comprehensions, and generic `copyto!` silently produced `1, 2, 3, …`
  regardless of contents.
- **CI actually runs the distributed implicit regression test** — the MPI job now
  installs OrdinaryDiffEq/LinearSolve/Krylov; previously the runtests gate
  silently skipped `test_mpi_implicit.jl` while the job stayed green.
- **`norm(u, p)` honors Base's contract for the special exponents**: `p = -Inf`
  (minimum `|x|`) and `p = 0` (count of nonzeros) previously went through the
  generic `abs(x)^p` branch and returned garbage; `p = 1` no longer pays a float
  power per element. Collection p-norms mirrored. (The p-norm stays expressed in
  global `mapreduce` vocabulary, which is what makes the MPI p-norm correct by
  inheritance.)

### Added
- **`benchmark/` harness** — stencil throughput (Local vs Threaded, Mcell/s) and
  MPI exchange cost vs message size including how much the split
  `start_/finish_halo_exchange!` overlap hides (`HALO_BENCH_QUICK=1` for smoke runs).
- **`Diagonal`-of-halo-array operators** (`mul!` 3/5-arg, `ldiv!` 2/3-arg) —
  Jacobi/error-weight preconditioning works on every backend.
- **`FieldCollection` is exported** (the concrete type behind the
  `MultiHaloArray`/`ArrayOfHaloArray` aliases).
- **Device-array test coverage for the reductions** — `sum`/`norm`/`dot`/
  `mapreduce`/`fill!`/`copyto!` on a JLArray-backed array (scalar indexing
  forbidden), locking the generic non-`Array` fallbacks of
  `_interior_acc`/`_interior_dot`; previously only the BLAS-1 ops were
  device-tested.

## [0.2.0]

### Added
- **`FunctionBC`** — a custom per-field boundary condition from a plain function,
  running inside `synchronize_halo!` like a built-in and on every backend (single,
  MPI, threaded). Called per face as `f(ghost, edge, side, dim, hw, origin)`, where
  `origin` is the global `CartesianIndex` of the ghost slab — so position-dependent
  conditions are a broadcast that stays correct under MPI/thread decomposition and
  runs on GPU. One mechanism now covers value-, gradient-, and position-based BCs;
  cross-field conditions remain `apply_coupled_bc!`.
- **Multi-GPU MPI example** (`examples/heat/multigpu_mpi_2d.jl`): one MPI rank per
  GPU, a device-resident `HaloArray`, and GPU-to-GPU **CUDA-aware-MPI** halo
  exchange — **validated on CINECA Leonardo** (1 and 4× A100, global `‖u‖₂`
  bit-identical to the CPU reference).
- **`examples/heat/RUNNING_ON_LEONARDO.md`** — a tested HPC deployment recipe
  (system OpenMPI + system parallel HDF5 + CUDA local toolkit + `srun --mpi=pmix_v3`).
- **`Adapt.jl` support** — move a `HaloArray` between host and device (`cu(halo)` /
  `adapt`), with device-following halo buffers.
- **LinearSolve / Krylov extension** — matrix-free solvers that operate directly on
  halo arrays as coordinate-free vectors: `HaloKrylov`, `HaloCG`, `HaloBiCGStab`,
  `HaloMINRES`, `HaloGMRES`.
- **`norm` for `MultiHaloArray` / `ArrayOfHaloArray` / `MaybeHaloArray`.**

### Changed
- **Kernel-region types renamed** for clarity ("region" now reads as a *range*
  concept; these are positioned launch windows): `CellKernelRegion`→`CellWindow`,
  `FaceKernelRegion`→`FaceWindow`, and the 2-colored (red-black) variants
  `ColoredCellKernelRegion`→`CellCheckerboard`, `ColoredFaceKernelRegion`→
  `FaceCheckerboard`. The cell/face range and window accessors were also made more
  idiomatic — the `get_` prefix was dropped (`get_send_view`/`get_recv_view`/
  `get_comm` keep theirs) and the separate colored accessors were folded into the
  base ones via an optional `color` argument (dispatch). The family is now
  `interior_cells(ranges[, color])`, `interior_faces(ranges, dim[, color])`,
  `interior_cell_window`/`interior_face_window(ranges[, …][, color])` (a `color`
  returns the checkerboard variant), plus `unit_vector(ranges, dim)`.
- **Face loops simplified to one accessor.** The separate `left_face`/
  `internal_face`/`right_face` (and their `*_window`) were collapsed into a single
  `interior_faces(ranges, dim)` — every face touching the interior — and the
  flux-divergence loop now scatters each face's flux onto *both* adjacent cells
  (the two boundary faces also write a ghost cell, which is in-bounds and harmless,
  so the per-face owned-side flags on `FaceWindow`/`FaceCheckerboard` are gone).
  `accumulate_flux_divergence!` keeps the same signature.
- **HDF5 is now a weak dependency** (`HaloArraysHDF5Ext`): `using HaloArrays` no
  longer pulls in HDF5 (and its MPI-built JLLs, which clash with a system
  CUDA-aware MPI). The I/O API loads only when you `using HDF5`.
- Examples use a single KernelAbstractions path for both CPU and GPU (removed the
  hand-written CPU scalar loops).
- `to_bc` (boundary-condition normalization) now uses multiple dispatch instead of
  an `if`/`isa` ladder, so a new `:symbol` shortcut can be registered by an
  extension via `to_bc(::Val{:name})` without editing the package.
- README trimmed to essentials.

### Performance
- Fast contiguous `@simd` interior reductions for `sum`/`dot`/`norm` (~5× on the
  matrix-free Krylov path), gated on `::Array` parents so GPU parents keep a
  device-side fallback.
- BLAS-1 contiguous SIMD kernels for `axpy!`/`axpby!`/`rmul!`/`lmul!` on `Array`
  parents; `swap!`/`rotate!`/`reflect!` made GPU-safe.
- Collection and `dot` reductions are now zero-allocation.

### Fixed
- Closed an `O(N)` hot-path allocation leak in collection/`Maybe` `norm`.
- Multi-GPU example world-age error (load the GPU package at top level).
- Keep interior reductions GPU-safe (SIMD only for `Array` parents).
- CI: documentation `@autodocs` source-file selection and the MPI-test environment
  updated for the HDF5 weak-dependency move; bumped Node-20 GitHub Actions to
  Node-24 versions.

## [0.1.0]

- Initial version: `LocalHaloArray`, `ThreadedHaloArray`, and MPI `HaloArray`
  behind one halo-exchange API; multi-field containers; boundary conditions
  (periodic, reflecting, antireflecting, repeating, custom, coupled); global
  reductions; `gather` and HDF5 output; OrdinaryDiffEq integration; thread-backend
  abstraction (OhMyThreads / Serial / Polyester); KernelAbstractions GPU path.

[0.2.0]: https://github.com/fafafrens/HaloArrays.jl/releases/tag/v0.2.0
[0.1.0]: https://github.com/fafafrens/HaloArrays.jl/releases/tag/v0.1.0
