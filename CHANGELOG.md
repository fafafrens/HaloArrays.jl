# Changelog

All notable changes to HaloArrays.jl are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`examples/poisson/cg_fused.jl`** — the performance counterpoint to the
  coordinate-free Krylov solvers: the same CG with its six per-iteration array
  sweeps fused into three (`p·Ap` accumulated inside the stencil sweep; the
  `x`/`r` updates and `‖r‖²` in one pass per tile). Fewer sweeps and half the
  task barriers make the threaded backend the fastest configuration (1.3–1.4×
  reproducibly on a laptop at 1024², more when thread placement punishes the
  unfused version); one `Allreduce` hook keeps it MPI-correct, and the script
  self-checks against the textbook `cg!`. Runs in the CI smoke tests.

### Changed
- **GPU examples synchronize once per sweep/step instead of after every kernel
  launch** (`heat/cpu_vs_gpu_2d.jl`, `tutorials/gpu.jl`, both `phi4_metal` and
  `su2_wilson_metal`): launches on one backend queue execute in order, so only
  the host-read boundary needs a sync — measured 2× on an M2 GPU, bit-identical
  results. The GPU tutorial now teaches the rule.
- `benchmark/stencil.jl` tiles along the last dimension (`dims=(1,nt)`), the
  layout the `ThreadedHaloArray` docstring recommends (its threaded halo
  refresh measures ~3.7× cheaper than the first-dimension split).

### Fixed
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
