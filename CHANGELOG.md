# Changelog

All notable changes to HaloArrays.jl are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

### Added
- **`benchmark/` harness** — stencil throughput (Local vs Threaded, Mcell/s) and
  MPI exchange cost vs message size including how much the split
  `start_/finish_halo_exchange!` overlap hides (`HALO_BENCH_QUICK=1` for smoke runs).
- **`Diagonal`-of-halo-array operators** (`mul!` 3/5-arg, `ldiv!` 2/3-arg) —
  Jacobi/error-weight preconditioning works on every backend.
- **`FieldCollection` is exported** (the concrete type behind the
  `MultiHaloArray`/`ArrayOfHaloArray` aliases).

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
