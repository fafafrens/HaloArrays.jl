# API essentials

A short tour of the names most programs need. Everything here is documented in
full on the reference pages that follow — this page is the "start here" so the
common API isn't buried among the advanced kernel and threading internals.

## Arrays

- Single-process: [`LocalHaloArray`](@ref)
- Distributed (MPI): [`HaloArray`](@ref) on a [`CartesianTopology`](@ref)
- Shared-memory tiles: [`ThreadedHaloArray`](@ref)
- Several fields on one grid: [`MultiHaloArray`](@ref) — built per backend with
  [`LocalMultiHaloArray`](@ref) / [`ThreadedMultiHaloArray`](@ref) — or, for
  index- rather than name-addressed fields, [`ArrayOfHaloArray`](@ref)

## Reading and sizing the data

- [`interior_view`](@ref) — a view of the ghost-free cells this process owns
  (read or write the owned region directly)
- Sizes: [`interior_size`](@ref) (owned), [`global_size`](@ref) (whole domain),
  [`storage_size`](@ref) (padded backing); index ranges with
  [`interior_range`](@ref)
- `parent` — the field container of a collection; `field_storages` —
  the raw padded backing array of every field, for ghost-offset stencils
  ([`field_storages`](@ref))

## Filling and updating

- `fill!(u, x)` sets the interior only; broadcast (`u .= …`) and reductions
  (`sum`, `maximum`, `mapreduce`, …) all operate on the interior
- Initialise from a coordinate function with
  [`fill_from_global_indices!`](@ref)

## Ghost cells

- [`synchronize_halo!`](@ref) fills every ghost cell — applying boundary
  conditions and exchanging between ranks/tiles — in one call
- Boundary-condition types: [`Periodic`](@ref), [`Reflecting`](@ref),
  [`Antireflecting`](@ref), [`Repeating`](@ref), and [`NoBoundaryCondition`](@ref)
  (opt out so you can fill an edge yourself)

## Finite-volume kernels

- [`FaceRanges`](@ref) + [`accumulate_flux_divergence!`](@ref) — the
  conservative face-flux update in one call
- [`CellRanges`](@ref) — owned-cell iteration for source terms and updates

## Distributed (MPI)

- [`CartesianTopology`](@ref) — the process decomposition
- [`gather_haloarray`](@ref) — collect a distributed array onto one rank

## HDF5 output

- [`create_haloarray_output_file`](@ref),
  [`write_haloarray_timestep!`](@ref), and
  [`gather_and_save_haloarray`](@ref)

---

The reference pages below document **every** exported name, grouped by area —
types, the array/layout/reduction core, halo exchange and boundary conditions,
and the loop & kernel-region machinery for GPU/threaded stencils.
