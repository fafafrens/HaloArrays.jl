# Benchmarks

Run benchmarks from the repository root. MPI benchmarks report the maximum
elapsed time across ranks for each sample.

Common options:

- `--ndims=2`
- `--owned-size=128,128`
- `--halo=1`
- `--samples=30`
- `--warmups=5`
- `--csv=path/to/results.csv`

`--local-size` is still accepted by `halo_exchange.jl` as a deprecated alias for
`--owned-size`, but new commands should use `--owned-size`.

## Halo Exchange

```sh
mpiexec -n 4 julia --project=. benchmarks/halo_exchange.jl --owned-size=128,128
mpiexec -n 4 julia --project=. benchmarks/halo_exchange.jl --ndims=3 --owned-size=64,64,64 --methods=blocking,waitall_unsafe,waitall,async_unsafe
```

`blocking` is the public `halo_exchange!` path. The other method names benchmark
compatibility wrappers and implementation variants.

## Reductions

Benchmarks `mapreduce`, `all`, and `any` for `HaloArray`, `LocalHaloArray`,
`ThreadedHaloArray`, and compatible `MultiHaloArray` cases.

```sh
mpiexec -n 4 julia --project=. benchmarks/reductions.jl --owned-size=128,128 --tile-dims=2,2
```

## Gather And HDF5

Benchmarks MPI gather and HDF5 write paths.

```sh
mpiexec -n 4 julia --project=. benchmarks/gather_hdf5.jl --owned-size=64,64 --output=/private/tmp/haloarrays_bench
```

The `--output` option is a path prefix. The script writes files with suffixes
for gather/save and append cases.

## Threaded Operations

Benchmarks threaded halo synchronization, boundary conditions, reductions, and a
heat-step style stencil for `ThreadedHaloArray`.

```sh
julia --project=. benchmarks/threaded.jl --owned-size=128,128 --tile-dims=2,2
```
