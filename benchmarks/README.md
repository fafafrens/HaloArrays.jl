# Benchmarks

Run the halo exchange benchmark from the repository root with MPI:

```sh
mpiexec -n 4 julia --project=. benchmarks/halo_exchange.jl
```

Useful options:

```sh
mpiexec -n 4 julia --project=. benchmarks/halo_exchange.jl --ndims=2 --local-size=128,128 --halo=1 --samples=50 --warmups=10
mpiexec -n 4 julia --project=. benchmarks/halo_exchange.jl --ndims=3 --local-size=64,64,64 --methods=waitall_unsafe,waitall,async_unsafe
```

The script reports wall-clock exchange time using the maximum elapsed time across ranks for each sample.

By default the benchmark excludes the legacy one-side `exchange` and `wait` methods because they can hang in MPI configurations where the benchmark posts each side sequentially. They are still available through `--methods=exchange,wait` when investigating those routines directly.
