# Benchmarks

Run the halo exchange benchmark from the repository root with MPI:

```sh
mpiexec -n 4 julia --project=. benchmarks/halo_exchange.jl
```

Useful options:

```sh
mpiexec -n 4 julia --project=. benchmarks/halo_exchange.jl --ndims=2 --local-size=128,128 --halo=1 --samples=50 --warmups=10
mpiexec -n 4 julia --project=. benchmarks/halo_exchange.jl --ndims=3 --local-size=64,64,64 --methods=blocking,waitall_unsafe,waitall,async_unsafe
```

The script reports wall-clock exchange time using the maximum elapsed time across ranks for each sample.

`blocking` is the public `halo_exchange!` path. The other method names benchmark compatibility wrappers and implementation variants.
