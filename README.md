# HaloArrays.jl

Distributed halo array utilities for Julia.

## Status
This repository now contains a Julia package skeleton with tests. Many tests require MPI and are gated behind environment flags to keep CI green.

## What’s here
- HaloArray and MultiHaloArray types, broadcasting over interiors
- Halo exchange implementations (waitall, async, unsafe variants)
- Boundary conditions (Reflecting, Antireflecting, Repeating, Periodic)
- CartesianTopology helpers
- Reductions and gather
- HDF5 parallel I/O helpers

## Running tests
- Unit tests (no MPI):
  ```bash
  HALOARRAYS_RUN_UNIT_TESTS=true julia --project=. -e 'using Pkg; Pkg.test()'
  ```
- MPI tests (require mpiexec):
  ```bash
  HALOARRAYS_RUN_MPI_TESTS=true mpiexec -n 4 julia --project=. -e 'using Pkg; Pkg.test()'
  ```


## Getting started
- Clone the repository:
  ```bash
  git clone https://github.com/fafafrens/Haloarray.git
  cd Haloarray
  ```
- Use or explore the Julia scripts directly. For example, from a Julia REPL in this directory you can include specific files:
  ```julia
  include("haloarray.jl")
  # or
  include("haloarrays.jl")
  ```

## Contributing
Issues and pull requests are welcome. Please open an issue to discuss major changes.
please help me :-)

## License
MIT — see [LICENSE](./LICENSE).

