# Examples

This folder contains example scripts that demonstrate using HaloArrays.jl to solve the heat equation in 1D, 2D, and 3D with periodic boundaries and parallel HDF5 I/O.

Run prerequisites
- Julia with the project instantiated at the repo root:
  - julia --project=. -e 'using Pkg; Pkg.instantiate()'
- MPI installed (OpenMPI on Linux/macOS). On Ubuntu runners, apt-get install -y libopenmpi-dev openmpi-bin.

General run pattern
- Use mpiexec to launch multiple ranks and run the script.
- From the repository root:
  - 1D: mpiexec -n 4 julia --project=. examples/tes_heat.jl
  - 2D: mpiexec -n 4 julia --project=. examples/tes_heat_2d.jl
  - 3D: mpiexec -n 4 julia --project=. examples/tes_heat_3d.jl

Notes
- Scripts create result.h5 (gather to root) and result_par.h5 (parallel write) datasets named "temp".
- If the machine has fewer cores than ranks, add --oversubscribe to mpiexec:
  - mpiexec --oversubscribe -n 4 julia --project=. examples/tes_heat_2d.jl
- The example scripts currently use include(...) for project files. If you prefer package usage, try replacing includes with using HaloArrays and ensure the module exports are sufficient.
- Some plotting code (using Plots) is included near the end of the scripts; comment it out in headless environments.
