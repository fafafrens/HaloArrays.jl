# Examples

This folder contains example scripts that demonstrate using HaloArrays.jl to solve the heat equation in 1D, 2D, and 3D with periodic boundaries and parallel HDF5 I/O.

Local no-MPI examples
- The local examples use `LocalHaloArray`, the same interior/halo API, and boundary conditions without MPI:
  - 1D: julia --project=. examples/heat_diffusion_local_1d.jl
  - 2D: julia --project=. examples/heat_diffusion_local_2d.jl
  - 3D: julia --project=. examples/heat_diffusion_local_3d.jl
- The finite-difference update is shared in `examples/heat_diffusion_common.jl`, so the same stencil can be used by local arrays and MPI-backed `HaloArray`s.

Local and threaded array example
- `examples/local_and_threaded_halo_arrays.jl` runs the same 2D heat diffusion update with:
  - `LocalHaloArray`, a single local array with halo cells.
  - `ThreadedHaloArray`, a local tiled array with halo exchange between tiles.
- Run it with:
  - julia --project=. examples/local_and_threaded_halo_arrays.jl
- `examples/local_and_threaded_diffeq.jl` solves the same heat equation through `DifferentialEquations.jl` with the same two array types. The threaded RHS uses `OhMyThreads.@tasks` over tiles.
- Run it with:
  - julia --project=. examples/local_and_threaded_diffeq.jl

DiffEq example
- `examples/ode_diffeq.jl` solves `du/dt = -0.1u` with `OrdinaryDiffEq.Tsit5()` for `LocalHaloArray`, `ThreadedHaloArray`, and MPI-backed `HaloArray`.
- The solver packages are optional example dependencies, not package dependencies:
  - julia --project=/tmp/haloarrays-ode-example -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.add(["DiffEqBase", "OrdinaryDiffEq"])'
  - julia --project=/tmp/haloarrays-ode-example examples/ode_diffeq.jl
  - mpiexec -n 2 julia --project=/tmp/haloarrays-ode-example examples/ode_diffeq.jl

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
