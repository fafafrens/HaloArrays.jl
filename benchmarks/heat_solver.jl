error("""
benchmarks/heat_solver.jl has been split into two explicit benchmarks:

  Local/threaded:
    julia --project=. benchmarks/heat_solver_local_threaded.jl --size=256,256 --tile-dims=2,1

  MPI comparison:
    mpiexec -n 4 julia --project=. benchmarks/heat_solver_mpi.jl --owned-size=64,64 --tile-dims=2,2
""")
