include("mpi_common.jl")

# MPI heat diffusion in 1-D, 2-D, and 3-D.  The solver and halo exchange live in
# mpi_common.jl (which includes common.jl); run_mpi_heat(Val(N)) builds the
# Cartesian topology for the requested dimension.  Pass save_hdf5=true to write
# a gathered snapshot, e.g. run_mpi_heat_2d(save_hdf5=true).
run_mpi_heat_1d(; owned_dims=(64,),        nt=50, kwargs...) = run_mpi_heat(Val(1); owned_dims, nt, kwargs...)
run_mpi_heat_2d(; owned_dims=(32, 32),     nt=50, kwargs...) = run_mpi_heat(Val(2); owned_dims, nt, kwargs...)
run_mpi_heat_3d(; owned_dims=(16, 16, 16), nt=20, kwargs...) = run_mpi_heat(Val(3); owned_dims, nt, kwargs...)

if abspath(PROGRAM_FILE) == @__FILE__
    run_mpi_heat_1d()
    run_mpi_heat_2d()
    run_mpi_heat_3d()
end
