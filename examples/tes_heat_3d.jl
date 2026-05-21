include("heat_diffusion_mpi_3d.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    run_mpi_heat_3d()
end
