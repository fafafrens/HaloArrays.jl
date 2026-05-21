include("heat_diffusion_mpi_2d.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    run_mpi_heat_2d()
end
