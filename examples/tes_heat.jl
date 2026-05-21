include("heat_diffusion_mpi_1d.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    run_mpi_heat_1d()
end
