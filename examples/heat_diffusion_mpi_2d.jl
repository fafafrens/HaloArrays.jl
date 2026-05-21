include("heat_diffusion_mpi_common.jl")

function run_mpi_heat_2d(; owned_dims=(32, 32), nt=50, kwargs...)
    return run_mpi_heat(Val(2); owned_dims, nt, kwargs...)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_mpi_heat_2d()
end
