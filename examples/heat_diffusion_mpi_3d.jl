include("heat_diffusion_mpi_common.jl")

function run_mpi_heat_3d(; owned_dims=(16, 16, 16), nt=20, kwargs...)
    return run_mpi_heat(Val(3); owned_dims, nt, kwargs...)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_mpi_heat_3d()
end
