include("heat_diffusion_mpi_common.jl")

function run_mpi_heat_1d(; owned_dims=(64,), nt=50, kwargs...)
    return run_mpi_heat(Val(1); owned_dims, nt, kwargs...)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_mpi_heat_1d()
end
