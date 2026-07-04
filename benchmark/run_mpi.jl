# Launch exchange_mpi.jl under mpiexec:  julia --project=benchmark benchmark/run_mpi.jl [nranks]
using MPI

np     = isempty(ARGS) ? 4 : parse(Int, ARGS[1])
script = joinpath(@__DIR__, "exchange_mpi.jl")

MPI.mpiexec() do exe
    run(`$exe -n $np $(Base.julia_cmd()) --project=$(Base.active_project()) $script`)
end
