include(joinpath(@__DIR__, "ideal_hydro_common.jl"))

function mpi_ideal_hydro_state(nx, ny; halo=1, boundary_condition=:periodic)
    topology = CartesianTopology(MPI.COMM_WORLD, (0, 0);
        periodic=(boundary_condition == :periodic, boundary_condition == :periodic))

    nx % topology.dims[1] == 0 ||
        throw(ArgumentError("nx=$nx must be divisible by topology.dims[1]=$(topology.dims[1])"))
    ny % topology.dims[2] == 0 ||
        throw(ArgumentError("ny=$ny must be divisible by topology.dims[2]=$(topology.dims[2])"))

    owned_cells = (nx ÷ topology.dims[1], ny ÷ topology.dims[2])
    return MultiHaloArray(Float64, owned_cells, halo, topology;
        boundary_conditions=ideal_hydro_boundary_conditions(boundary_condition))
end

function run_mpi_ideal_hydro_2d(; nx=64, ny=64, halo=1, boundary_condition=:periodic, kwargs...)
    u = mpi_ideal_hydro_state(nx, ny; halo, boundary_condition)
    return run_ideal_hydro_2d!(u; kwargs...)
end

function main()
    u, info, initial, final = run_mpi_ideal_hydro_2d()
    print_hydro_summary("MPI MultiHaloArray", u, info, initial, final)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
