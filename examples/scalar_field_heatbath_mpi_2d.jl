include(joinpath(@__DIR__, "scalar_field_heatbath_common.jl"))

function main()
    owned_cells = (32, 32)
    sweeps = 50
    p = heatbath_parameters()
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    topology = CartesianTopology(MPI.COMM_WORLD, (0, 0); periodic=(true, true))

    mpi_phi = HaloArray(Float64, owned_cells, 1, topology; boundary_condition=:periodic)
    run_heatbath!(mpi_phi, heatbath_rng(mpi_phi, 1234), p; sweeps)
    mpi_obs = observables(mpi_phi)
    rank == 0 && print_observables("HaloArray MPI", mpi_phi, mpi_obs)

    mpi_fields = MultiHaloArray(Float64, owned_cells, 1, topology;
        boundary_conditions=(phi=:periodic, chi=:periodic))
    run_heatbath!(mpi_fields, heatbath_rng(mpi_fields, 4321), p; sweeps)
    mpi_fields_obs = observables(mpi_fields)
    rank == 0 && print_observables("MPI MultiHaloArray", mpi_fields, mpi_fields_obs)

    mpi_replicas = ArrayOfHaloArray(Float64, owned_cells, 1, topology;
        boundary_conditions=fill(:periodic, 2))
    run_heatbath!(mpi_replicas, heatbath_rng(mpi_replicas, 5678), p; sweeps)
    mpi_replicas_obs = observables(mpi_replicas)
    rank == 0 && print_observables("MPI ArrayOfHaloArray", mpi_replicas, mpi_replicas_obs)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
