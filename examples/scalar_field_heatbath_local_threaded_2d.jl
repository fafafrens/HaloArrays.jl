include(joinpath(@__DIR__, "scalar_field_heatbath_common.jl"))

function main()
    n = (32, 32)
    sweeps = 50
    p = heatbath_parameters()

    local_phi = LocalHaloArray(Float64, n, 1; boundary_condition=:periodic)
    local_history = Float64[]
    run_heatbath!(local_phi, heatbath_rng(local_phi, 1234), p; sweeps, history=local_history)
    local_obs = observables(local_phi)
    print_observables("LocalHaloArray", local_phi, local_obs)
    print_free_scalar_check("LocalHaloArray", local_phi, local_obs, p)
    print_magnetization_trace("LocalHaloArray", local_history)

    tile_dims = (2, 2)
    tile_cells = ntuple(d -> n[d] ÷ tile_dims[d], Val(2))
    threaded_phi = ThreadedHaloArray(Float64, tile_cells, 1; dims=tile_dims, boundary_condition=:periodic)
    run_heatbath!(threaded_phi, heatbath_rng(threaded_phi, 1234), p; sweeps)
    threaded_obs = observables(threaded_phi)
    print_observables("ThreadedHaloArray", threaded_phi, threaded_obs)
    print_free_scalar_check("ThreadedHaloArray", threaded_phi, threaded_obs, p)

    local_fields = MultiHaloArray(LocalHaloArray, Float64, n, 1;
        boundary_conditions=(phi=:periodic, chi=:periodic))
    run_heatbath!(local_fields, heatbath_rng(local_fields, 4321), p; sweeps)
    print_observables("Local MultiHaloArray", local_fields, observables(local_fields))

    local_replicas = ArrayOfHaloArray(LocalHaloArray, Float64, n, 1;
        boundary_conditions=fill(:periodic, 2))
    run_heatbath!(local_replicas, heatbath_rng(local_replicas, 5678), p; sweeps)
    print_observables("Local ArrayOfHaloArray", local_replicas, observables(local_replicas))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
