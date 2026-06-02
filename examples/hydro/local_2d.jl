include(joinpath(@__DIR__, "common.jl"))

function ideal_hydro_state(nx, ny; halo=1, boundary_condition=:periodic)
    return LocalMultiHaloArray(Float64, (nx, ny), halo;
        boundary_conditions=ideal_hydro_boundary_conditions(boundary_condition))
end

function run_local_ideal_hydro_2d(; nx=64, ny=64, halo=1, boundary_condition=:periodic, kwargs...)
    u = ideal_hydro_state(nx, ny; halo, boundary_condition)
    return run_ideal_hydro_2d!(u; kwargs...)
end

function main()
    u, info, initial, final = run_local_ideal_hydro_2d()
    print_hydro_summary("LocalMultiHaloArray", u, info, initial, final)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
