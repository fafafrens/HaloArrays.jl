include(joinpath(@__DIR__, "finite_volume_diffeq_common.jl"))

burgers_flux(u) = 0.5 * u^2
burgers_initial_profile(x) = 0.5 + exp(-100 * (x - 0.35)^2)

function rusanov_flux(ul, ur)
    wavespeed = max(abs(ul), abs(ur))
    return 0.5 * (burgers_flux(ul) + burgers_flux(ur)) - 0.5 * wavespeed * (ur - ul)
end

burgers_rhs!(du, u, dx, t) = _fv_rhs_1d!(du, u, rusanov_flux, dx)
fill_burgers_initial_condition!(u) = _fv_fill_profile_1d!(u, burgers_initial_profile)

function solve_burgers_diffeq(u0; steps=300, cfl=0.4)
    dx = 1 / global_size(u0)[1]
    dt = cfl * dx / 1.5
    return _fv_solve_diffeq_1d(burgers_rhs!, u0, dx; dt, steps, info=(; dx))
end

run_local_burgers_diffeq(; nx=200, steps=300, cfl=0.4) =
    _fv_run_local_1d(fill_burgers_initial_condition!, solve_burgers_diffeq; nx, steps, cfl)

run_threaded_burgers_diffeq(; nx=200, tile_dims=(4,), steps=300, cfl=0.4) =
    _fv_run_threaded_1d(
        fill_burgers_initial_condition!,
        solve_burgers_diffeq;
        nx,
        tile_dims,
        steps,
        cfl,
    )

run_mpi_burgers_diffeq(; nx=200, steps=300, cfl=0.4) =
    _fv_run_mpi_1d(fill_burgers_initial_condition!, solve_burgers_diffeq; nx, steps, cfl)

function main()
    nx = 200
    steps = 300
    cfl = 0.4

    local_u, local_sol, local_info, local_m0, local_m1 =
        run_local_burgers_diffeq(; nx, steps, cfl)
    _fv_root_print_summary("OrdinaryDiffEq LocalHaloArray", local_u, local_sol, local_info, local_m0, local_m1)

    threaded_u, threaded_sol, threaded_info, threaded_m0, threaded_m1 =
        run_threaded_burgers_diffeq(; nx, tile_dims=(4,), steps, cfl)
    _fv_root_print_summary(
        "OrdinaryDiffEq ThreadedHaloArray",
        threaded_u,
        threaded_sol,
        threaded_info,
        threaded_m0,
        threaded_m1,
    )

    mpi_u, mpi_sol, mpi_info, mpi_m0, mpi_m1 =
        run_mpi_burgers_diffeq(; nx, steps, cfl)
    _fv_root_print_summary("OrdinaryDiffEq HaloArray MPI", mpi_u, mpi_sol, mpi_info, mpi_m0, mpi_m1)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
