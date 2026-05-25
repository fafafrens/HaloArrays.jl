include(joinpath(@__DIR__, "finite_volume_diffeq_common.jl"))

initial_profile(x) = 1.0 + 0.25 * sin(2 * pi * x) + 0.1 * cos(4 * pi * x)
periodic_coordinate(x) = mod(x, 1.0)
upwind_flux(ul, ur, velocity) = velocity >= 0 ? velocity * ul : velocity * ur

advection_rhs!(du, u, p, t) = _fv_rhs_1d!(du, u, upwind_flux, p.dx, p.velocity)
fill_advection_initial_condition!(u) = _fv_fill_profile_1d!(u, initial_profile)

function solve_advection_diffeq(u0; velocity=1.0, steps=200, cfl=0.4)
    dx = 1 / global_size(u0)[1]
    dt = cfl * dx / abs(velocity)
    p = (; velocity, dx)
    return _fv_solve_diffeq_1d(advection_rhs!, u0, p; dt, steps, info=p)
end

function _exact_value(global_i, nx, velocity, time)
    x = (global_i - 0.5) / nx
    return initial_profile(periodic_coordinate(x - velocity * time))
end

max_exact_error(u, velocity, time) = _fv_max_exact_error_1d(u, _exact_value, velocity, time)

function _with_exact_error(result)
    u, sol, info, initial_mass, final_mass = result
    exact_error = max_exact_error(u, info.velocity, info.time)
    return u, sol, info, initial_mass, final_mass, exact_error
end

run_local_advection_diffeq(; nx=256, velocity=1.0, steps=200, cfl=0.4) =
    _with_exact_error(_fv_run_local_1d(
        fill_advection_initial_condition!,
        solve_advection_diffeq;
        nx,
        velocity,
        steps,
        cfl,
    ))

run_threaded_advection_diffeq(; nx=256, tile_dims=(4,), velocity=1.0, steps=200, cfl=0.4) =
    _with_exact_error(_fv_run_threaded_1d(
        fill_advection_initial_condition!,
        solve_advection_diffeq;
        nx,
        tile_dims,
        velocity,
        steps,
        cfl,
    ))

run_mpi_advection_diffeq(; nx=256, velocity=1.0, steps=200, cfl=0.4) =
    _with_exact_error(_fv_run_mpi_1d(
        fill_advection_initial_condition!,
        solve_advection_diffeq;
        nx,
        velocity,
        steps,
        cfl,
    ))

function main()
    nx = 256
    velocity = 1.0
    steps = 200
    cfl = 0.4

    local_u, local_sol, local_info, local_m0, local_m1, local_err =
        run_local_advection_diffeq(; nx, velocity, steps, cfl)
    _fv_root_print_summary(
        "OrdinaryDiffEq LocalHaloArray",
        local_u,
        local_sol,
        local_info,
        local_m0,
        local_m1;
        exact_error=local_err,
    )

    threaded_u, threaded_sol, threaded_info, threaded_m0, threaded_m1, threaded_err =
        run_threaded_advection_diffeq(; nx, tile_dims=(4,), velocity, steps, cfl)
    _fv_root_print_summary(
        "OrdinaryDiffEq ThreadedHaloArray",
        threaded_u,
        threaded_sol,
        threaded_info,
        threaded_m0,
        threaded_m1;
        exact_error=threaded_err,
    )

    mpi_u, mpi_sol, mpi_info, mpi_m0, mpi_m1, mpi_err =
        run_mpi_advection_diffeq(; nx, velocity, steps, cfl)
    _fv_root_print_summary(
        "OrdinaryDiffEq HaloArray MPI",
        mpi_u,
        mpi_sol,
        mpi_info,
        mpi_m0,
        mpi_m1;
        exact_error=mpi_err,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
