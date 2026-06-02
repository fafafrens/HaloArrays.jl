using HaloArrays
using MPI
using DiffEqBase: ODEProblem
using OrdinaryDiffEq
using OhMyThreads: tforeach, tmapreduce
using Printf

if !MPI.Initialized()
    MPI.Init()
end

const _FVSerialHaloArray = Union{HaloArray,LocalHaloArray}

function _fv_zero_storage!(u::_FVSerialHaloArray)
    fill!(parent(u), zero(eltype(u)))
    return u
end

function _fv_zero_storage!(u::ThreadedHaloArray)
    tforeach(1:tile_count(u); scheduler=:static) do tile_id
        fill!(tile_parent(u, tile_id), zero(eltype(u)))
    end
    return u
end

function _fv_prepare_rhs!(du, u)
    start_halo_exchange!(u)
    _fv_zero_storage!(du)
    finish_halo_exchange!(u)
    boundary_condition!(u)
    return du
end

function _fv_accumulate_flux_1d!(du_data, u_data, ranges::FaceRanges, numerical_flux, dx, args...)
    invdx = inv(dx)
    offset = get_unit_vector(ranges, 1)

    @inbounds for IL in get_left_face(ranges, 1)
        IR = IL + offset
        du_data[IR] += numerical_flux(u_data[IL], u_data[IR], args...) * invdx
    end

    @inbounds for IL in get_internal_face(ranges)
        IR = IL + offset
        flux = numerical_flux(u_data[IL], u_data[IR], args...) * invdx
        du_data[IL] -= flux
        du_data[IR] += flux
    end

    @inbounds for IL in get_right_face(ranges, 1)
        IR = IL + offset
        du_data[IL] -= numerical_flux(u_data[IL], u_data[IR], args...) * invdx
    end

    return du_data
end

function _fv_rhs_1d!(du::_FVSerialHaloArray, u::_FVSerialHaloArray, numerical_flux, dx, args...)
    _fv_prepare_rhs!(du, u)
    _fv_accumulate_flux_1d!(parent(du), parent(u), FaceRanges(u), numerical_flux, dx, args...)
    return du
end

function _fv_rhs_1d!(du::ThreadedHaloArray, u::ThreadedHaloArray, numerical_flux, dx, args...)
    _fv_prepare_rhs!(du, u)
    ranges = FaceRanges(u)

    tforeach(1:tile_count(u); scheduler=:static) do tile_id
        _fv_accumulate_flux_1d!(
            tile_parent(du, tile_id),
            tile_parent(u, tile_id),
            ranges,
            numerical_flux,
            dx,
            args...,
        )
    end

    return du
end

function _fv_fill_profile_1d!(u::_FVSerialHaloArray, profile)
    nx = global_size(u)[1]

    fill_from_global_indices!(u) do I
        profile((I[1] - 0.5) / nx)
    end

    synchronize_halo!(u)
    return u
end

function _fv_fill_profile_1d!(u::ThreadedHaloArray, profile)
    nx = global_size(u)[1]

    for I in CartesianIndices(axes(u))
        u[Tuple(I)...] = profile((I[1] - 0.5) / nx)
    end

    synchronize_halo!(u)
    return u
end

function _fv_solve_diffeq_1d(rhs!, u0, p; dt, steps, info=(;))
    tspan = (0.0, steps * dt)
    prob = ODEProblem{true}(rhs!, u0, tspan, p)
    sol = solve(prob, Tsit5(); dt, adaptive=false, save_everystep=false)
    u = sol.u[end]
    synchronize_halo!(u)
    return u, sol, merge(info, (; dt, time=last(tspan)))
end

_fv_mass(u) = sum(u) / global_size(u)[1]
_fv_mass(u, dx) = dx * sum(u)

function _fv_run_local_1d(fill_initial!, solve_problem!; nx, halo=1, boundary_condition=:periodic, kwargs...)
    u0 = LocalHaloArray(Float64, (nx,), halo; boundary_condition)
    fill_initial!(u0)
    initial_mass = _fv_mass(u0)
    u, sol, info = solve_problem!(u0; kwargs...)
    return u, sol, info, initial_mass, _fv_mass(u, info.dx)
end

function _fv_run_threaded_1d(
        fill_initial!,
        solve_problem!;
        nx,
        tile_dims=(4,),
        halo=1,
        boundary_condition=:periodic,
        kwargs...,
)
    length(tile_dims) == 1 ||
        throw(ArgumentError("1D examples require one tile dimension, got $tile_dims"))
    nx % tile_dims[1] == 0 ||
        throw(ArgumentError("nx=$nx must be divisible by tile_dims[1]=$(tile_dims[1])"))

    u0 = ThreadedHaloArray(
        Float64,
        (nx ÷ tile_dims[1],),
        halo;
        dims=tile_dims,
        boundary_condition,
    )
    fill_initial!(u0)
    initial_mass = _fv_mass(u0)
    u, sol, info = solve_problem!(u0; kwargs...)
    return u, sol, info, initial_mass, _fv_mass(u, info.dx)
end

function _fv_run_mpi_1d(fill_initial!, solve_problem!; nx, halo=1, boundary_condition=:periodic, kwargs...)
    comm = MPI.COMM_WORLD
    nranks = MPI.Comm_size(comm)
    nx % nranks == 0 ||
        throw(ArgumentError("nx=$nx must be divisible by MPI ranks=$nranks"))

    topology = CartesianTopology(comm, (0,); periodic=(boundary_condition == :periodic,))
    u0 = HaloArray(Float64, (nx ÷ nranks,), halo, topology; boundary_condition)
    fill_initial!(u0)
    initial_mass = _fv_mass(u0)
    u, sol, info = solve_problem!(u0; kwargs...)
    return u, sol, info, initial_mass, _fv_mass(u, info.dx)
end

function _fv_max_exact_error_1d(u::_FVSerialHaloArray, exact_value, args...)
    h = halo_width(u)
    nx = global_size(u)[1]
    local_error = 0.0

    @inbounds for I in CartesianIndices(interior_range(u))
        storage_i = Tuple(I)
        owned_i = ntuple(d -> storage_i[d] - h, Val(ndims(u)))
        global_i = owned_to_global_index(u, owned_i)[1]
        local_error = max(local_error, abs(parent(u)[I] - exact_value(global_i, nx, args...)))
    end

    return u isa HaloArray ? MPI.Allreduce(local_error, max, get_comm(u)) : local_error
end

function _fv_max_exact_error_1d(u::ThreadedHaloArray, exact_value, args...)
    h = halo_width(u)
    nx = global_size(u)[1]
    owned_tile_size = tile_size(u)

    return tmapreduce(tile_id -> begin
        coord = tile_coordinates(u, tile_id)
        data = tile_parent(u, tile_id)
        tile_error = 0.0

        @inbounds for I in CartesianIndices(interior_range(u))
            storage_i = Tuple(I)
            owned_i = storage_i[1] - h
            global_i = (coord[1] - 1) * owned_tile_size[1] + owned_i
            tile_error = max(tile_error, abs(data[I] - exact_value(global_i, nx, args...)))
        end

        tile_error
    end, max, 1:tile_count(u); scheduler=:static)
end

function _fv_root_print_summary(label, u, sol, info, initial_mass, final_mass; exact_error=nothing)
    max_value = maximum(u)
    mass_error = final_mass - initial_mass

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        if isnothing(exact_error)
            @printf(
                "%-32s nx=%d dt=%.3e time=%.3f saved=%d max=%.6f mass_error=%.3e\n",
                label,
                global_size(u)[1],
                info.dt,
                info.time,
                length(sol.t) - 1,
                max_value,
                mass_error,
            )
        else
            @printf(
                "%-32s nx=%d a=%.2f dt=%.3e time=%.3f saved=%d max=%.6f mass_error=%.3e exact_error=%.3e\n",
                label,
                global_size(u)[1],
                info.velocity,
                info.dt,
                info.time,
                length(sol.t) - 1,
                max_value,
                mass_error,
                exact_error,
            )
        end
    end

    return nothing
end
