include("common.jl")

using Base.Threads: @threads, nthreads

_as_tuple(x::Number, n::Int) = ntuple(_ -> x, n)
_as_tuple(x, ::Int) = Tuple(x)

function stable_heat_dt(alpha, cfl, dx)
    dxs = dx isa Number ? (dx,) : Tuple(dx)
    return cfl / (alpha * sum(inv(abs2(d)) for d in dxs))
end

function problem_size_from_topology(owned_size, topology_dims)
    return ntuple(d -> owned_size[d] * topology_dims[d], length(owned_size))
end

function heat_initial_value(I, problem_size)
    center = ntuple(d -> (problem_size[d] + 1) / 2, length(problem_size))
    widths = ntuple(d -> problem_size[d] / 10, length(problem_size))
    exponent = sum(((I[d] - center[d]) / widths[d])^2 for d in eachindex(I))
    return exp(-exponent)
end

@inline function _heat_step_data!(dest_data, src_data, alpha, dt, dxs, offsets, range, ::Val{N}) where {N}
    @inbounds for I in CartesianIndices(range)
        laplacian = zero(eltype(src_data))
        for dim in 1:N
            offset = offsets[dim]
            laplacian += (src_data[I + offset] - 2 * src_data[I] + src_data[I - offset]) / dxs[dim]^2
        end
        dest_data[I] = src_data[I] + alpha * dt * laplacian
    end
    return dest_data
end

function fill_heat_initial!(halo::Union{HaloArray,LocalHaloArray}, problem_size)
    fill!(parent(halo), 0.0)
    fill_from_global_indices!(halo) do I
        heat_initial_value(I, problem_size)
    end
    synchronize_halo!(halo)
    return halo
end

function fill_heat_initial!(halo::ThreadedHaloArray{T,N}, problem_size) where {T,N}
    fill!(halo, 0.0)
    h = halo_width(halo)
    tile_dims = tile_size(halo)

    @threads :static for tile_id in 1:tile_count(halo)
        coords = tile_coordinates(halo, tile_id)
        tile_offset = ntuple(d -> (coords[d] - 1) * tile_dims[d], Val(N))
        data = tile_parent(halo, tile_id)

        @inbounds for I in CartesianIndices(interior_range(halo, tile_id))
            global_I = ntuple(d -> tile_offset[d] + I[d] - h, Val(N))
            data[I] = heat_initial_value(global_I, problem_size)
        end
    end

    synchronize_halo!(halo)
    return halo
end

function heat_step!(dest::Union{HaloArray{T,N},LocalHaloArray{T,N}},
        src::Union{HaloArray{S,N},LocalHaloArray{S,N}}, alpha, dt, dx) where {T,S,N}
    dxs = _as_tuple(dx, N)
    offsets = CartesianIndex.(versors(Val(N)))
    src_data = parent(src)
    dest_data = parent(dest)
    _heat_step_data!(dest_data, src_data, alpha, dt, dxs, offsets, interior_range(src), Val(N))

    return dest
end

function heat_step!(dest::ThreadedHaloArray{T,N}, src::ThreadedHaloArray{S,N}, alpha, dt, dx) where {T,S,N}
    dxs = _as_tuple(dx, N)
    offsets = CartesianIndex.(versors(Val(N)))
    range = interior_range(src)
    src_tiles = parent(src)
    dest_tiles = parent(dest)

    @threads :static for tile_id in eachindex(src_tiles)
        src_data = src_tiles[tile_id]
        dest_data = dest_tiles[tile_id]
        _heat_step_data!(dest_data, src_data, alpha, dt, dxs, offsets, range, Val(N))
    end

    return dest
end

function copy_halo_storage!(dest::Union{HaloArray,LocalHaloArray}, src::Union{HaloArray,LocalHaloArray})
    copyto!(parent(dest), parent(src))
    return dest
end

function copy_halo_storage!(dest::ThreadedHaloArray, src::ThreadedHaloArray)
    @threads :static for tile_id in 1:tile_count(dest)
        copyto!(tile_parent(dest, tile_id), tile_parent(src, tile_id))
    end
    return dest
end

function solve_heat_steps!(u, scratch; steps, alpha, dt, dx)
    current = u
    next = scratch

    for _ in 1:steps
        synchronize_halo!(current)
        heat_step!(next, current, alpha, dt, dx)
        current, next = next, current
    end

    synchronize_halo!(current)
    if current !== u
        copy_halo_storage!(u, current)
        synchronize_halo!(u)
    end
    return u
end

function heat_single_step!(scratch, u; alpha, dt, dx)
    synchronize_halo!(u)
    heat_step!(scratch, u, alpha, dt, dx)
    return scratch
end

function snapshot_interior(halo::LocalHaloArray)
    return Array(interior_view(halo))
end

function snapshot_interior(halo::ThreadedHaloArray{T,N}) where {T,N}
    data = Array{T}(undef, size(halo))
    tile_dims = tile_size(halo)

    for tile_id in 1:tile_count(halo)
        coords = tile_coordinates(halo, tile_id)
        inds = ntuple(Val(N)) do d
            first_owned = (coords[d] - 1) * tile_dims[d] + 1
            last_owned = coords[d] * tile_dims[d]
            first_owned:last_owned
        end
        data[inds...] .= interior_view(halo, tile_id)
    end

    return data
end

function benchmark_case!(rows, benchmark, name, f, samples, warmups, metadata; comm=nothing, rank=0)
    times = benchmark_times!(f, samples, warmups; comm=comm)
    if rank == 0
        print_summary(name, times)
        push!(rows, benchmark_record(benchmark, name, times; metadata=copy(metadata)))
    end
    return times
end

function compare_backends!(mpi_u, mpi_tmp, local_u, local_tmp, threaded_u, threaded_tmp,
        problem_size; steps, alpha, dt, dx, comm, rank)
    fill_heat_initial!(mpi_u, problem_size)
    solve_heat_steps!(mpi_u, mpi_tmp; steps=steps, alpha=alpha, dt=dt, dx=dx)
    mpi_solution = gather_haloarray(mpi_u; root=0)

    if rank == 0
        fill_heat_initial!(local_u, problem_size)
        fill_heat_initial!(threaded_u, problem_size)
        solve_heat_steps!(local_u, local_tmp; steps=steps, alpha=alpha, dt=dt, dx=dx)
        solve_heat_steps!(threaded_u, threaded_tmp; steps=steps, alpha=alpha, dt=dt, dx=dx)

        local_solution = snapshot_interior(local_u)
        threaded_solution = snapshot_interior(threaded_u)
        mpi_error = maximum(abs.(mpi_solution .- local_solution))
        threaded_error = maximum(abs.(threaded_solution .- local_solution))
        println("  MPI vs Local max error:      ", mpi_error)
        println("  Threaded vs Local max error: ", threaded_error)
        println()
    end

    MPI.Barrier(comm)
    return nothing
end

function main()
    comm = ensure_mpi()
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)

    options = parse_args()
    nd = option_int(options, "ndims", 2)
    halo_width_value = option_int(options, "halo", 1)
    samples = option_int(options, "samples", 20)
    warmups = option_int(options, "warmups", 3)
    steps = option_int(options, "steps", 10)
    alpha = parse(Float64, option_string(options, "alpha", "0.01"))
    cfl = parse(Float64, option_string(options, "cfl", "0.4"))
    owned_size_per_rank = option_owned_size(options, nd, 64)
    tile_dims = option_tuple(options, "tile-dims", nd, 2)

    topology = make_periodic_topology(comm, nd)
    problem_size = problem_size_from_topology(owned_size_per_rank, topology.dims)
    dx = ntuple(d -> 1.0 / problem_size[d], nd)
    dt = stable_heat_dt(alpha, cfl, dx)

    mpi_u = HaloArray(Float64, owned_size_per_rank, halo_width_value, topology; boundary_condition=:periodic)
    mpi_tmp = similar(mpi_u)

    local_u = local_tmp = threaded_u = threaded_tmp = nothing
    if rank == 0
        local_u = LocalHaloArray(Float64, problem_size, halo_width_value; boundary_condition=:periodic)
        local_tmp = similar(local_u)

        threaded_tile_size = tile_size_from_owned_size(problem_size, tile_dims)
        threaded_u = ThreadedHaloArray(Float64, threaded_tile_size, halo_width_value;
            dims=tile_dims, boundary_condition=:periodic)
        threaded_tmp = similar(threaded_u)
    end

    if rank == 0
        println("Heat solver backend benchmark")
        println("  ranks:                ", nproc)
        println("  topology:             ", topology.dims)
        println("  ndims:                ", nd)
        println("  MPI owned size/rank:  ", owned_size_per_rank)
        println("  global problem size:  ", problem_size)
        println("  threaded tile dims:   ", tile_dims)
        println("  Julia threads:        ", nthreads())
        println("  halo width:           ", halo_width_value)
        println("  steps per sample:     ", steps)
        println("  alpha:                ", alpha)
        println("  dt:                   ", dt)
        println("  samples:              ", samples)
        println("  warmups:              ", warmups)
        println()
    end

    compare_backends!(mpi_u, mpi_tmp, local_u, local_tmp, threaded_u, threaded_tmp, problem_size;
        steps=steps, alpha=alpha, dt=dt, dx=dx, comm=comm, rank=rank)

    fill_heat_initial!(mpi_u, problem_size)
    rank == 0 && fill_heat_initial!(local_u, problem_size)
    rank == 0 && fill_heat_initial!(threaded_u, problem_size)

    metadata = Dict{String,Any}(
        "ranks" => nproc,
        "topology" => joined_tuple(topology.dims),
        "ndims" => nd,
        "owned_size_per_rank" => joined_tuple(owned_size_per_rank),
        "global_size" => joined_tuple(problem_size),
        "tile_dims" => joined_tuple(tile_dims),
        "threads" => nthreads(),
        "halo_width" => halo_width_value,
        "steps" => steps,
        "alpha" => alpha,
        "dt" => dt,
    )
    rows = Dict{String,Any}[]

    benchmark_case!(rows, "heat_solver", "mpi_haloarray", () -> begin
        solve_heat_steps!(mpi_u, mpi_tmp; steps=steps, alpha=alpha, dt=dt, dx=dx)
    end, samples, warmups, metadata; comm=comm, rank=rank)

    benchmark_case!(rows, "heat_solver", "mpi_synchronize_halo", () -> begin
        synchronize_halo!(mpi_u)
    end, samples, warmups, metadata; comm=comm, rank=rank)

    benchmark_case!(rows, "heat_solver", "mpi_heat_step_only", () -> begin
        heat_step!(mpi_tmp, mpi_u, alpha, dt, dx)
    end, samples, warmups, metadata; comm=comm, rank=rank)

    benchmark_case!(rows, "heat_solver", "mpi_single_step", () -> begin
        heat_single_step!(mpi_tmp, mpi_u; alpha=alpha, dt=dt, dx=dx)
    end, samples, warmups, metadata; comm=comm, rank=rank)

    if rank == 0
        benchmark_case!(rows, "heat_solver", "local_haloarray", () -> begin
            solve_heat_steps!(local_u, local_tmp; steps=steps, alpha=alpha, dt=dt, dx=dx)
        end, samples, warmups, metadata)

        benchmark_case!(rows, "heat_solver", "local_synchronize_halo", () -> begin
            synchronize_halo!(local_u)
        end, samples, warmups, metadata)

        benchmark_case!(rows, "heat_solver", "local_heat_step_only", () -> begin
            heat_step!(local_tmp, local_u, alpha, dt, dx)
        end, samples, warmups, metadata)

        benchmark_case!(rows, "heat_solver", "local_single_step", () -> begin
            heat_single_step!(local_tmp, local_u; alpha=alpha, dt=dt, dx=dx)
        end, samples, warmups, metadata)

        benchmark_case!(rows, "heat_solver", "threaded_haloarray", () -> begin
            solve_heat_steps!(threaded_u, threaded_tmp; steps=steps, alpha=alpha, dt=dt, dx=dx)
        end, samples, warmups, metadata)

        benchmark_case!(rows, "heat_solver", "threaded_synchronize_halo", () -> begin
            synchronize_halo!(threaded_u)
        end, samples, warmups, metadata)

        benchmark_case!(rows, "heat_solver", "threaded_heat_step_only", () -> begin
            heat_step!(threaded_tmp, threaded_u, alpha, dt, dx)
        end, samples, warmups, metadata)

        benchmark_case!(rows, "heat_solver", "threaded_single_step", () -> begin
            heat_single_step!(threaded_tmp, threaded_u; alpha=alpha, dt=dt, dx=dx)
        end, samples, warmups, metadata)

        maybe_write_csv(options, rows)
    end

    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
