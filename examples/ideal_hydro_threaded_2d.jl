include(joinpath(@__DIR__, "ideal_hydro_common.jl"))

function default_tile_dims_2d(nx, ny, nthreads=Base.Threads.nthreads())
    nthreads > 0 || throw(ArgumentError("number of threads must be positive"))
    candidates = NTuple{2,Int}[]

    for d1 in 1:nthreads
        nthreads % d1 == 0 || continue
        d2 = nthreads ÷ d1

        if nx % d1 == 0 && ny % d2 == 0
            push!(candidates, (d1, d2))
        end
    end

    isempty(candidates) &&
        throw(ArgumentError("cannot split problem size ($nx, $ny) into $nthreads threaded tiles with integer tile sizes"))

    best = first(candidates)
    best_score = abs((nx ÷ best[1]) - (ny ÷ best[2]))

    for dims in Iterators.drop(candidates, 1)
        score = abs((nx ÷ dims[1]) - (ny ÷ dims[2]))
        if score < best_score
            best = dims
            best_score = score
        end
    end

    return best
end

function check_thread_tile_count(tile_dims)
    ntile = prod(tile_dims)
    nthread = Base.Threads.nthreads()
    ntile == nthread ||
        throw(ArgumentError("threaded hydro requires prod(tile_dims) == Threads.nthreads(); got prod(tile_dims)=$ntile and Threads.nthreads()=$nthread"))
    return nothing
end

function threaded_ideal_hydro_state(nx, ny; tile_dims=nothing, halo=1, boundary_condition=:periodic)
    tile_dims = isnothing(tile_dims) ? default_tile_dims_2d(nx, ny) : tile_dims
    check_thread_tile_count(tile_dims)

    nx % tile_dims[1] == 0 ||
        throw(ArgumentError("nx=$nx must be divisible by tile_dims[1]=$(tile_dims[1])"))
    ny % tile_dims[2] == 0 ||
        throw(ArgumentError("ny=$ny must be divisible by tile_dims[2]=$(tile_dims[2])"))

    tile_size = (nx ÷ tile_dims[1], ny ÷ tile_dims[2])
    return ThreadedMultiHaloArray(Float64, tile_size, halo;
        dims=tile_dims,
        boundary_conditions=ideal_hydro_boundary_conditions(boundary_condition))
end

function run_threaded_ideal_hydro_2d(;
        nx=64,
        ny=64,
        tile_dims=nothing,
        halo=1,
        boundary_condition=:periodic,
        kwargs...,
)
    u = threaded_ideal_hydro_state(nx, ny; tile_dims, halo, boundary_condition)
    return run_ideal_hydro_2d!(u; kwargs...)
end

function main()
    u, info, initial, final = run_threaded_ideal_hydro_2d()
    print_hydro_summary("ThreadedMultiHaloArray", u, info, initial, final)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
