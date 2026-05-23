include("common.jl")

using Base.Threads: nthreads
using OhMyThreads: @tasks

function sync_variant_names(options)
    raw = option_string(options, "variants", "serial,tasks,threads")
    names = Symbol.(strip.(split(raw, ",")))
    allowed = (:serial, :tasks, :threads)
    unknown = setdiff(names, allowed)
    isempty(unknown) || error("Unknown variants: $(join(unknown, ", "))")
    return names
end

function sync_variant_functions()
    return Dict{Symbol,Function}(
        :serial => threaded_synchronize_halo_serial!,
        :tasks => threaded_synchronize_halo_tasks!,
        :threads => synchronize_halo_threads!,
    )
end

function make_threaded_sync_halo(::Val{N}, owned_size, halo_width, tile_dims, boundary_condition) where {N}
    tile_size = tile_size_from_owned_size(owned_size, tile_dims)
    halo = ThreadedHaloArray(Float64, tile_size, halo_width; dims=tile_dims, boundary_condition)
    fill_benchmark_data!(halo)
    return halo
end

function threaded_synchronize_halo_serial!(halo::ThreadedHaloArray)
    @inbounds for tile_id in eachindex(parent(halo))
        HaloArrays._threaded_synchronize_tile!(halo, tile_id)
    end
    return halo
end

function threaded_synchronize_halo_tasks!(halo::ThreadedHaloArray)
    @tasks for tile_id in eachindex(parent(halo))
        HaloArrays._threaded_synchronize_tile!(halo, tile_id)
    end
    return halo
end

function same_storage(a::ThreadedHaloArray, b::ThreadedHaloArray)
    return all(i -> parent(a)[i] == parent(b)[i], eachindex(parent(a)))
end

function check_variants(::Val{N}, owned_size, halo_width, tile_dims, boundary_condition, variants, functions) where {N}
    reference = make_threaded_sync_halo(Val(N), owned_size, halo_width, tile_dims, boundary_condition)
    threaded_synchronize_halo_serial!(reference)

    for variant in variants
        halo = make_threaded_sync_halo(Val(N), owned_size, halo_width, tile_dims, boundary_condition)
        functions[variant](halo)
        same_storage(halo, reference) ||
            error("threaded sync variant $(variant) produced a different halo state")
    end
    return nothing
end

function benchmark_variant!(rows, name, halo, f, samples, warmups, timer, metadata)
    run! = () -> f(halo)
    times = benchmark_times!(run!, samples, warmups, timer)
    allocations = allocation_bytes!(run!, warmups)
    print_summary(name, times; allocations)
    push!(rows, benchmark_record("threaded_sync_variants", name, times; metadata=copy(metadata), allocations))
    return nothing
end

function main()
    options = parse_args()
    ndims = option_int(options, "ndims", 2)
    halo_width = option_int(options, "halo", 1)
    samples = option_int(options, "samples", 30)
    warmups = option_int(options, "warmups", 5)
    owned_size = option_owned_size(options, ndims, 128)
    tile_dims = option_tuple(options, "tile-dims", ndims, 2)
    boundary_condition = Symbol(option_string(options, "boundary", "repeating"))
    variants = sync_variant_names(options)
    timer = Symbol(option_string(options, "timer", "manual"))
    functions = sync_variant_functions()

    check_variants(Val(ndims), owned_size, halo_width, tile_dims, boundary_condition, variants, functions)

    println("Threaded synchronization variant benchmark")
    println("  ndims:       ", ndims)
    println("  owned size:  ", owned_size)
    println("  tile dims:   ", tile_dims)
    println("  tile size:   ", tile_size_from_owned_size(owned_size, tile_dims))
    println("  Julia threads: ", nthreads())
    println("  halo width:  ", halo_width)
    println("  boundary:    ", boundary_condition)
    println("  variants:    ", join(variants, ", "))
    println("  samples:     ", samples)
    println("  warmups:     ", warmups)
    println("  timer:       ", timer)
    println()

    metadata = Dict{String,Any}(
        "ndims" => ndims,
        "owned_size" => joined_tuple(owned_size),
        "halo_width" => halo_width,
        "tile_dims" => joined_tuple(tile_dims),
        "tile_size" => joined_tuple(tile_size_from_owned_size(owned_size, tile_dims)),
        "boundary" => string(boundary_condition),
        "threads" => nthreads(),
    )
    rows = Dict{String,Any}[]

    for variant in variants
        halo = make_threaded_sync_halo(Val(ndims), owned_size, halo_width, tile_dims, boundary_condition)
        benchmark_variant!(rows, string(variant), halo, functions[variant], samples, warmups, timer, metadata)
    end

    maybe_write_csv(options, rows)
end

main()
