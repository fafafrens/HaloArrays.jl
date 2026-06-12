include("common.jl")

using OhMyThreads: @tasks

function make_threaded_halo(::Val{N}, interior_size, halo_width, tile_dims) where {N}
    tile_size = tile_size_from_owned_size(interior_size, tile_dims)
    return ThreadedHaloArray(Float64, tile_size, halo_width; dims=tile_dims, boundary_condition=:repeating)
end

function _threaded_stencil_tile!(dest_tile, src_tile, offsets, range, ::Val{N}) where {N}
    @inbounds for I in CartesianIndices(range)
        laplacian = zero(eltype(src_tile))
        for dim in 1:N
            offset = offsets[dim]
            laplacian += src_tile[I + offset] - 2 * src_tile[I] + src_tile[I - offset]
        end
        dest_tile[I] = src_tile[I] + 0.1 * laplacian
    end
    return dest_tile
end

function threaded_stencil_step!(dest::ThreadedHaloArray{T,N}, src::ThreadedHaloArray{T,N}) where {T,N}
    synchronize_halo!(src)
    offsets = CartesianIndex.(versors(Val(N)))
    range = interior_range(src)

    @tasks for tile_id in 1:tile_count(src)
        src_tile = tile_parent(src, tile_id)
        dest_tile = tile_parent(dest, tile_id)
        _threaded_stencil_tile!(dest_tile, src_tile, offsets, range, Val(N))
    end

    return dest
end

function benchmark_case!(rows, name, f, samples, warmups, metadata)
    sink = Ref{Any}()
    times = benchmark_times!(samples, warmups) do
        sink[] = f()
    end
    print_summary(name, times)
    push!(rows, benchmark_record("threaded", name, times; metadata=copy(metadata)))
    return sink[]
end

function main()
    options = parse_args()
    ndims = option_int(options, "ndims", 2)
    halo_width = option_int(options, "halo", 1)
    samples = option_int(options, "samples", 30)
    warmups = option_int(options, "warmups", 5)
    interior_size = option_owned_size(options, ndims, 64)
    tile_dims = option_tuple(options, "tile-dims", ndims, 2)

    halo = make_threaded_halo(Val(ndims), interior_size, halo_width, tile_dims)
    dest = similar(halo)
    fill_benchmark_data!(halo)
    fill!(dest, 0.0)

    println("ThreadedHaloArray benchmark")
    println("  ndims:       ", ndims)
    println("  owned size:  ", interior_size)
    println("  tile dims:   ", tile_dims)
    println("  tile size:   ", tile_size(halo))
    println("  halo width:  ", halo_width)
    println("  samples:     ", samples)
    println("  warmups:     ", warmups)
    println()

    metadata = Dict{String,Any}(
        "ndims" => ndims,
        "interior_size" => joined_tuple(interior_size),
        "tile_dims" => joined_tuple(tile_dims),
        "tile_size" => joined_tuple(tile_size(halo)),
        "halo_width" => halo_width,
    )
    rows = Dict{String,Any}[]

    benchmark_case!(rows, "synchronize_halo", () -> begin
        synchronize_halo!(halo)
    end, samples, warmups, metadata)

    benchmark_case!(rows, "boundary_condition", () -> begin
        boundary_condition!(halo)
    end, samples, warmups, metadata)

    benchmark_case!(rows, "mapreduce", () -> begin
        mapreduce(abs2, +, halo)
    end, samples, warmups, metadata)

    benchmark_case!(rows, "stencil_step", () -> begin
        threaded_stencil_step!(dest, halo)
    end, samples, warmups, metadata)

    maybe_write_csv(options, rows)
end

main()
