include("common.jl")

using Polyester              # enables PolyesterBackend (HaloArraysPolyesterExt)
using Base.Threads: nthreads

# Compare the ThreadBackend implementations on the operations that route through
# the trait (tile_foreach / tile_mapreduce dispatched on thread_backend(u)):
# synchronize, boundary condition, fill!, mapreduce, and broadcast. Run with
# several Julia threads to see the difference, e.g.
#   julia --project=. -t 4 benchmarks/thread_backends.jl --tile-dims=4,1

const _BACKENDS = Dict{Symbol,Any}(
    :ohmythreads => OhMyThreadsBackend(),
    :serial      => SerialBackend(),
    :polyester   => PolyesterBackend(),
)

function backend_names(options)
    raw = option_string(options, "backends", "ohmythreads,serial,polyester")
    names = Symbol.(strip.(split(raw, ",")))
    unknown = setdiff(names, collect(keys(_BACKENDS)))
    isempty(unknown) ||
        error("Unknown backends: $(join(unknown, ", ")). Choose from $(join(sort(collect(keys(_BACKENDS))), ", ")).")
    return names
end

function make_halo(interior_size, halo_width, tile_dims, backend)
    tile_size = tile_size_from_owned_size(interior_size, tile_dims)
    halo = ThreadedHaloArray(Float64, tile_size, halo_width;
        dims=tile_dims, boundary_condition=:repeating, thread_backend=backend)
    fill_benchmark_data!(halo)
    return halo
end

# Each case is a trait-routed operation whose dispatch depends on the backend.
function backend_cases(halo, dest)
    return (
        ("synchronize", () -> synchronize_halo_threads!(halo)),
        ("boundary",    () -> boundary_condition_threads!(halo)),
        ("fill",        () -> fill!(halo, 1.0)),
        ("mapreduce",   () -> mapreduce(abs2, +, halo)),
        ("broadcast",   () -> (dest .= halo .* 2)),
    )
end

function main()
    options = parse_args()
    ndims = option_int(options, "ndims", 2)
    halo_width = option_int(options, "halo", 1)
    samples = option_int(options, "samples", 30)
    warmups = option_int(options, "warmups", 5)
    interior_size = option_owned_size(options, ndims, 128)
    tile_dims = option_tuple(options, "tile-dims", ndims, 2)
    timer = Symbol(option_string(options, "timer", "manual"))
    backends = backend_names(options)

    println("ThreadBackend comparison benchmark")
    println("  ndims:         ", ndims)
    println("  owned size:    ", interior_size)
    println("  tile dims:     ", tile_dims)
    println("  tile size:     ", tile_size_from_owned_size(interior_size, tile_dims))
    println("  halo width:    ", halo_width)
    println("  Julia threads: ", nthreads())
    println("  backends:      ", join(backends, ", "))
    println("  samples:       ", samples)
    println("  warmups:       ", warmups)
    println("  timer:         ", timer)
    nthreads() == 1 &&
        @warn "Running with a single Julia thread; start julia with `-t N` to compare backends meaningfully."
    println()

    rows = Dict{String,Any}[]
    for name in backends
        halo = make_halo(interior_size, halo_width, tile_dims, _BACKENDS[name])
        dest = similar(halo)
        fill!(dest, 0.0)

        metadata = Dict{String,Any}(
            "backend" => string(name),
            "ndims" => ndims,
            "interior_size" => joined_tuple(interior_size),
            "tile_dims" => joined_tuple(tile_dims),
            "tile_size" => joined_tuple(tile_size_from_owned_size(interior_size, tile_dims)),
            "halo_width" => halo_width,
            "threads" => nthreads(),
        )

        println("[", name, "]")
        for (case, f) in backend_cases(halo, dest)
            times = benchmark_times!(f, samples, warmups, timer)
            allocations = allocation_bytes!(f, warmups)
            label = string(case, "/", name)
            print_summary(label, times; allocations)
            push!(rows, benchmark_record("thread_backends", label, times; metadata=copy(metadata), allocations))
        end
        println()
    end

    maybe_write_csv(options, rows)
end

main()
