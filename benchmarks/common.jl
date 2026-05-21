using MPI
using HaloArrays

function parse_args(args=ARGS)
    options = Dict{String,String}()
    for arg in args
        startswith(arg, "--") || continue
        key_value = split(arg[3:end], "="; limit=2)
        if length(key_value) == 1
            options[key_value[1]] = "true"
        else
            options[key_value[1]] = key_value[2]
        end
    end
    return options
end

option_string(options, name, default="") = get(options, name, default)
option_int(options, name, default) = parse(Int, get(options, name, string(default)))

function option_bool(options, name, default=false)
    raw = lowercase(get(options, name, string(default)))
    raw in ("1", "true", "yes", "on") && return true
    raw in ("0", "false", "no", "off") && return false
    error("--$name must be true or false")
end

function _default_tuple(default, ndims)
    default isa Tuple && return ntuple(d -> Int(default[d]), Val(ndims))
    return ntuple(_ -> Int(default), Val(ndims))
end

function option_tuple(options, name, ndims, default)
    raw = get(options, name, "")
    isempty(raw) && return _default_tuple(default, ndims)

    values = parse.(Int, split(raw, ","))
    length(values) == ndims || error("--$name must have $ndims comma-separated values")
    return Tuple(values)
end

function option_owned_size(options, ndims, default)
    if haskey(options, "owned-size")
        return option_tuple(options, "owned-size", ndims, default)
    elseif haskey(options, "local-size")
        @warn "--local-size is deprecated; use --owned-size instead" maxlog=1
        return option_tuple(options, "local-size", ndims, default)
    else
        return _default_tuple(default, ndims)
    end
end

function median_value(values)
    sorted = sort(values)
    n = length(sorted)
    mid = n ÷ 2
    return isodd(n) ? sorted[mid + 1] : (sorted[mid] + sorted[mid + 1]) / 2
end

function quantile_value(values, q)
    sorted = sort(values)
    index = clamp(ceil(Int, q * length(sorted)), 1, length(sorted))
    return sorted[index]
end

barrier(::Nothing) = nothing
barrier(comm) = MPI.Barrier(comm)

time_seconds(::Nothing) = time_ns() / 1.0e9
time_seconds(comm) = MPI.Wtime()

function max_elapsed(elapsed, ::Nothing)
    return elapsed
end

function max_elapsed(elapsed, comm)
    return MPI.Allreduce(elapsed, MPI.MAX, comm)
end

function benchmark_times!(f, samples, warmups; comm=nothing)
    for _ in 1:warmups
        barrier(comm)
        f()
    end

    times = Vector{Float64}(undef, samples)
    for sample in 1:samples
        barrier(comm)
        t0 = time_seconds(comm)
        f()
        barrier(comm)
        elapsed = time_seconds(comm) - t0
        times[sample] = max_elapsed(elapsed, comm)
    end
    return times
end

function print_summary(name, times)
    println(rpad(name, 26),
        " min=", round(1e6 * minimum(times), digits=2), " us",
        " median=", round(1e6 * median_value(times), digits=2), " us",
        " p90=", round(1e6 * quantile_value(times, 0.90), digits=2), " us",
        " max=", round(1e6 * maximum(times), digits=2), " us")
end

function benchmark_record(benchmark, name, times; metadata=Dict{String,Any}())
    row = Dict{String,Any}(
        "benchmark" => benchmark,
        "case" => name,
        "samples" => length(times),
        "min_us" => 1e6 * minimum(times),
        "median_us" => 1e6 * median_value(times),
        "p90_us" => 1e6 * quantile_value(times, 0.90),
        "max_us" => 1e6 * maximum(times),
    )
    merge!(row, metadata)
    return row
end

function _csv_escape(value)
    text = string(value)
    if occursin(",", text) || occursin("\"", text) || occursin("\n", text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function default_csv_columns(rows)
    base_columns = ["benchmark", "case", "samples", "min_us", "median_us", "p90_us", "max_us"]
    metadata_columns = String[]
    for row in rows
        for key in keys(row)
            key_string = string(key)
            key_string in base_columns && continue
            key_string in metadata_columns || push!(metadata_columns, key_string)
        end
    end
    sort!(metadata_columns)
    return vcat(base_columns, metadata_columns)
end

function write_csv(path, rows; columns=nothing)
    isempty(path) && return nothing
    isempty(rows) && return nothing

    cols = columns === nothing ? default_csv_columns(rows) : columns
    dir = dirname(path)
    if !isempty(dir) && dir != "."
        mkpath(dir)
    end

    open(path, "w") do io
        println(io, join(cols, ","))
        for row in rows
            println(io, join((_csv_escape(get(row, col, "")) for col in cols), ","))
        end
    end
    return nothing
end

function maybe_write_csv(options, rows; columns=nothing)
    path = option_string(options, "csv", "")
    return write_csv(path, rows; columns=columns)
end

function ensure_mpi()
    MPI.Initialized() || MPI.Init()
    return MPI.COMM_WORLD
end

function make_periodic_topology(comm, ndims)
    dims = MPI.Dims_create(MPI.Comm_size(comm), ntuple(_ -> 0, Val(ndims)))
    return CartesianTopology(comm, Tuple(Int.(dims)); periodic=ntuple(_ -> true, Val(ndims)))
end

function fill_benchmark_data!(halo::Union{HaloArray,LocalHaloArray})
    fill!(parent(halo), -1.0)
    fill_from_global_indices!(halo) do I
        return sum((d + 10) * I[d] for d in 1:length(I))
    end
    return halo
end

function fill_benchmark_data!(halo::ThreadedHaloArray)
    fill!(halo, -1.0)
    for I in CartesianIndices(axes(halo))
        idx = Tuple(I)
        halo[idx...] = sum((d + 10) * idx[d] for d in 1:length(idx))
    end
    return halo
end

function fill_benchmark_data!(halo::MultiHaloArray)
    for field in values(halo.arrays)
        fill_benchmark_data!(field)
    end
    return halo
end

function tile_size_from_owned_size(owned_size, tile_dims)
    all(d -> owned_size[d] % tile_dims[d] == 0, eachindex(owned_size)) ||
        error("--owned-size must be divisible by --tile-dims for ThreadedHaloArray benchmarks")
    return ntuple(d -> owned_size[d] ÷ tile_dims[d], length(owned_size))
end

function joined_tuple(values)
    return join(values, "x")
end
