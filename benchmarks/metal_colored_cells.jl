include("common.jl")

using KernelAbstractions
using Metal

const KA = KernelAbstractions

@kernel function naive_colored_cell_kernel!(
        u,
        first_i::Int,
        first_j::Int,
        color::Int,
)
    ii, jj = @index(Global, NTuple)
    i = first_i + ii - 1
    j = first_j + jj - 1

    if mod(i + j, 2) == color
        u[i, j] = 0.25f0 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])
    end
end

@kernel function compressed_colored_cell_kernel!(
        u,
        region::CellCheckerboard{2},
)
    I = cell_index(region, @index(Global, NTuple))

    if is_cell_index_inbounds(region, I)
        i, j = I
        u[i, j] = 0.25f0 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])
    end
end

@kernel function manual_compressed_dim1_kernel!(
        u,
        first_i::Int,
        first_j::Int,
        full_i::Int,
        color::Int,
)
    ii, jj = @index(Global, NTuple)

    j = first_j + jj - 1
    base_i = first_i + 2 * (ii - 1)
    wanted_i_parity = mod(color - mod(j, 2), 2)
    i = base_i + mod(wanted_i_parity - mod(base_i, 2), 2)

    if i <= first_i + full_i - 1
        u[i, j] = 0.25f0 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])
    end
end

@kernel function manual_compressed_dim2_kernel!(
        u,
        first_i::Int,
        first_j::Int,
        full_j::Int,
        color::Int,
)
    i, jj = @index(Global, NTuple)

    i = first_i + i - 1
    base_j = first_j + 2 * (jj - 1)
    wanted_j_parity = mod(color - mod(i, 2), 2)
    j = base_j + mod(wanted_j_parity - mod(base_j, 2), 2)

    if j <= first_j + full_j - 1
        u[i, j] = 0.25f0 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])
    end
end

function option_int_list(options, name, default)
    raw = option_string(options, name, "")
    isempty(raw) && return collect(default)
    return parse.(Int, split(raw, ","))
end

function benchmark_gpu_sweeps!(kernel_calls, backend, steps, samples, warmups)
    for _ in 1:warmups
        for _ in 1:steps
            kernel_calls()
        end
    end
    KA.synchronize(backend)

    times = Vector{Float64}(undef, samples)
    for sample in 1:samples
        t0 = time_ns()
        for _ in 1:steps
            kernel_calls()
        end
        KA.synchronize(backend)
        times[sample] = (time_ns() - t0) / 1.0e9 / steps
    end

    return times
end

function time_naive!(kernel!, backend, u, ranges; steps, warmups, samples)
    region = get_interior_cell_window(ranges)
    first_i, first_j = Tuple(region.first)

    return benchmark_gpu_sweeps!(backend, steps, samples, warmups) do
        kernel!(u, first_i, first_j, 0; ndrange=region.size)
        kernel!(u, first_i, first_j, 1; ndrange=region.size)
    end
end

function time_compressed!(kernel!, backend, u, ranges, dim; steps, warmups, samples)
    region0 = get_interior_cell_window(ranges, 0, Dim(dim))
    region1 = get_interior_cell_window(ranges, 1, Dim(dim))

    return benchmark_gpu_sweeps!(backend, steps, samples, warmups) do
        kernel!(u, region0; ndrange=region0.size)
        kernel!(u, region1; ndrange=region1.size)
    end
end

function time_manual!(kernel!, backend, u, ranges, dim; steps, warmups, samples)
    region = get_interior_cell_window(ranges)
    first_i, first_j = Tuple(region.first)
    full_i, full_j = region.size
    launch_size = dim == 1 ? (cld(full_i, 2), full_j) : (full_i, cld(full_j, 2))
    full_compressed = dim == 1 ? full_i : full_j

    return benchmark_gpu_sweeps!(backend, steps, samples, warmups) do
        kernel!(u, first_i, first_j, full_compressed, 0; ndrange=launch_size)
        kernel!(u, first_i, first_j, full_compressed, 1; ndrange=launch_size)
    end
end

function run_case!(rows, nx, ny, steps, samples, warmups, include_manual)
    seed_data = rand(Float32, nx + 2, ny + 2)
    u_naive = Metal.MtlArray(seed_data)
    u_compressed_dim1 = Metal.MtlArray(seed_data)
    u_compressed_dim2 = Metal.MtlArray(seed_data)
    halo = LocalHaloArray(u_naive, 1, :repeating)
    ranges = CellRanges(halo)
    backend = KA.get_backend(u_naive)

    naive! = naive_colored_cell_kernel!(backend)
    compressed! = compressed_colored_cell_kernel!(backend)

    naive = time_naive!(naive!, backend, u_naive, ranges; steps, warmups, samples)
    compressed_dim1 = time_compressed!(compressed!, backend, u_compressed_dim1, ranges, 1; steps, warmups, samples)
    compressed_dim2 = time_compressed!(compressed!, backend, u_compressed_dim2, ranges, 2; steps, warmups, samples)

    metadata = Dict{String,Any}(
        "size" => "$(nx)x$(ny)",
        "steps_per_sample" => steps,
    )

    println("size=($nx, $ny), steps/sample=$steps, samples=$samples")
    print_summary("naive_full_launch", naive)
    print_summary("compressed_dim1", compressed_dim1)
    print_summary("compressed_dim2", compressed_dim2)
    println("  median speedup dim 1   ", round(median_value(naive) / median_value(compressed_dim1); digits=3), "x")
    println("  median speedup dim 2   ", round(median_value(naive) / median_value(compressed_dim2); digits=3), "x")

    push!(rows, benchmark_record("metal_colored_cells", "naive_full_launch", naive; metadata))
    push!(rows, benchmark_record("metal_colored_cells", "compressed_dim1", compressed_dim1; metadata))
    push!(rows, benchmark_record("metal_colored_cells", "compressed_dim2", compressed_dim2; metadata))

    if include_manual
        u_manual_dim1 = Metal.MtlArray(seed_data)
        u_manual_dim2 = Metal.MtlArray(seed_data)
        manual_dim1! = manual_compressed_dim1_kernel!(backend)
        manual_dim2! = manual_compressed_dim2_kernel!(backend)
        manual_dim1 = time_manual!(manual_dim1!, backend, u_manual_dim1, ranges, 1; steps, warmups, samples)
        manual_dim2 = time_manual!(manual_dim2!, backend, u_manual_dim2, ranges, 2; steps, warmups, samples)

        print_summary("manual_dim1", manual_dim1)
        print_summary("manual_dim2", manual_dim2)
        println("  median speedup manual1 ", round(median_value(naive) / median_value(manual_dim1); digits=3), "x")
        println("  median speedup manual2 ", round(median_value(naive) / median_value(manual_dim2); digits=3), "x")
        push!(rows, benchmark_record("metal_colored_cells", "manual_dim1", manual_dim1; metadata))
        push!(rows, benchmark_record("metal_colored_cells", "manual_dim2", manual_dim2; metadata))
    end

    println()
    return rows
end

function main()
    options = parse_args()
    sizes = option_int_list(options, "sizes", (128, 256, 512, 1024))
    steps = option_int(options, "steps", 50)
    samples = option_int(options, "samples", 10)
    warmups = option_int(options, "warmups", 3)
    include_manual = option_bool(options, "include-manual", true)

    println("Metal colored cell benchmark")
    println("  one timing unit: color 0 + color 1 red-black sweep")
    println("  naive:           full owned-cell launch with parity branch")
    println("  compressed:      half-size launch with final compressed-dim bound check")
    println("  include manual:  ", include_manual)
    println("  sizes:           ", join(sizes, ", "))
    println("  steps/sample:    ", steps)
    println("  samples:         ", samples)
    println("  warmups:         ", warmups)
    println()

    rows = Dict{String,Any}[]
    for n in sizes
        run_case!(rows, n, n, steps, samples, warmups, include_manual)
    end

    maybe_write_csv(options, rows)
end

main()
