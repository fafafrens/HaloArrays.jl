include("common.jl")

# ============================================================
# Checkerboard stencil: in-place vs out-of-place vs single-pass Jacobi.
#
# A red-black (checkerboard) sweep updates a cell as the average of its four
# neighbours in two passes by colour, so the in-place update is race-free
# (a colour only reads the opposite colour). That coloring is *only* needed to
# make the update in-place. This benchmark measures what it costs, three ways —
# one "unit" is one full update of every interior cell:
#
#   inplace  : 2 colour passes, read+write the SAME array (red-black Gauss-Seidel)
#   outplace : 2 colour passes, read u write un (both colours fill un = Jacobi)
#   jacobi   : 1 pass over the whole grid, read u write un (no colouring needed)
#
# The finding (see benchmark/README.md) is that the single-pass Jacobi is
# ~2-3x faster than the two-pass checkerboard on BOTH backends, for different
# reasons — on the GPU because it is one kernel launch instead of two (launch
# latency), on the CPU because a contiguous @simd pass replaces the checkerboard's
# stride-2 access (which defeats vectorisation). In-place vs out-of-place, at the
# same two-pass structure, is within noise: the coloring is the cost, not the
# aliasing. The trade is Jacobi's slower per-iteration convergence and 2x memory.
#
# CPU (plain loops) always runs. The Metal GPU variants run only if Metal loads
# — so `--project=examples` measures both backends, `--project=benchmark` the CPU.
#
#   julia --project=benchmark  -t 8 benchmark/checkerboard_inout.jl --sizes=256,512,1024,2048
#   julia --project=examples   -t 8 benchmark/checkerboard_inout.jl --sizes=256,512,1024,2048
# ============================================================

using Base.Threads: @threads, nthreads
using Statistics: median
using Printf: @printf

# ---- CPU: plain column-major loops (j outer / threaded, i inner) ----

function cpu_inplace!(u, lo, hi)
    for color in 0:1
        @threads :static for j in lo[2]:hi[2]
            istart = lo[1] + mod(color - mod(lo[1] + j, 2), 2)
            @inbounds for i in istart:2:hi[1]
                u[i, j] = 0.25f0 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])
            end
        end
    end
end

function cpu_outplace!(un, u, lo, hi)
    for color in 0:1
        @threads :static for j in lo[2]:hi[2]
            istart = lo[1] + mod(color - mod(lo[1] + j, 2), 2)
            @inbounds for i in istart:2:hi[1]
                un[i, j] = 0.25f0 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])
            end
        end
    end
end

function cpu_jacobi!(un, u, lo, hi)
    @threads :static for j in lo[2]:hi[2]
        @inbounds @simd for i in lo[1]:hi[1]
            un[i, j] = 0.25f0 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])
        end
    end
end

# ---- Metal: KernelAbstractions kernels, loaded only if Metal is present ----

const HAS_METAL = try
    @eval using KernelAbstractions
    @eval using Metal
    true
catch
    false
end

if HAS_METAL
    @eval begin
        const KA = KernelAbstractions

        @kernel function gpu_inplace!(u, region::CellCheckerboard{2})
            I = cell_index(region, @index(Global, NTuple))
            if is_cell_index_inbounds(region, I)
                i, j = I
                @inbounds u[i, j] = 0.25f0 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])
            end
        end

        @kernel function gpu_outplace!(un, u, region::CellCheckerboard{2})
            I = cell_index(region, @index(Global, NTuple))
            if is_cell_index_inbounds(region, I)
                i, j = I
                @inbounds un[i, j] = 0.25f0 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])
            end
        end

        @kernel function gpu_jacobi!(un, u, region::CellWindow{2})
            i, j = cell_index(region, @index(Global, NTuple))
            @inbounds un[i, j] = 0.25f0 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])
        end
    end
end

# ---- timing (median of `samples`, `steps` sweeps each) ----

function time_median(call!, sync!; steps, samples, warmups)
    for _ in 1:warmups, _ in 1:steps; call!(); end
    sync!()
    t = Vector{Float64}(undef, samples)
    for s in 1:samples
        t0 = time_ns()
        for _ in 1:steps; call!(); end
        sync!()
        t[s] = (time_ns() - t0) / 1e9 / steps
    end
    return median(t)
end

mcell(n, t) = n^2 / t / 1e6

function run_cpu(rows, n; steps, samples, warmups)
    u = rand(Float32, n + 2, n + 2); un = copy(u)
    lo = (2, 2); hi = (n + 1, n + 1)
    noop = () -> nothing
    ti = time_median(() -> cpu_inplace!(u, lo, hi), noop; steps, samples, warmups)
    to = time_median(() -> cpu_outplace!(un, u, lo, hi), noop; steps, samples, warmups)
    tj = time_median(() -> cpu_jacobi!(un, u, lo, hi), noop; steps, samples, warmups)
    @printf("CPU  %5d²  inplace %6.0f | outplace %6.0f (%.2fx) | jacobi-1pass %6.0f (%.2fx)  Mcell/s\n",
        n, mcell(n, ti), mcell(n, to), ti / to, mcell(n, tj), ti / tj)
    meta = Dict{String,Any}("size" => "$(n)x$(n)", "backend" => "cpu", "threads" => nthreads())
    push!(rows, benchmark_record("checkerboard_inout", "cpu_inplace", [ti]; metadata = meta))
    push!(rows, benchmark_record("checkerboard_inout", "cpu_outplace", [to]; metadata = meta))
    push!(rows, benchmark_record("checkerboard_inout", "cpu_jacobi", [tj]; metadata = meta))
end

function run_metal(rows, n; steps, samples, warmups)
    seed = rand(Float32, n + 2, n + 2)
    u = Metal.MtlArray(seed); un = Metal.MtlArray(copy(seed))
    halo = LocalHaloArray(u, 1, :repeating)
    ranges = CellRanges(halo)
    backend = KA.get_backend(u)
    sync! = () -> KA.synchronize(backend)

    cb0 = interior_cell_window(ranges, 0, Dim(2))
    cb1 = interior_cell_window(ranges, 1, Dim(2))
    full = interior_cell_window(ranges)
    ki = gpu_inplace!(backend); ko = gpu_outplace!(backend); kj = gpu_jacobi!(backend)

    ti = time_median(() -> (ki(u, cb0; ndrange = cb0.size); ki(u, cb1; ndrange = cb1.size)), sync!; steps, samples, warmups)
    to = time_median(() -> (ko(un, u, cb0; ndrange = cb0.size); ko(un, u, cb1; ndrange = cb1.size)), sync!; steps, samples, warmups)
    tj = time_median(() -> kj(un, u, full; ndrange = full.size), sync!; steps, samples, warmups)
    @printf("GPU  %5d²  inplace %6.0f | outplace %6.0f (%.2fx) | jacobi-1launch %6.0f (%.2fx)  Mcell/s\n",
        n, mcell(n, ti), mcell(n, to), ti / to, mcell(n, tj), ti / tj)
    meta = Dict{String,Any}("size" => "$(n)x$(n)", "backend" => "metal")
    push!(rows, benchmark_record("checkerboard_inout", "gpu_inplace", [ti]; metadata = meta))
    push!(rows, benchmark_record("checkerboard_inout", "gpu_outplace", [to]; metadata = meta))
    push!(rows, benchmark_record("checkerboard_inout", "gpu_jacobi", [tj]; metadata = meta))
end

function main()
    options = parse_args()
    raw = option_string(options, "sizes", "256,512,1024,2048")
    sizes = parse.(Int, split(raw, ","))
    steps = option_int(options, "steps", 50)
    samples = option_int(options, "samples", 15)
    warmups = option_int(options, "warmups", 5)

    println("Checkerboard in-place / out-of-place / single-pass Jacobi")
    println("  speedups are relative to in-place; CPU threads = ", nthreads(),
        HAS_METAL ? "; Metal GPU present" : "; Metal not loaded (CPU only)")
    println()

    rows = Dict{String,Any}[]
    for n in sizes
        run_cpu(rows, n; steps, samples, warmups)
        HAS_METAL && run_metal(rows, n; steps, samples, warmups)
    end
    maybe_write_csv(options, rows)
end

main()
