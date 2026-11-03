# GPU stencil launches on a halo array, any dimension: Cartesian addressing
# (`cell_index`, neighbours at `I ± eμ`) vs linear-index addressing
# (`linear_cell_window`, neighbours at `i ± sμ`), plus the main-branch `mod(·, 2)`
# colored map against the current `& 1`. Every variant does the same arithmetic in
# the same order, so the results are checked to be bit-identical.
#
#   julia --project=examples benchmark/gpu_linear_cell_window.jl                  # Metal
#   julia --project=@haloarrays-cuda benchmark/gpu_linear_cell_window.jl --backend=cuda --eltypes=Float32,Float64
#
# Options: --backend=metal|cuda, --dims=2,3,4, --sizes=1024,128,48 (one per dim),
# --eltypes=Float32[,Float64], --steps=20, --samples=10, --warmups=3, --csv=path.
include("common.jl")

using KernelAbstractions
const KA = KernelAbstractions

const options = parse_args()
const BACKEND = lowercase(option_string(options, "backend", "metal"))
if BACKEND == "cuda"
    using CUDA
    device_array(x) = CuArray(x)
    device_name() = CUDA.name(CUDA.device())
elseif BACKEND == "metal"
    using Metal
    device_array(x) = MtlArray(x)
    device_name() = String(Metal.device().name)
else
    error("--backend must be metal or cuda, got $BACKEND")
end

# the colored map of the main branch (before the `& 1` change), for the comparison
@inline function cell_index_mod(region::CellCheckerboard{N,C}, J::NTuple{N,<:Integer}) where {N,C}
    first_tuple = Tuple(region.first)
    base = ntuple(d -> first_tuple[d] + (d == C ? 2 * (Int(J[d]) - 1) : Int(J[d]) - 1), Val(N))
    delta = mod(region.color - region.parity - sum(base), 2)
    return ntuple(d -> base[d] + (d == C ? delta : 0), Val(N))
end

# damped nearest-neighbour average; identical expression in every kernel
@inline function stencil(ϕ, ϕ₊, ϕ₋, ω)
    d = length(ϕ₊)
    return (one(ω) - ω) * ϕ + ω * (sum(ϕ₊) + sum(ϕ₋)) / (2d)
end

# ---- one checkerboard color, in place --------------------------------------

@kernel function colored_cartesian_mod!(u, region, e, ω)
    J = @index(Global, NTuple)
    Is = cell_index_mod(region, J)
    if is_cell_index_inbounds(region, Is)
        @inbounds begin
            I = CartesianIndex(Is)
            u[I] = stencil(u[I], map(eμ -> u[I + eμ], e), map(eμ -> u[I - eμ], e), ω)
        end
    end
end

@kernel function colored_cartesian!(u, region, e, ω)
    J = @index(Global, NTuple)
    Is = cell_index(region, J)
    if is_cell_index_inbounds(region, Is)
        @inbounds begin
            I = CartesianIndex(Is)
            u[I] = stencil(u[I], map(eμ -> u[I + eμ], e), map(eμ -> u[I - eμ], e), ω)
        end
    end
end

@kernel function colored_linear!(u, w, ω)
    J = @index(Global, NTuple)
    if is_cell_linear_index_inbounds(w, J)
        @inbounds begin
            i = cell_linear_index(w, J)
            s = cell_strides(w)
            u[i] = stencil(u[i], map(sμ -> u[i + sμ], s), map(sμ -> u[i - sμ], s), ω)
        end
    end
end

# ---- all interior cells, out of place --------------------------------------

@kernel function full_cartesian!(v, @Const(u), region, e, ω)
    J = @index(Global, NTuple)
    @inbounds begin
        I = CartesianIndex(cell_index(region, J))
        v[I] = stencil(u[I], map(eμ -> u[I + eμ], e), map(eμ -> u[I - eμ], e), ω)
    end
end

@kernel function full_linear!(v, @Const(u), w, ω)
    J = @index(Global, NTuple)
    @inbounds begin
        i = cell_linear_index(w, J)
        s = cell_strides(w)
        v[i] = stencil(u[i], map(sμ -> u[i + sμ], s), map(sμ -> u[i - sμ], s), ω)
    end
end

# ---- timing ----------------------------------------------------------------

function time_sweeps(f, backend; steps, samples, warmups)
    for _ in 1:warmups, _ in 1:steps
        f()
    end
    KA.synchronize(backend)
    times = zeros(samples)
    for k in 1:samples
        t0 = time_ns()
        for _ in 1:steps
            f()
        end
        KA.synchronize(backend)
        times[k] = (time_ns() - t0) / 1e9 / steps
    end
    return times
end

option_list(options, name, default) = (raw = option_string(options, name, ""); isempty(raw) ? default : split(raw, ","))

function run_case!(rows, ::Val{N}, L, ::Type{T}; steps, samples, warmups) where {N,T}
    ω = T(0.5)
    seed = T.(rand(ntuple(_ -> L + 2, N)...))
    make() = LocalHaloArray(device_array(copy(seed)), 1, :periodic)
    u_mod, u_cart, u_lin = make(), make(), make()
    backend = KA.get_backend(parent(u_mod))
    e = unit_vector(u_mod)
    ranges = CellRanges(u_mod)
    regions = (interior_cell_window(ranges, 0), interior_cell_window(ranges, 1))
    windows = (linear_cell_window(u_lin, 0), linear_cell_window(u_lin, 1))
    k_mod!, k_cart!, k_lin! = colored_cartesian_mod!(backend), colored_cartesian!(backend), colored_linear!(backend)

    # one timing unit: color 0 + color 1, each after its halo refresh
    t_mod = time_sweeps(backend; steps, samples, warmups) do
        for c in 1:2
            synchronize_halo!(u_mod); k_mod!(parent(u_mod), regions[c], e, ω; ndrange=regions[c].size)
        end
    end
    t_cart = time_sweeps(backend; steps, samples, warmups) do
        for c in 1:2
            synchronize_halo!(u_cart); k_cart!(parent(u_cart), regions[c], e, ω; ndrange=regions[c].size)
        end
    end
    t_lin = time_sweeps(backend; steps, samples, warmups) do
        for c in 1:2
            synchronize_halo!(u_lin); k_lin!(parent(u_lin), windows[c], ω; ndrange=windows[c].size)
        end
    end
    same_colored = Array(parent(u_mod)) == Array(parent(u_cart)) == Array(parent(u_lin))

    # out of place: the same source field, one destination per variant
    u = make(); synchronize_halo!(u)
    v_cart, v_lin = make(), make()
    region = interior_cell_window(ranges)
    window = linear_cell_window(u)
    f_cart!, f_lin! = full_cartesian!(backend), full_linear!(backend)
    t_fcart = time_sweeps(backend; steps, samples, warmups) do
        f_cart!(parent(v_cart), parent(u), region, e, ω; ndrange=region.size)
    end
    t_flin = time_sweeps(backend; steps, samples, warmups) do
        f_lin!(parent(v_lin), parent(u), window, ω; ndrange=window.size)
    end
    same_full = Array(interior_view(v_cart)) == Array(interior_view(v_lin))

    sites = L^N
    per_site(times) = round(1e9 * median_value(times) / sites, digits=3)
    println("d=$N  L=$L  $T  ($(sites) sites)")
    print_summary("colored cartesian mod", t_mod);   println("   ", per_site(t_mod), " ns/site")
    print_summary("colored cartesian &1", t_cart);   println("   ", per_site(t_cart), " ns/site")
    print_summary("colored linear", t_lin);          println("   ", per_site(t_lin), " ns/site")
    print_summary("full cartesian", t_fcart);        println("   ", per_site(t_fcart), " ns/site")
    print_summary("full linear", t_flin);            println("   ", per_site(t_flin), " ns/site")
    println("  colored: &1 vs mod ", round(median_value(t_mod) / median_value(t_cart), digits=3),
            "x,  linear vs mod ", round(median_value(t_mod) / median_value(t_lin), digits=3),
            "x   |   full: linear vs cartesian ", round(median_value(t_fcart) / median_value(t_flin), digits=3), "x")
    println("  bit-identical results: colored ", same_colored, ", full ", same_full)
    println()

    metadata = Dict{String,Any}("backend" => BACKEND, "device" => device_name(), "dims" => N,
                                "size" => L, "eltype" => string(T), "steps_per_sample" => steps)
    for (name, times) in (("colored_cartesian_mod", t_mod), ("colored_cartesian", t_cart), ("colored_linear", t_lin),
                          ("full_cartesian", t_fcart), ("full_linear", t_flin))
        push!(rows, benchmark_record("gpu_linear_cell_window", name, times; metadata))
    end
    return rows
end

function main()
    dims = parse.(Int, option_list(options, "dims", ["2", "3", "4"]))
    sizes = parse.(Int, option_list(options, "sizes", ["1024", "128", "48"]))
    length(sizes) == length(dims) || error("--sizes needs one size per entry of --dims")
    eltypes = map(s -> getfield(Base, Symbol(s)), option_list(options, "eltypes", ["Float32"]))
    steps = option_int(options, "steps", 20)
    samples = option_int(options, "samples", 10)
    warmups = option_int(options, "warmups", 3)

    println("GPU linear cell window benchmark")
    println("  backend:         ", BACKEND, " (", device_name(), ")")
    println("  one timing unit: colored = color 0 + color 1 sweep (halo refresh included); full = one out-of-place pass")
    println("  dims/sizes:      ", join(("$(d)D $(L)" for (d, L) in zip(dims, sizes)), ", "))
    println("  eltypes:         ", join(eltypes, ", "))
    println("  steps/sample:    ", steps, "   samples: ", samples, "   warmups: ", warmups)
    println()

    rows = Dict{String,Any}[]
    for (d, L) in zip(dims, sizes), T in eltypes
        run_case!(rows, Val(d), L, T; steps, samples, warmups)
    end
    maybe_write_csv(options, rows)
end

main()
