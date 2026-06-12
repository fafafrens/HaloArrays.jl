# phi4_metal_2d.jl
#
# Simple 2D phi^4 Metropolis simulation using:
#   - HaloArrays.jl
#   - Metal.jl
#   - KernelAbstractions.jl
#   - checkerboard colored cell kernels
#
# The random numbers are generated into compact color-launch buffers,
# not into full halo-array-sized buffers.

using HaloArrays
using KernelAbstractions
using Metal
using Random
using Printf
using Statistics

const KA = KernelAbstractions

# ----------------------------
# Parameters
# ----------------------------

Base.@kwdef struct Phi4Params{T}
    m2::T       # bare mass squared
    λ::T        # quartic coupling
    κ::T        # nearest-neighbor gradient coupling
    ϵ::T        # proposal amplitude
end

# ----------------------------
# Initialization
# ----------------------------

function initialize_phi4_field(n::NTuple{2, Int}, halo::Int; amplitude = 0.1f0)
    # Create on CPU first because filling by ordinary Julia code is convenient.
    phi_cpu = LocalHaloArray(Float32, n, halo; boundary_condition = :periodic)

    interior = interior_view(phi_cpu)
    interior .= amplitude .* (2.0f0 .* rand(Float32, size(interior)) .- 1.0f0)

    synchronize_halo!(phi_cpu)

    # Move the full halo-padded storage to Metal.
    phi_gpu = LocalHaloArray(Metal.MtlArray(parent(phi_cpu)), halo, :periodic)
    synchronize_halo!(phi_gpu)

    return phi_gpu
end

# ----------------------------
# GPU kernel
# ----------------------------

@kernel function phi4_metropolis_color_kernel!(
        phi,
        rand_proposal,
        rand_accept,
        region::ColoredCellKernelRegion{2},
        params::Phi4Params{Float32},
    )
    # J is the compact launch index.
    # I is the physical storage index inside the halo-padded parent array.
    J = @index(Global, NTuple)
    I = cell_index(region, J)

    if is_cell_index_inbounds(region, I)
        i = I[1]
        j = I[2]

        @inbounds begin
            old = phi[i, j]

            # Random numbers are indexed by compact launch coordinates, not by storage index.
            r_prop = rand_proposal[J...]
            r_acc = rand_accept[J...]

            # Symmetric proposal:
            #     phi_new = phi_old + ϵ * uniform(-1,1)
            η = 2.0f0 * r_prop - 1.0f0
            new = old + params.ϵ * η

            # Periodic halos make these neighbor reads valid.
            xp = phi[i + 1, j]
            xm = phi[i - 1, j]
            yp = phi[i, j + 1]
            ym = phi[i, j - 1]

            old2 = old * old
            new2 = new * new

            # Local potential:
            #     V(phi) = 1/2 m² phi² + λ phi⁴
            ΔV =
                0.5f0 * params.m2 * (new2 - old2) +
                params.λ * (new2 * new2 - old2 * old2)

            # Bond contribution:
            #     κ/2 * sum_nn (phi_x - phi_nn)^2
            ΔG = 0.5f0 * params.κ * (
                (new - xp)^2 - (old - xp)^2 +
                    (new - xm)^2 - (old - xm)^2 +
                    (new - yp)^2 - (old - yp)^2 +
                    (new - ym)^2 - (old - ym)^2
            )

            ΔS = ΔV + ΔG

            if ΔS <= 0.0f0 || r_acc < exp(-ΔS)
                phi[i, j] = new
            end
        end
    end
end

# ----------------------------
# Random buffers
# ----------------------------

struct RandomBuffers{A}
    proposal::A
    accept::A
end

function make_random_buffers(phi)
    ranges = CellRanges(phi)
    region0 = get_colored_interior_cell_region(ranges, 0; compressed_dim = 2)

    proposal = Metal.zeros(Float32, region0.size...)
    accept = Metal.zeros(Float32, region0.size...)

    return RandomBuffers(proposal, accept)
end

function refill_random_buffers!(buffers::RandomBuffers)
    # Metal.rand! fills the existing MtlArray buffers on the GPU.
    # These buffers have the compact colored launch size, not the full halo-array size.
    Metal.rand!(buffers.proposal)
    Metal.rand!(buffers.accept)

    return buffers
end

# ----------------------------
# One checkerboard sweep
# ----------------------------

function phi4_sweep!(kernel!, backend, phi, params, buffers::RandomBuffers)
    ranges = CellRanges(phi)

    for color in 0:1
        # Before updating one color, halos must reflect the current opposite color.
        synchronize_halo!(phi)
        KA.synchronize(backend)

        region = get_colored_interior_cell_region(ranges, color; compressed_dim = 2)
        #any(==(0), region.size) && continue

        refill_random_buffers!(buffers)

        kernel!(
            parent(phi),
            buffers.proposal,
            buffers.accept,
            region,
            params;
            ndrange = region.size,
        )

        KA.synchronize(backend)
    end

    synchronize_halo!(phi)
    KA.synchronize(backend)

    return phi
end

# ----------------------------
# Measurements
# ----------------------------

function magnetization(phi)
    x = Array(interior_view(phi))
    return mean(Float64.(x))
end

function phi2_mean(phi)
    x = Array(interior_view(phi))
    return mean(abs2, Float64.(x))
end

# ----------------------------
# Main driver
# ----------------------------

function run_phi4_metal_haloarray_2d(;
        n = (128, 128),
        nsweeps = 3_000,
        measure_every = 100,
        m2 = -1.0f0,
        λ = 1.0f0,
        κ = 1.0f0,
        ϵ = 0.5f0,
        groupsize = (16, 16),
    )
    halo = 1

    phi = initialize_phi4_field(n, halo)

    backend = KA.get_backend(parent(phi))
    @show backend
    kernel! = phi4_metropolis_color_kernel!(backend, groupsize)

    params = Phi4Params(
        m2 = Float32(m2),
        λ = Float32(λ),
        κ = Float32(κ),
        ϵ = Float32(ϵ),
    )

    buffers = make_random_buffers(phi)

    @printf("2D phi^4 Metropolis with Metal + HaloArrays\n")
    @printf(" lattice: %d x %d\n", n[1], n[2])
    @printf(" groupsize: %s\n", string(groupsize))
    @printf(
        " m2 = %.4f, λ = %.4f, κ = %.4f, ϵ = %.4f\n",
        params.m2, params.λ, params.κ, params.ϵ
    )

    for sweep in 1:nsweeps
        phi4_sweep!(kernel!, backend, phi, params, buffers)

        if sweep % measure_every == 0
            m = magnetization(phi)
            p2 = phi2_mean(phi)
            @printf(
                " sweep %6d | <phi> = %+ .6e | <phi^2> = %.6e\n",
                sweep, m, p2
            )
        end
    end

    return phi
end

function main()
    run_phi4_metal_haloarray_2d()
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
