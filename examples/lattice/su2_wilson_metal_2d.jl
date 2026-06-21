# su2_wilson_metal_2d.jl
#
# 2D pure SU(2) Wilson plaquette Metropolis simulation using:
#   - HaloArrays.jl
#   - Metal.jl
#   - KernelAbstractions.jl
#   - PhiloxRNG.jl
#   - checkerboard colored link updates
#
# Representation:
#   U = a0 I + i a1 σ1 + i a2 σ2 + i a3 σ3
#
# Gauge links are grouped in an ArrayOfHaloArray with shape (4, 2):
#   first field index:  quaternion component a = 1:4
#   second field index: lattice direction μ = 1:2
#
# The GPU kernels still receive the concrete parent MtlArrays.

using HaloArrays
using KernelAbstractions
using Metal
using PhiloxRNG
using Random
using Printf
using Statistics

const KA = KernelAbstractions

Base.@kwdef struct SU2WilsonParams{T}
    β::T
    ϵ::T
end

# Gauge links are stored as an ArrayOfHaloArray with field shape (4, 2):
#
#   U[a, μ]
#
# where
#
#   a = 1:4  -> quaternion component (a0, a1, a2, a3)
#   μ = 1:2  -> lattice direction (x, y)
#
# Thus:
#
#   U[1,1], U[2,1], U[3,1], U[4,1] are x0, x1, x2, x3
#   U[1,2], U[2,2], U[3,2], U[4,2] are y0, y1, y2, y3

@inline function su2_mul(
        a0::T, a1::T, a2::T, a3::T,
        b0::T, b1::T, b2::T, b3::T,
    ) where {T}
    return (
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + b0 * a1 - (a2 * b3 - a3 * b2),
        a0 * b2 + b0 * a2 - (a3 * b1 - a1 * b3),
        a0 * b3 + b0 * a3 - (a1 * b2 - a2 * b1),
    )
end

@inline function su2_retr_prod(
        a0::T, a1::T, a2::T, a3::T,
        b0::T, b1::T, b2::T, b3::T,
    ) where {T}
    return 2 * (a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3)
end

@inline function su2_normalize(a0::T, a1::T, a2::T, a3::T) where {T}
    nrm = sqrt(a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3)
    invn = inv(nrm)
    return (a0 * invn, a1 * invn, a2 * invn, a3 * invn)
end

@inline function su2_exp_algebra(α1::Float32, α2::Float32, α3::Float32)
    r = sqrt(α1 * α1 + α2 * α2 + α3 * α3)

    if r == 0.0f0
        return (1.0f0, 0.0f0, 0.0f0, 0.0f0)
    else
        half = 0.5f0 * r
        c = cos(half)
        s_over_r = sin(half) / r
        return (c, s_over_r * α1, s_over_r * α2, s_over_r * α3)
    end
end

@inline function local_site_id_2d(local_i::UInt32, local_j::UInt32, nx::UInt32)
    return UInt64(local_i - 0x00000001) +
        UInt64(nx) * UInt64(local_j - 0x00000001)
end

@inline function su2_counter_2d(
        region::CellCheckerboard{2},
        I::NTuple{2, <:Integer},
        sweep::UInt32,
        color::UInt32,
        dir::UInt32,
    )
    first_tuple = Tuple(region.first)

    local_i = UInt32(Int(I[1]) - first_tuple[1] + 1)
    local_j = UInt32(Int(I[2]) - first_tuple[2] + 1)
    nx = UInt32(region.full_size[1])

    site = local_site_id_2d(local_i, local_j, nx)

    ctr0 = site
    ctr1 = UInt64(sweep) |
        (UInt64(color & 0x0000ffff) << 32) |
        (UInt64(dir & 0x0000ffff) << 48)

    return ctr0, ctr1
end

function initialize_su2_links_2d(n::NTuple{2, Int}, halo::Int)
    # Cold start: every link is identity, U = (1,0,0,0).
    #
    # Field layout:
    #   U[a, μ]
    #   a = 1:4 quaternion component
    #   μ = 1:x, 2:y
    U = ArrayOfHaloArray(
        LocalHaloArray,
        Float32,
        (4, 2),
        n,
        halo;
        boundary_condition=:periodic,
        storage=Metal.zeros,
    )

    fill!(U, 0.0f0)
    interior_view(U[1, 1]) .= 1.0f0
    interior_view(U[1, 2]) .= 1.0f0
    synchronize_halo!(U)

    return U
end

@inline function staple_x_2d(
        x0, x1, x2, x3,
        y0, y1, y2, y3,
        i::Int, j::Int,
    )
    a = su2_mul(
        y0[i + 1, j], y1[i + 1, j], y2[i + 1, j], y3[i + 1, j],
        x0[i, j + 1], -x1[i, j + 1], -x2[i, j + 1], -x3[i, j + 1],
    )
    up = su2_mul(
        a[1], a[2], a[3], a[4],
        y0[i, j], -y1[i, j], -y2[i, j], -y3[i, j],
    )

    b = su2_mul(
        y0[i + 1, j - 1], -y1[i + 1, j - 1], -y2[i + 1, j - 1], -y3[i + 1, j - 1],
        x0[i, j - 1], -x1[i, j - 1], -x2[i, j - 1], -x3[i, j - 1],
    )
    down = su2_mul(
        b[1], b[2], b[3], b[4],
        y0[i, j - 1], y1[i, j - 1], y2[i, j - 1], y3[i, j - 1],
    )

    return (
        up[1] + down[1],
        up[2] + down[2],
        up[3] + down[3],
        up[4] + down[4],
    )
end

@inline function staple_y_2d(
        x0, x1, x2, x3,
        y0, y1, y2, y3,
        i::Int, j::Int,
    )
    a = su2_mul(
        x0[i, j + 1], x1[i, j + 1], x2[i, j + 1], x3[i, j + 1],
        y0[i + 1, j], -y1[i + 1, j], -y2[i + 1, j], -y3[i + 1, j],
    )
    right = su2_mul(
        a[1], a[2], a[3], a[4],
        x0[i, j], -x1[i, j], -x2[i, j], -x3[i, j],
    )

    b = su2_mul(
        x0[i - 1, j + 1], -x1[i - 1, j + 1], -x2[i - 1, j + 1], -x3[i - 1, j + 1],
        y0[i - 1, j], -y1[i - 1, j], -y2[i - 1, j], -y3[i - 1, j],
    )
    left = su2_mul(
        b[1], b[2], b[3], b[4],
        x0[i - 1, j], x1[i - 1, j], x2[i - 1, j], x3[i - 1, j],
    )

    return (
        right[1] + left[1],
        right[2] + left[2],
        right[3] + left[3],
        right[4] + left[4],
    )
end

@kernel function su2_metropolis_x_kernel!(
        x0, x1, x2, x3,
        y0, y1, y2, y3,
        region::CellCheckerboard{2},
        params::SU2WilsonParams{Float32},
        key::UInt64,
        sweep::UInt32,
        color::UInt32,
    )
    J = @index(Global, NTuple)
    I = cell_index(region, J)

    if is_cell_index_inbounds(region, I)
        i = I[1]
        j = I[2]

        @inbounds begin
            ctr0, ctr1 = su2_counter_2d(region, I, sweep, color, 0x00000001)
            r1, r2, r3, r4 = randu01_f32(ctr0, ctr1, key)

            α1 = params.ϵ * (2.0f0 * r1 - 1.0f0)
            α2 = params.ϵ * (2.0f0 * r2 - 1.0f0)
            α3 = params.ϵ * (2.0f0 * r3 - 1.0f0)

            R = su2_exp_algebra(α1, α2, α3)

            u0 = x0[i, j]
            u1 = x1[i, j]
            u2 = x2[i, j]
            u3 = x3[i, j]

            unew = su2_mul(R[1], R[2], R[3], R[4], u0, u1, u2, u3)
            unew = su2_normalize(unew[1], unew[2], unew[3], unew[4])

            staple = staple_x_2d(x0, x1, x2, x3, y0, y1, y2, y3, i, j)

            old_retr = su2_retr_prod(u0, u1, u2, u3, staple[1], staple[2], staple[3], staple[4])
            new_retr = su2_retr_prod(
                unew[1], unew[2], unew[3], unew[4],
                staple[1], staple[2], staple[3], staple[4]
            )

            ΔS = -0.5f0 * params.β * (new_retr - old_retr)

            if ΔS <= 0.0f0 || r4 < exp(-ΔS)
                x0[i, j] = unew[1]
                x1[i, j] = unew[2]
                x2[i, j] = unew[3]
                x3[i, j] = unew[4]
            end
        end
    end
end

@kernel function su2_metropolis_y_kernel!(
        x0, x1, x2, x3,
        y0, y1, y2, y3,
        region::CellCheckerboard{2},
        params::SU2WilsonParams{Float32},
        key::UInt64,
        sweep::UInt32,
        color::UInt32,
    )
    J = @index(Global, NTuple)
    I = cell_index(region, J)

    if is_cell_index_inbounds(region, I)
        i = I[1]
        j = I[2]

        @inbounds begin
            ctr0, ctr1 = su2_counter_2d(region, I, sweep, color, 0x00000002)
            r1, r2, r3, r4 = randu01_f32(ctr0, ctr1, key)

            α1 = params.ϵ * (2.0f0 * r1 - 1.0f0)
            α2 = params.ϵ * (2.0f0 * r2 - 1.0f0)
            α3 = params.ϵ * (2.0f0 * r3 - 1.0f0)

            R = su2_exp_algebra(α1, α2, α3)

            u0 = y0[i, j]
            u1 = y1[i, j]
            u2 = y2[i, j]
            u3 = y3[i, j]

            unew = su2_mul(R[1], R[2], R[3], R[4], u0, u1, u2, u3)
            unew = su2_normalize(unew[1], unew[2], unew[3], unew[4])

            staple = staple_y_2d(x0, x1, x2, x3, y0, y1, y2, y3, i, j)

            old_retr = su2_retr_prod(u0, u1, u2, u3, staple[1], staple[2], staple[3], staple[4])
            new_retr = su2_retr_prod(
                unew[1], unew[2], unew[3], unew[4],
                staple[1], staple[2], staple[3], staple[4]
            )

            ΔS = -0.5f0 * params.β * (new_retr - old_retr)

            if ΔS <= 0.0f0 || r4 < exp(-ΔS)
                y0[i, j] = unew[1]
                y1[i, j] = unew[2]
                y2[i, j] = unew[3]
                y3[i, j] = unew[4]
            end
        end
    end
end

function su2_wilson_sweep!(
        kx!, ky!, backend,
        U::ArrayOfHaloArray,
        params::SU2WilsonParams{Float32},
        key::UInt64,
        sweep::Integer,
    )
    ranges = CellRanges(U)

    for color in 0:1
        synchronize_halo!(U)
        KA.synchronize(backend)

        region = get_interior_cell_window(ranges, color; compressed_dim = 2)
        any(iszero, region.size) && continue

        kx!(
            parent(U[1, 1]), parent(U[2, 1]), parent(U[3, 1]), parent(U[4, 1]),
            parent(U[1, 2]), parent(U[2, 2]), parent(U[3, 2]), parent(U[4, 2]),
            region,
            params,
            key,
            UInt32(sweep),
            UInt32(color);
            ndrange = region.size,
        )

        KA.synchronize(backend)
    end

    for color in 0:1
        synchronize_halo!(U)
        KA.synchronize(backend)

        region = get_interior_cell_window(ranges, color; compressed_dim = 2)
        any(iszero, region.size) && continue

        ky!(
            parent(U[1, 1]), parent(U[2, 1]), parent(U[3, 1]), parent(U[4, 1]),
            parent(U[1, 2]), parent(U[2, 2]), parent(U[3, 2]), parent(U[4, 2]),
            region,
            params,
            key,
            UInt32(sweep),
            UInt32(color);
            ndrange = region.size,
        )

        KA.synchronize(backend)
    end

    synchronize_halo!(U)
    KA.synchronize(backend)

    return U
end

@inline cpu_adj(u) = (u[1], -u[2], -u[3], -u[4])

@inline function cpu_su2_mul(a, b)
    return su2_mul(a[1], a[2], a[3], a[4], b[1], b[2], b[3], b[4])
end

function average_plaquette(U::ArrayOfHaloArray)
    synchronize_halo!(U)

    x0 = Array(parent(U[1, 1])); x1 = Array(parent(U[2, 1])); x2 = Array(parent(U[3, 1])); x3 = Array(parent(U[4, 1]))
    y0 = Array(parent(U[1, 2])); y1 = Array(parent(U[2, 2])); y2 = Array(parent(U[3, 2])); y3 = Array(parent(U[4, 2]))

    nx = size(x0, 1) - 2
    ny = size(x0, 2) - 2

    s = 0.0
    count = 0

    @inbounds for j in 2:(ny + 1)
        for i in 2:(nx + 1)
            Ux = (x0[i, j], x1[i, j], x2[i, j], x3[i, j])
            Uy_xp = (y0[i + 1, j], y1[i + 1, j], y2[i + 1, j], y3[i + 1, j])
            Ux_yp_d = cpu_adj((x0[i, j + 1], x1[i, j + 1], x2[i, j + 1], x3[i, j + 1]))
            Uy_d = cpu_adj((y0[i, j], y1[i, j], y2[i, j], y3[i, j]))

            p = cpu_su2_mul(cpu_su2_mul(cpu_su2_mul(Ux, Uy_xp), Ux_yp_d), Uy_d)

            s += Float64(p[1]) # p[1] = (1/2) ReTr plaquette
            count += 1
        end
    end

    return s / count
end

function run_su2_wilson_metal_haloarray_2d(;
        n = (64, 64),
        nsweeps = 2_000,
        measure_every = 100,
        β = 2.0f0,
        ϵ = 0.35f0,
        groupsize = (16, 16),
        key = UInt64(0x123456789abcdef0),
    )
    halo = 1

    U = initialize_su2_links_2d(n, halo)

    backend = KA.get_backend(parent(U[1, 1]))
    @show backend

    kx! = su2_metropolis_x_kernel!(backend, groupsize)
    ky! = su2_metropolis_y_kernel!(backend, groupsize)

    params = SU2WilsonParams(β = Float32(β), ϵ = Float32(ϵ))

    @printf("2D pure SU(2) Wilson Metropolis with Metal + HaloArrays + PhiloxRNG\n")
    @printf(" lattice: %d x %d\n", n[1], n[2])
    @printf(" groupsize: %s\n", string(groupsize))
    @printf(" β = %.4f, ϵ = %.4f\n", params.β, params.ϵ)
    @printf(" Philox key: 0x%016x\n", key)

    for sweep in 1:nsweeps
        su2_wilson_sweep!(kx!, ky!, backend, U, params, key, sweep)

        if sweep % measure_every == 0
            plaq = average_plaquette(U)
            @printf(" sweep %6d | <P> = %.8f\n", sweep, plaq)
        end
    end

    return U
end

function main()
    run_su2_wilson_metal_haloarray_2d()
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
