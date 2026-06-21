using HaloArrays
using OhMyThreads: tforeach
using Printf
using Random

Base.@kwdef struct SU2WilsonCPUParams{T}
    beta::T
    proposal_eps::T
end

@inline function su2_mul(a0::T, a1::T, a2::T, a3::T,
        b0::T, b1::T, b2::T, b3::T) where {T}
    return (
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + b0 * a1 - (a2 * b3 - a3 * b2),
        a0 * b2 + b0 * a2 - (a3 * b1 - a1 * b3),
        a0 * b3 + b0 * a3 - (a1 * b2 - a2 * b1),
    )
end

@inline function su2_retr_prod(a0::T, a1::T, a2::T, a3::T,
        b0::T, b1::T, b2::T, b3::T) where {T}
    return 2 * (a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3)
end

@inline function su2_normalize(a0::T, a1::T, a2::T, a3::T) where {T}
    invn = inv(sqrt(a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3))
    return (a0 * invn, a1 * invn, a2 * invn, a3 * invn)
end

@inline function su2_exp_algebra(alpha1::Float32, alpha2::Float32, alpha3::Float32)
    r = sqrt(alpha1 * alpha1 + alpha2 * alpha2 + alpha3 * alpha3)

    if r == 0.0f0
        return (1.0f0, 0.0f0, 0.0f0, 0.0f0)
    else
        half = 0.5f0 * r
        c = cos(half)
        s_over_r = sin(half) / r
        return (c, s_over_r * alpha1, s_over_r * alpha2, s_over_r * alpha3)
    end
end

function initialize_su2_links_2d(::Type{LocalHaloArray}, n::NTuple{2,Int}, halo::Int)
    U = ArrayOfHaloArray(LocalHaloArray, Float32, (4, 2), n, halo;
        boundary_condition=:periodic)
    set_cold_su2_links!(U)
    return U
end

function initialize_su2_links_2d(::Type{ThreadedHaloArray}, n::NTuple{2,Int}, halo::Int;
        tile_dims=(Base.Threads.nthreads(), 1))
    prod(tile_dims) == Base.Threads.nthreads() ||
        throw(ArgumentError("threaded example expects prod(tile_dims) == Threads.nthreads()"))
    all(d -> n[d] % tile_dims[d] == 0, 1:2) ||
        throw(ArgumentError("problem size $n is not divisible by tile_dims $tile_dims"))
    tile_size = (n[1] ÷ tile_dims[1], n[2] ÷ tile_dims[2])
    U = ArrayOfHaloArray(ThreadedHaloArray, Float32, (4, 2), tile_size, halo;
        dims=tile_dims, boundary_condition=:periodic)
    set_cold_su2_links!(U)
    return U
end

function set_cold_su2_links!(U::ArrayOfHaloArray)
    fill!(U, 0.0f0)
    set_field_interior!(U[1, 1], 1.0f0)
    set_field_interior!(U[1, 2], 1.0f0)
    synchronize_halo!(U)
    return U
end

function set_field_interior!(u::LocalHaloArray, value)
    interior_view(u) .= value
    return u
end

function set_field_interior!(u::ThreadedHaloArray, value)
    for tile_id in 1:tile_count(u)
        interior_view(u, tile_id) .= value
    end
    return u
end

link_arrays(U::ArrayOfHaloArray) = (
    parent(U[1, 1]), parent(U[2, 1]), parent(U[3, 1]), parent(U[4, 1]),
    parent(U[1, 2]), parent(U[2, 2]), parent(U[3, 2]), parent(U[4, 2]),
)

link_arrays(U::ArrayOfHaloArray, tile_id::Integer) = (
    tile_parent(U[1, 1], tile_id), tile_parent(U[2, 1], tile_id),
    tile_parent(U[3, 1], tile_id), tile_parent(U[4, 1], tile_id),
    tile_parent(U[1, 2], tile_id), tile_parent(U[2, 2], tile_id),
    tile_parent(U[3, 2], tile_id), tile_parent(U[4, 2], tile_id),
)

@inline function staple_x_2d(arrays, i::Int, j::Int)
    x0, x1, x2, x3, y0, y1, y2, y3 = arrays
    a = su2_mul(
        y0[i + 1, j], y1[i + 1, j], y2[i + 1, j], y3[i + 1, j],
        x0[i, j + 1], -x1[i, j + 1], -x2[i, j + 1], -x3[i, j + 1],
    )
    up = su2_mul(a[1], a[2], a[3], a[4],
        y0[i, j], -y1[i, j], -y2[i, j], -y3[i, j])

    b = su2_mul(
        y0[i + 1, j - 1], -y1[i + 1, j - 1], -y2[i + 1, j - 1], -y3[i + 1, j - 1],
        x0[i, j - 1], -x1[i, j - 1], -x2[i, j - 1], -x3[i, j - 1],
    )
    down = su2_mul(b[1], b[2], b[3], b[4],
        y0[i, j - 1], y1[i, j - 1], y2[i, j - 1], y3[i, j - 1])

    return (up[1] + down[1], up[2] + down[2], up[3] + down[3], up[4] + down[4])
end

@inline function staple_y_2d(arrays, i::Int, j::Int)
    x0, x1, x2, x3, y0, y1, y2, y3 = arrays
    a = su2_mul(
        x0[i, j + 1], x1[i, j + 1], x2[i, j + 1], x3[i, j + 1],
        y0[i + 1, j], -y1[i + 1, j], -y2[i + 1, j], -y3[i + 1, j],
    )
    right = su2_mul(a[1], a[2], a[3], a[4],
        x0[i, j], -x1[i, j], -x2[i, j], -x3[i, j])

    b = su2_mul(
        x0[i - 1, j + 1], -x1[i - 1, j + 1], -x2[i - 1, j + 1], -x3[i - 1, j + 1],
        y0[i - 1, j], -y1[i - 1, j], -y2[i - 1, j], -y3[i - 1, j],
    )
    left = su2_mul(b[1], b[2], b[3], b[4],
        x0[i - 1, j], x1[i - 1, j], x2[i - 1, j], x3[i - 1, j])

    return (right[1] + left[1], right[2] + left[2], right[3] + left[3], right[4] + left[4])
end

@inline function random_su2_left_step(rng, eps)
    alpha1 = eps * (2.0f0 * rand(rng, Float32) - 1.0f0)
    alpha2 = eps * (2.0f0 * rand(rng, Float32) - 1.0f0)
    alpha3 = eps * (2.0f0 * rand(rng, Float32) - 1.0f0)
    return su2_exp_algebra(alpha1, alpha2, alpha3), rand(rng, Float32)
end

@inline function update_x_link!(arrays, I::CartesianIndex{2}, params, rng)
    x0, x1, x2, x3, _, _, _, _ = arrays
    i, j = Tuple(I)
    R, accept = random_su2_left_step(rng, params.proposal_eps)

    u0 = x0[i, j]
    u1 = x1[i, j]
    u2 = x2[i, j]
    u3 = x3[i, j]

    unew = su2_mul(R[1], R[2], R[3], R[4], u0, u1, u2, u3)
    unew = su2_normalize(unew[1], unew[2], unew[3], unew[4])
    staple = staple_x_2d(arrays, i, j)

    old_retr = su2_retr_prod(u0, u1, u2, u3, staple[1], staple[2], staple[3], staple[4])
    new_retr = su2_retr_prod(unew[1], unew[2], unew[3], unew[4],
        staple[1], staple[2], staple[3], staple[4])
    delta_s = -0.5f0 * params.beta * (new_retr - old_retr)

    if delta_s <= 0.0f0 || accept < exp(-delta_s)
        x0[i, j] = unew[1]
        x1[i, j] = unew[2]
        x2[i, j] = unew[3]
        x3[i, j] = unew[4]
    end

    return nothing
end

@inline function update_y_link!(arrays, I::CartesianIndex{2}, params, rng)
    _, _, _, _, y0, y1, y2, y3 = arrays
    i, j = Tuple(I)
    R, accept = random_su2_left_step(rng, params.proposal_eps)

    u0 = y0[i, j]
    u1 = y1[i, j]
    u2 = y2[i, j]
    u3 = y3[i, j]

    unew = su2_mul(R[1], R[2], R[3], R[4], u0, u1, u2, u3)
    unew = su2_normalize(unew[1], unew[2], unew[3], unew[4])
    staple = staple_y_2d(arrays, i, j)

    old_retr = su2_retr_prod(u0, u1, u2, u3, staple[1], staple[2], staple[3], staple[4])
    new_retr = su2_retr_prod(unew[1], unew[2], unew[3], unew[4],
        staple[1], staple[2], staple[3], staple[4])
    delta_s = -0.5f0 * params.beta * (new_retr - old_retr)

    if delta_s <= 0.0f0 || accept < exp(-delta_s)
        y0[i, j] = unew[1]
        y1[i, j] = unew[2]
        y2[i, j] = unew[3]
        y3[i, j] = unew[4]
    end

    return nothing
end

function update_direction_color!(::LocalHaloBackend, U, rng, params, dir::Int, color::Int)
    arrays = link_arrays(U)
    update! = dir == 1 ? update_x_link! : update_y_link!

    for indices in get_interior_cells(CellRanges(U), color)
        @inbounds for I in indices
            update!(arrays, I, params, rng)
        end
    end

    return U
end

function update_direction_color!(::ThreadedHaloBackend, U, rngs, params, dir::Int, color::Int)
    regions = get_interior_cells(CellRanges(U), color)
    update! = dir == 1 ? update_x_link! : update_y_link!

    tforeach(1:tile_count(U); scheduler=:static) do tile_id
        arrays = link_arrays(U, tile_id)
        rng = rngs[tile_id]
        for indices in regions
            @inbounds for I in indices
                update!(arrays, I, params, rng)
            end
        end
    end

    return U
end

function su2_wilson_sweep!(U::ArrayOfHaloArray, rngs, params)
    backend = halo_backend(U)

    for dir in 1:2
        for color in 0:1
            update_direction_color!(backend, U, rngs, params, dir, color)
            synchronize_halo!(U)
        end
    end

    return U
end

@inline su2_adj(u) = (u[1], -u[2], -u[3], -u[4])
@inline su2_mul_tuple(a, b) = su2_mul(a[1], a[2], a[3], a[4], b[1], b[2], b[3], b[4])

function plaquette_sum_count(arrays, indices)
    x0, x1, x2, x3, y0, y1, y2, y3 = arrays
    total = 0.0
    count = 0

    @inbounds for I in indices
        i, j = Tuple(I)
        Ux = (x0[i, j], x1[i, j], x2[i, j], x3[i, j])
        Uy_xp = (y0[i + 1, j], y1[i + 1, j], y2[i + 1, j], y3[i + 1, j])
        Ux_yp_d = su2_adj((x0[i, j + 1], x1[i, j + 1], x2[i, j + 1], x3[i, j + 1]))
        Uy_d = su2_adj((y0[i, j], y1[i, j], y2[i, j], y3[i, j]))
        plaq = su2_mul_tuple(su2_mul_tuple(su2_mul_tuple(Ux, Uy_xp), Ux_yp_d), Uy_d)
        total += Float64(plaq[1])
        count += 1
    end

    return total, count
end

function average_plaquette(U::ArrayOfHaloArray)
    synchronize_halo!(U)
    return average_plaquette(halo_backend(U), U)
end

function average_plaquette(::LocalHaloBackend, U::ArrayOfHaloArray)
    total, count = plaquette_sum_count(link_arrays(U), get_interior_cells(CellRanges(U)))
    return total / count
end

function average_plaquette(::ThreadedHaloBackend, U::ArrayOfHaloArray)
    total = 0.0
    count = 0

    for tile_id in 1:tile_count(U)
        tile_total, tile_count_ = plaquette_sum_count(
            link_arrays(U, tile_id),
            get_interior_cells(CellRanges(U)),
        )
        total += tile_total
        count += tile_count_
    end

    return total / count
end

function run_local_su2_wilson_2d(; n=(32, 32), nsweeps=100, measure_every=20,
        beta=2.0f0, proposal_eps=0.35f0, seed=1234)
    U = initialize_su2_links_2d(LocalHaloArray, n, 1)
    rng = MersenneTwister(seed)
    params = SU2WilsonCPUParams(beta=Float32(beta), proposal_eps=Float32(proposal_eps))

    @printf("2D pure SU(2) Wilson Metropolis with LocalHaloArray\n")
    @printf(" lattice: %d x %d beta=%.4f eps=%.4f\n", n[1], n[2], params.beta, params.proposal_eps)

    for sweep in 1:nsweeps
        su2_wilson_sweep!(U, rng, params)
        sweep % measure_every == 0 &&
            @printf(" local    sweep %6d | <P> = %.8f\n", sweep, average_plaquette(U))
    end

    return U
end

function run_threaded_su2_wilson_2d(; n=(32, 32), nsweeps=100, measure_every=20,
        beta=2.0f0, proposal_eps=0.35f0, seed=1234,
        tile_dims=(Base.Threads.nthreads(), 1))
    U = initialize_su2_links_2d(ThreadedHaloArray, n, 1; tile_dims)
    rngs = [MersenneTwister(seed + tile_id) for tile_id in 1:tile_count(U)]
    params = SU2WilsonCPUParams(beta=Float32(beta), proposal_eps=Float32(proposal_eps))

    @printf("2D pure SU(2) Wilson Metropolis with ThreadedHaloArray\n")
    @printf(" lattice: %d x %d tile_dims=%s threads=%d beta=%.4f eps=%.4f\n",
        n[1], n[2], string(tile_dims), Base.Threads.nthreads(), params.beta, params.proposal_eps)

    for sweep in 1:nsweeps
        su2_wilson_sweep!(U, rngs, params)
        sweep % measure_every == 0 &&
            @printf(" threaded sweep %6d | <P> = %.8f\n", sweep, average_plaquette(U))
    end

    return U
end

function main()
    n = (32, 32)
    nsweeps = 100
    measure_every = 20

    local_U = run_local_su2_wilson_2d(; n, nsweeps, measure_every)
    threaded_U = run_threaded_su2_wilson_2d(; n, nsweeps, measure_every)

    @printf("\nfinal local plaquette:    %.8f\n", average_plaquette(local_U))
    @printf("final threaded plaquette: %.8f\n", average_plaquette(threaded_U))
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
