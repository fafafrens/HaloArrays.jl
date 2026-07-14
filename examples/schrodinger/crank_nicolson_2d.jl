# ============================================================
# HaloArrays.jl — 2-D time-dependent Schrödinger equation
#
# Run with:
#   julia --project=examples -t 4 examples/schrodinger/crank_nicolson_2d.jl
#
# A coherent Gaussian wave packet evolves in an isotropic harmonic trap,
#
#     i ∂ψ/∂t = Hψ,       H = -1/2 ∇² + 1/2 ω²(x² + y²).
#
# Crank–Nicolson advances one time step by solving the linear system
#
#     (I + i Δt H/2) ψⁿ⁺¹ = (I - i Δt H/2) ψⁿ.
#
# The whole example needs only ONE stencil kernel: everything we ever apply
# to ψ has the form  out = β·ψ + α·Hψ  for some pair of scalars (α, β) —
#
#     Hψ itself          →  α = 1,        β = 0
#     right-hand side    →  α = -iΔt/2,   β = 1
#     left-hand side     →  α = +iΔt/2,   β = 1
#
# so `shifted_hamiltonian!` below is the single piece of numerics, and the
# same code runs unchanged on LocalHaloArray (serial) and ThreadedHaloArray
# (one tile per thread). The checks at the end use three pieces of physics:
# probability and energy are conserved, and the packet centre follows the
# classical circular harmonic-oscillator orbit.
# ============================================================

using HaloArrays
using LinearSolve
using Krylov                 # loading it activates HaloArraysLinearSolveExt
using LinearAlgebra: dot
using Printf

# ψ = 0 outside the box: antireflecting ghosts give a Dirichlet wall.
const DIRICHLET_2D = (
    (Antireflecting(), Antireflecting()),
    (Antireflecting(), Antireflecting()),
)

"""
    shifted_hamiltonian!(out, ψ, p, α, β) -> out

Compute `out = β*ψ + α*H*ψ` in one sweep, without a temporary for `H*ψ`.
`H` is the finite-difference Hamiltonian: 5-point Laplacian plus the trap
potential stored in `p.potential`. First refresh the ghost cells, then apply
the stencil tile by tile — a `LocalHaloArray` is a single tile run serially,
a `ThreadedHaloArray` runs one tile per thread.
"""
function shifted_hamiltonian!(out, ψ, p, α, β)
    out === ψ && throw(ArgumentError("the Hamiltonian application is not alias-safe"))
    synchronize_halo!(ψ)
    ex, ey = CartesianIndex(1, 0), CartesianIndex(0, 1)

    stencil! = function (tile)
        o = tile_parent(out, tile)
        u = tile_parent(ψ, tile)
        V = tile_parent(p.potential, tile)
        @inbounds for I in CartesianIndices(interior_range(ψ, tile))
            uI = u[I]
            laplacian = (u[I + ex] - 2uI + u[I - ex]) * p.inv_hx2 +
                        (u[I + ey] - 2uI + u[I - ey]) * p.inv_hy2
            o[I] = β * uI + α * (-0.5 * laplacian + V[I] * uI)
        end
        return nothing
    end

    backend = ψ isa ThreadedHaloArray ? thread_backend(ψ) : SerialBackend()
    tile_foreach(backend, stencil!, 1:tile_count(ψ); scheduler = :static)
    return out
end

"""
    observables(ψ, x, y, p, cell_area)

Total probability ∫|ψ|², the packet centre (⟨x⟩, ⟨y⟩), and the energy ⟨ψ|H|ψ⟩,
all as discrete sums over the grid. `dot` and `mapreduce` reduce over interior
cells only, so the ghost layers never pollute the physics.
"""
function observables(ψ, x, y, p, cell_area)
    probability = cell_area * real(dot(ψ, ψ))
    xmean = cell_area * mapreduce((z, q) -> abs2(z) * q, +, ψ, x) / probability
    ymean = cell_area * mapreduce((z, q) -> abs2(z) * q, +, ψ, y) / probability
    Hψ = shifted_hamiltonian!(similar(ψ), ψ, p, 1, 0)
    energy = cell_area * real(dot(ψ, Hψ)) / probability
    return (; probability, energy, centre = (xmean, ymean))
end

"""
    evolve!(ψ, p; steps, reltol, abstol) -> ψ

Advance `steps` Crank–Nicolson steps. The left-hand operator is matrix-free
(a `FunctionOperator` that just calls the stencil), and the LinearSolve cache
is built ONCE: each step overwrites `cache.b` with the new right-hand side and
calls `solve!` again, so the GMRES workspace is reused and the previous ψ
warm-starts the next solve.
"""
function evolve!(ψ, p; steps, reltol = 1.0e-11, abstol = 1.0e-13)
    rhs = shifted_hamiltonian!(similar(ψ), ψ, p, -p.half_i_dt, 1)
    lhs = FunctionOperator(
        (out, v, _, q, _) -> shifted_hamiltonian!(out, v, q, q.half_i_dt, 1),
        similar(ψ), similar(ψ);
        islinear = true, isconstant = true, p,
    )
    cache = init(
        LinearProblem(lhs, rhs; u0 = copy(ψ)),
        HaloGMRES(restart = 20);
        reltol, abstol, maxiters = 100,
    )

    for step in 1:steps
        step == 1 || shifted_hamiltonian!(cache.b, ψ, p, -p.half_i_dt, 1)
        sol = solve!(cache)
        sol.retcode == ReturnCode.Success ||
            error("HaloGMRES failed at Schrödinger step $step: $(sol.retcode)")
        copyto!(ψ, sol.u)
    end
    return ψ
end

"""
    solve_backend(ψ; lengths, ω, dt, steps, centre)

Build the grid geometry, the trap potential, and the initial coherent state on
the given backend, then evolve and measure. A coherent state is the harmonic
ground state displaced to `centre` and boosted so that it orbits the origin —
its centre must follow the classical trajectory exactly.
"""
function solve_backend(ψ; lengths, ω, dt, steps, centre)
    nx, ny = global_size(ψ)
    hx, hy = lengths[1] / nx, lengths[2] / ny
    momentum = (-ω * centre[2], ω * centre[1])   # tangential kick → circular orbit

    # Cell-centred coordinates as plain Float64 halo arrays; the potential and
    # the initial state are then ordinary (interior-only) broadcasts over them.
    x = similar(ψ, Float64)
    y = similar(ψ, Float64)
    fill_from_global_indices!(I -> -lengths[1] / 2 + (I[1] - 0.5) * hx, x)
    fill_from_global_indices!(I -> -lengths[2] / 2 + (I[2] - 0.5) * hy, y)
    potential = similar(ψ, Float64)
    potential .= 0.5 .* ω^2 .* (x .^ 2 .+ y .^ 2)

    ψ .= sqrt(ω / π) .*
        exp.(-0.5 .* ω .* ((x .- centre[1]) .^ 2 .+ (y .- centre[2]) .^ 2)) .*
        cis.(momentum[1] .* x .+ momentum[2] .* y)
    cell_area = hx * hy
    ψ .= ψ ./ sqrt(cell_area * real(dot(ψ, ψ)))   # unit probability on the grid

    p = (; potential, inv_hx2 = inv(hx^2), inv_hy2 = inv(hy^2), half_i_dt = 0.5im * dt)
    initial = observables(ψ, x, y, p, cell_area)
    evolve!(ψ, p; steps)
    final = observables(ψ, x, y, p, cell_area)
    return (; ψ, initial, final, cell_area)
end

function main(;
        n = 48, lengths = (12.0, 12.0), ω = 1.0, dt = 0.025, steps = 32,
        tile_dims = (1, 4), centre = (1.4, 0.0)
    )
    all(n % d == 0 for d in tile_dims) ||
        throw(ArgumentError("n=$n must be divisible by each tile dimension $tile_dims"))

    local_ψ = LocalHaloArray(ComplexF64, (n, n), 1; boundary_condition = DIRICHLET_2D)
    threaded_ψ = ThreadedHaloArray(
        ComplexF64, (n ÷ tile_dims[1], n ÷ tile_dims[2]), 1;
        dims = tile_dims, boundary_condition = DIRICHLET_2D,
    )

    local_result = solve_backend(local_ψ; lengths, ω, dt, steps, centre)
    threaded_result = solve_backend(threaded_ψ; lengths, ω, dt, steps, centre)

    # After time t the classical centre has rotated by ωt.
    time = steps * dt
    expected_centre = (
        centre[1] * cos(ω * time) - centre[2] * sin(ω * time),
        centre[2] * cos(ω * time) + centre[1] * sin(ω * time),
    )

    @printf("2-D harmonic-oscillator coherent state: n=%d, t=%.3f, Δt=%.3f\n", n, time, dt)
    println("backend       probability drift    energy drift       centre error")
    for (name, result) in (("Local", local_result), ("Threaded", threaded_result))
        probability_drift = abs(result.final.probability - result.initial.probability)
        energy_drift = abs(result.final.energy - result.initial.energy)
        centre_error = hypot(
            result.final.centre[1] - expected_centre[1],
            result.final.centre[2] - expected_centre[2],
        )
        @printf(
            "%-10s    %12.3e       %12.3e       %12.3e\n",
            name, probability_drift, energy_drift, centre_error,
        )
        # Crank–Nicolson is unitary: probability and energy must be conserved
        # to solver tolerance; the centre error is the O(h², Δt²) discretisation.
        abs(result.final.probability - 1) < 1.0e-8 ||
            error("$name probability drift is too large")
        energy_drift < 1.0e-8 || error("$name energy drift is too large")
        centre_error < 6.0e-2 ||
            error("$name centre does not follow the expected classical orbit")
    end

    backend_l2_error = sqrt(
        local_result.cell_area * mapreduce(
            (a, b) -> abs2(a - b), +,
            local_result.ψ, LocalHaloArray(threaded_result.ψ),
        )
    )
    @printf("Local/Threaded weighted L2 difference: %.3e\n", backend_l2_error)
    @printf("expected centre: (%.6f, %.6f)\n", expected_centre...)
    backend_l2_error < 1.0e-8 ||
        error("Local and Threaded solutions disagree: $backend_l2_error")

    return (; local_result, threaded_result, expected_centre, backend_l2_error)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
