# ============================================================
# HaloArrays.jl — OrdinaryDiffEq.jl tutorial
#
# Requirements (examples environment):
#   julia --project=examples -e '
#     using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
#
# Run with:
#   julia --project=examples tutorial_diffeq.jl
#
# Sections:
#   1. Why HaloArrays works with OrdinaryDiffEq
#   2. The RHS contract — synchronize_halo! in the RHS
#   3. Solving a scalar decay ODE (LocalHaloArray)
#   4. Heat equation with Tsit5 (adaptive time-stepping)
#   5. Multi-field ODE state (LocalMultiHaloArray)
#   6. ThreadedHaloArray as ODE state
#   7. Comparing DiffEq with manual time-stepping
# ============================================================

using HaloArrays
using DiffEqBase
using OrdinaryDiffEq
using Printf

# ============================================================
# 1. WHY HaloArrays WORKS WITH OrdinaryDiffEq
# ============================================================
#
# OrdinaryDiffEq.jl uses an abstract interface to work with
# arbitrary array types.  Specifically it needs:
#
#   • Base.similar(u)              — allocate workspace arrays
#   • Base.copy(u)                 — snapshot the current state
#   • Base.copyto!(dest, src)      — update state in-place
#   • Broadcast (.=, .+, .-=, …)  — arithmetic stages
#   • DiffEqBase.recursive_length  — count degrees of freedom
#
# HaloArrays implements all of these.  The DiffEqBase extension
# (loaded automatically when both packages are imported) also
# provides `recursive_length`, `NAN_CHECK`, and
# `ODE_DEFAULT_UNSTABLE_CHECK` for every halo container type.
#
# The only obligation on the user is the RHS contract described
# in Section 2.

println("=" ^ 60)
println("HaloArrays.jl — OrdinaryDiffEq tutorial")
println("=" ^ 60)

# ============================================================
# 2. THE RHS CONTRACT — synchronize_halo! IN THE RHS
# ============================================================
#
# OrdinaryDiffEq calls the RHS function f(du, u, p, t) multiple
# times per time step (once per Runge-Kutta stage for explicit
# methods; more for implicit ones).
#
# Between RHS calls, the solver performs arithmetic on `u`
# (forming stage values).  This arithmetic goes through
# broadcast, which only touches interior cells.  Ghost cells
# are therefore left in an unpredictable state after each stage.
#
# The CONTRACT is:
#
#   At the top of every RHS call, before reading any ghost cell,
#   call synchronize_halo!(u) to refresh the ghost layer.
#
# This keeps ghost validity explicit and predictable regardless
# of which solver or time-step algorithm is used.
#
# CORRECT pattern:
#
#   function my_rhs!(du, u, p, t)
#       synchronize_halo!(u)   ← always first
#       # ... stencil work using parent(u) ...
#       return du
#   end
#
# INCORRECT (crashes or gives wrong results):
#
#   function my_rhs!(du, u, p, t)
#       # ... stencil reading parent(u) without refreshing ghosts ...
#   end

println()
println("Section 2 — RHS contract explained (no runnable output)")

# ============================================================
# 3. SOLVING A SCALAR DECAY ODE (LocalHaloArray)
# ============================================================
#
# du/dt = -λ u    (exact solution: u(t) = u₀ exp(-λ t))
#
# This example intentionally does NOT use the stencil (no ghost
# read), but we include synchronize_halo! for good practice and
# to demonstrate the minimal plumbing.

println()
println("=" ^ 60)
println("Section 3 — Scalar decay ODE")
println("=" ^ 60)

function decay_rhs!(du, u, p, _t)
    lambda = p
    synchronize_halo!(u)
    du .= -lambda .* u      # broadcast over interior cells only
    return du
end

function run_decay_example(; nx=8, ny=8, lambda=0.5, tspan=(0.0, 2.0))
    u0 = LocalHaloArray(Float64, (nx, ny), 1; boundary_condition=:repeating)
    # Initial condition: fill with global indices
    fill_from_global_indices!(u0) do I
        1.0 + 0.1 * (I[1] + I[2])
    end

    prob    = ODEProblem(decay_rhs!, u0, tspan, lambda)
    sol     = solve(prob, Tsit5(); reltol=1e-8, abstol=1e-10, save_everystep=false)
    u_final = sol.u[end]

    # Verify against exact solution
    expected_factor = exp(-lambda * last(tspan))
    max_err = maximum(abs, interior_view(u_final) .-
                           expected_factor .* interior_view(u0))

    @printf("  decay: nx=%d×%d  λ=%.2f  t=%.1f  max_error=%.2e\n",
        nx, ny, lambda, last(tspan), max_err)
    return max_err
end

err = run_decay_example()
@assert err < 1e-6 "decay ODE error too large: $err"

# ============================================================
# 4. HEAT EQUATION WITH Tsit5 (ADAPTIVE TIME-STEPPING)
# ============================================================
#
# ∂u/∂t = α ∇²u   on [0,1]²  periodic
#
# The RHS evaluates the finite-difference Laplacian on the
# ghost-padded parent array.  The solver picks its own time
# steps based on the error tolerance.

println()
println("=" ^ 60)
println("Section 4 — Heat equation with adaptive Tsit5")
println("=" ^ 60)

function heat_laplacian_rhs!(du, u, p, _t)
    alpha, dx, dy = p
    synchronize_halo!(u)

    data  = parent(u)
    ddata = parent(du)
    ex = CartesianIndex(1, 0)
    ey = CartesianIndex(0, 1)
    inv_dx2 = alpha / dx^2
    inv_dy2 = alpha / dy^2

    for I in CartesianIndices(interior_range(u))
        ddata[I] = (data[I+ex] - 2*data[I] + data[I-ex]) * inv_dx2 +
                   (data[I+ey] - 2*data[I] + data[I-ey]) * inv_dy2
    end
    return du
end

function run_heat_diffeq(; n=(32, 32), alpha=0.01, tspan=(0.0, 0.5),
        reltol=1e-5, abstol=1e-7)
    dx = 1.0 / n[1]
    dy = 1.0 / n[2]

    u0 = LocalHaloArray(Float64, n, 1; boundary_condition=:periodic)
    fill_from_global_indices!(u0) do I
        cx, cy = (n[1]+1)/2.0, (n[2]+1)/2.0
        r2 = ((I[1]-cx)/(n[1]/6))^2 + ((I[2]-cy)/(n[2]/6))^2
        return exp(-r2)
    end

    p    = (alpha, dx, dy)
    prob = ODEProblem{true}(heat_laplacian_rhs!, u0, tspan, p)
    sol  = solve(prob, Tsit5(); reltol, abstol, saveat=0.1)

    u_end        = sol.u[end]
    total_energy = sum(interior_view(u_end))
    peak         = maximum(interior_view(u_end))

    @printf("  heat DiffEq: n=%dx%d  steps=%d  t=%.2f  peak=%.4f  total=%.4f\n",
        n..., length(sol.t)-1, last(sol.t), peak, total_energy)
    return sol
end

sol_heat = run_heat_diffeq()

# Access saved snapshots
println("  saved time points: ", round.(sol_heat.t; digits=2))
println("  typeof(sol.u[1])  : ", typeof(sol_heat.u[1]))   # LocalHaloArray

# ============================================================
# 5. MULTI-FIELD ODE STATE (LocalMultiHaloArray)
# ============================================================
#
# Pass a MultiHaloArray (or LocalMultiHaloArray) directly as the
# ODE state.  The solver treats it as a single composite array.
# Access named fields inside the RHS through the named-tuple API.
#
# Example: coupled system for two species (reaction-diffusion)
#   ∂A/∂t = DA ∇²A - k A B
#   ∂B/∂t = DB ∇²B - k A B

println()
println("=" ^ 60)
println("Section 5 — Multi-field ODE state")
println("=" ^ 60)

function reaction_diffusion_rhs!(du, u, p, _t)
    DA, DB, k, dx = p
    synchronize_halo!(u)    # refreshes BOTH fields at once

    dA = parent(du.A)
    dB = parent(du.B)
    A  = parent(u.A)
    B  = parent(u.B)
    ex = CartesianIndex(1, 0)
    ey = CartesianIndex(0, 1)
    inv_dx2 = inv(dx^2)

    for I in CartesianIndices(interior_range(u.A))
        lapA = (A[I+ex] - 2*A[I] + A[I-ex] + A[I+ey] - 2*A[I] + A[I-ey]) * inv_dx2
        lapB = (B[I+ex] - 2*B[I] + B[I-ex] + B[I+ey] - 2*B[I] + B[I-ey]) * inv_dx2
        reaction = k * A[I] * B[I]
        dA[I] = DA * lapA - reaction
        dB[I] = DB * lapB - reaction
    end
    return du
end

function run_reaction_diffusion(; n=(24, 24), tspan=(0.0, 0.2))
    dx  = 1.0 / n[1]
    bc  = ((Periodic(), Periodic()), (Periodic(), Periodic()))
    u0  = LocalMultiHaloArray(Float64, n, 1;
        boundary_conditions=(A=bc, B=bc))

    fill_from_global_indices!(u0.A) do I
        0.5 + 0.1*sin(2π*I[1]/n[1])
    end
    fill_from_global_indices!(u0.B) do I
        0.5 + 0.1*cos(2π*I[2]/n[2])
    end

    p    = (0.1, 0.05, 1.0, dx)
    prob = ODEProblem{true}(reaction_diffusion_rhs!, u0, tspan, p)
    sol  = solve(prob, Tsit5(); reltol=1e-4, abstol=1e-6, save_everystep=false)

    u_end = sol.u[end]
    @printf("  reaction-diffusion: n=%dx%d  maxA=%.4f  maxB=%.4f\n",
        n..., maximum(u_end.A), maximum(u_end.B))
    return sol
end

run_reaction_diffusion()

# ============================================================
# 6. ThreadedHaloArray AS ODE STATE
# ============================================================
#
# Swap LocalHaloArray for ThreadedHaloArray — no other change
# needed.  The solver uses the same broadcast and copyto!
# interface; the parallel tile dispatch is transparent.
#
# The RHS must still call synchronize_halo!(u) at the top.
# For ThreadedHaloArray this synchronises the tile ghost cells
# in shared memory (no MPI involved).

println()
println("=" ^ 60)
println("Section 6 — ThreadedHaloArray as ODE state")
println("=" ^ 60)

function run_heat_diffeq_threaded(;
        nx=32, alpha=0.01, tspan=(0.0, 0.2),
        tile_dims=(max(1, Threads.nthreads()),))

    nx % tile_dims[1] == 0 ||
        throw(ArgumentError("nx must be divisible by tile_dims[1]"))

    tile_size = (nx ÷ tile_dims[1],)
    dx = 1.0 / nx

    u0 = ThreadedHaloArray(Float64, tile_size, 1;
        dims=tile_dims, boundary_condition=:periodic)

    # Fill using logical global index
    for I in CartesianIndices(axes(u0))
        u0[Tuple(I)...] = exp(-50*((I[1]/nx) - 0.5)^2)
    end

    # RHS for 1-D heat on a ThreadedHaloArray
    function threaded_heat_rhs!(du, u, p, _t)
        alpha_p, dx_p = p
        synchronize_halo!(u)   # tile ghost exchange + BCs
        data  = parent(u)
        ddata = parent(du)
        e = CartesianIndex(1)
        for I in CartesianIndices(interior_range(u))
            ddata[I] = alpha_p * (data[I+e] - 2*data[I] + data[I-e]) / dx_p^2
        end
        return du
    end

    p    = (alpha, dx)
    prob = ODEProblem{true}(threaded_heat_rhs!, u0, tspan, p)
    sol  = solve(prob, Tsit5(); reltol=1e-5, abstol=1e-7, save_everystep=false)

    u_end = sol.u[end]
    @printf("  threaded heat: nx=%d  tiles=%d  peak=%.4f\n",
        nx, tile_count(u_end), maximum(u_end))
    return sol
end

run_heat_diffeq_threaded()

# ============================================================
# 7. COMPARING DiffEq WITH MANUAL TIME-STEPPING
# ============================================================
#
# Use this to validate your RHS implementation: the manual
# explicit-Euler solution (or any reference) should agree with
# the DiffEq solution to within the chosen tolerance.

println()
println("=" ^ 60)
println("Section 7 — Comparing DiffEq with manual solver")
println("=" ^ 60)

function manual_heat_euler(u0, alpha, dx, dt, nt)
    current = copy(u0)
    nxt     = similar(u0)
    e       = CartesianIndex(1, 0)
    f       = CartesianIndex(0, 1)
    inv_dx2 = alpha / dx^2

    for _ in 1:nt
        synchronize_halo!(current)
        data  = parent(current)
        dnext = parent(nxt)
        for I in CartesianIndices(interior_range(current))
            dnext[I] = data[I] + dt * (
                (data[I+e] - 2*data[I] + data[I-e]) * inv_dx2 +
                (data[I+f] - 2*data[I] + data[I-f]) * inv_dx2)
        end
        current, nxt = nxt, current
    end
    synchronize_halo!(current)
    return current
end

function compare_solvers(; n=(32, 32), alpha=0.01, tspan=(0.0, 0.1))
    dx = 1.0 / n[1]
    # Stable explicit step for reference
    dt = 0.4 * dx^2 / (2 * alpha)
    nt = round(Int, last(tspan) / dt)

    u0 = LocalHaloArray(Float64, n, 1; boundary_condition=:periodic)
    fill_from_global_indices!(u0) do I
        exp(-50*(((I[1]-n[1]/2)/n[1])^2 + ((I[2]-n[2]/2)/n[2])^2))
    end

    # Manual Euler reference
    u_manual = manual_heat_euler(u0, alpha, dx, dt, nt)

    # DiffEq with tighter tolerance than Euler error
    p    = (alpha, dx, dx)
    prob = ODEProblem{true}(heat_laplacian_rhs!, copy(u0), tspan, p)
    sol  = solve(prob, Tsit5(); reltol=1e-7, abstol=1e-9, save_everystep=false)
    u_diffeq = sol.u[end]

    max_diff = maximum(abs, interior_view(u_diffeq) .- interior_view(u_manual))

    @printf("  comparison: n=%dx%d  Euler_nt=%d  max_diff=%.2e\n",
        n..., nt, max_diff)
    return max_diff
end

diff = compare_solvers()
println("  DiffEq vs Euler agree within: ", round(diff; sigdigits=3))

println()
println("DiffEq tutorial complete.")
