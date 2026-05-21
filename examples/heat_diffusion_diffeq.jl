# Heat diffusion solved with OrdinaryDiffEq.jl (Tsit5 adaptive solver)
#
# Demonstrates using LocalHaloArray as the ODE state in an ODEProblem.
# The RHS function explicitly refreshes the halo after each evaluation so
# that ghost cells are always up to date before the stencil is applied.
#
# Usage (from the repo root):
#   julia --project examples/heat_diffusion_diffeq.jl

using HaloArrays
using DiffEqBase
using OrdinaryDiffEq
include("heat_diffusion_common.jl")

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
const N_HEAT = (64, 64)          # interior grid size
const ALPHA  = 0.01              # thermal diffusivity
const DX     = (1.0 / N_HEAT[1], 1.0 / N_HEAT[2])  # grid spacing
const TSPAN  = (0.0, 0.5)

# ---------------------------------------------------------------------------
# ODE right-hand side: du/dt = α ∇²u
#
# The LocalHaloArray halo holds ghost cells needed by the finite-difference
# stencil.  We refresh them at the start of each RHS call so that the stencil
# always reads valid data.  This is the standard pattern for HaloArrays.
# ---------------------------------------------------------------------------
function heat_rhs!(du, u, p, t)
    alpha, dx = p

    # Refresh ghost cells (boundary condition fill + interior copy at edges)
    synchronize_halo!(u)

    heat_step!(du, u, alpha, 1.0, dx)   # re-use heat_step! as a stencil kernel;
                                          # dt=1 so the output IS the Laplacian × α
    return du
end

# heat_step! computes  du = u + alpha*dt*Laplacian(u).
# We want just alpha*Laplacian(u), so we subtract u afterwards.
function _laplacian_rhs!(du, u, p, t)
    alpha, dx = p
    synchronize_halo!(u)
    heat_step!(du, u, alpha, 1.0, dx)
    # du currently holds u + alpha*∇²u; subtract u to get the rate
    interior_view(du) .-= interior_view(u)
    return du
end

# ---------------------------------------------------------------------------
# Initial condition: centred Gaussian
# ---------------------------------------------------------------------------
function make_initial_condition()
    u0 = LocalHaloArray(Float64, N_HEAT, 1; boundary_condition=:periodic)
    fill_centered_gaussian!(u0; baseline=0.0, amplitude=1.0)
    return u0
end

# ---------------------------------------------------------------------------
# Solve with Tsit5 (adaptive)
# ---------------------------------------------------------------------------
function run_heat_diffeq(; reltol=1e-4, abstol=1e-6, saveat=0.05)
    u0 = make_initial_condition()

    p  = (ALPHA, DX)
    prob = ODEProblem{true}(_laplacian_rhs!, u0, TSPAN, p)

    @info "Solving 2-D heat equation with Tsit5 on $(N_HEAT) grid…"
    sol = solve(prob, Tsit5(); reltol=reltol, abstol=abstol, saveat=saveat)

    if sol.retcode != ReturnCode.Success
        @warn "Solver did not succeed: $(sol.retcode)"
    else
        @info "Solved successfully.  $(length(sol.t)) saved time steps, t ∈ $(extrema(sol.t))"
    end

    # Simple scalar diagnostics
    u_end = sol.u[end]
    total_energy = sum(interior_view(u_end))
    peak_value   = maximum(interior_view(u_end))
    @info "Final state — total: $(round(total_energy; sigdigits=5)), peak: $(round(peak_value; sigdigits=5))"

    return sol
end

# ---------------------------------------------------------------------------
# Compare DiffEq solution against the manual time-stepping reference
# ---------------------------------------------------------------------------
function compare_with_manual(; nt=200)
    u0     = make_initial_condition()
    u_ref  = copy(u0)
    u_test = copy(u0)

    # Manual reference
    dt_stable = stable_heat_dt(ALPHA, 0.4, DX)
    actual_nt = round(Int, last(TSPAN) / dt_stable)
    solve_heat!(u_ref; alpha=ALPHA, dt=dt_stable, dx=DX, nt=actual_nt)

    # DiffEq solution at the same final time
    p    = (ALPHA, DX)
    prob = ODEProblem{true}(_laplacian_rhs!, u_test, TSPAN, p)
    sol  = solve(prob, Tsit5(); reltol=1e-5, abstol=1e-7, save_everystep=false)
    u_diffeq = sol.u[end]

    max_err = maximum(abs.(interior_view(u_diffeq) .- interior_view(u_ref)))
    @info "Max absolute difference between DiffEq and manual solver: $(max_err)"
    return max_err
end

if abspath(PROGRAM_FILE) == @__FILE__
    sol = run_heat_diffeq()
    err = compare_with_manual()
    println("Comparison error: ", err)
end
