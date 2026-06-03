# ============================================================
# Coupled boundary conditions — characteristic non-reflecting outflow
#
#   julia --project=. examples/finite_volume/acoustics_characteristic_1d.jl
#
# 1-D linear acoustics couples two fields, pressure `p` and velocity `u`:
#
#     ∂p/∂t + ρc² ∂u/∂x = 0
#     ∂u/∂t + (1/ρ) ∂p/∂x = 0
#
# Its characteristics are  w± = p ± ρc·u,  travelling at speeds ±c. A
# *non-reflecting* boundary keeps the OUTGOING characteristic (extrapolated from
# the interior) and sets the INCOMING one to zero — a rule that mixes BOTH fields,
# so it cannot be expressed as an independent per-field boundary condition.
#
# We store (p, u) as an ArrayOfHaloArray, mark the x-boundaries `:noboundary`
# (so synchronize_halo! leaves them alone), and fill them with a coupled BC.
# ============================================================

using HaloArrays
using Printf

# A coupled boundary condition is just a type carrying its parameters.
struct AcousticOutflow <: AbstractCoupledBoundaryCondition
    rho::Float64
    c::Float64
end

# The hook: fill every field's ghost on one (side, dim) from all fields' edges.
# `Side{2}` is the high (right) end → outgoing wave is w⁺ = p + ρc·u;
# `Side{1}` is the low (left) end   → outgoing wave is w⁻ = p − ρc·u.
function HaloArrays.apply_coupled_bc!(bc::AcousticOutflow, state, ::Side{S}, ::Dim{1}) where {S}
    p, u = eachfield(state)
    ρc  = bc.rho * bc.c
    sgn = S == 2 ? 1.0 : -1.0

    p_edge = get_send_view(Side(S), Dim(1), p)[1]   # interior cell at the boundary
    u_edge = get_send_view(Side(S), Dim(1), u)[1]
    w_out  = p_edge + sgn * ρc * u_edge             # outgoing characteristic; incoming = 0

    get_recv_view(Side(S), Dim(1), p)[1] = 0.5 * w_out          # p = (w⁺+w⁻)/2,  w_in=0
    get_recv_view(Side(S), Dim(1), u)[1] = sgn * 0.5 * w_out / ρc  # u = (w⁺−w⁻)/(2ρc)
    return nothing
end

# One explicit-Euler step with a local Lax–Friedrichs flux for the 2×2 system.
function acoustics_step!(state, bc, ρ, c, dt, dx)
    apply_coupled_bc!(bc, state)            # fill the :noboundary x-ghosts coupled-ly
    p, u = eachfield(state)
    pp, uu = parent(p), parent(u)
    rng = interior_range(p)[1]
    ρc2 = ρ * c^2

    Fp(i) = 0.5 * (ρc2 * uu[i] + ρc2 * uu[i+1]) - 0.5 * c * (pp[i+1] - pp[i])
    Fu(i) = 0.5 * (pp[i] / ρ + pp[i+1] / ρ) - 0.5 * c * (uu[i+1] - uu[i])

    pnew = [pp[i] - dt / dx * (Fp(i) - Fp(i - 1)) for i in rng]
    unew = [uu[i] - dt / dx * (Fu(i) - Fu(i - 1)) for i in rng]
    interior_view(p) .= pnew
    interior_view(u) .= unew
    return state
end

# total acoustic energy on the owned cells: ½∫(p²/ρc² + ρu²)
energy(state, ρ, c) = let (p, u) = eachfield(state)
    0.5 * sum(interior_view(p) .^ 2 ./ (ρ * c^2) .+ ρ .* interior_view(u) .^ 2)
end

function run_acoustics(; nx=400, ρ=1.0, c=1.0, cfl=0.8, nt=900)
    dx = 1.0 / nx
    dt = cfl * dx / c
    bc = AcousticOutflow(ρ, c)

    # both x-ends opt out of the per-field BC; we fill them with the coupled BC
    state = ArrayOfHaloArray(LocalHaloArray, Float64, (2,), (nx,), 1;
        boundary_condition=((:noboundary, :noboundary),))
    p, u = eachfield(state)

    # a right-moving Gaussian pulse: set u = p/(ρc) so w⁻ = p − ρc·u = 0
    for i in 1:nx
        x = (i - 0.5) / nx
        pulse = exp(-((x - 0.4) / 0.05)^2)
        p[i] = pulse
        u[i] = pulse / (ρ * c)
    end

    E0 = energy(state, ρ, c)
    for _ in 1:nt
        acoustics_step!(state, bc, ρ, c, dt, dx)
    end
    E1 = energy(state, ρ, c)

    @printf("acoustics: nx=%d  nt=%d  E0=%.4e  E_final=%.4e  retained=%.2e\n",
        nx, nt, E0, E1, E1 / E0)
    println(E1 / E0 < 1e-3 ?
        "  ✓ pulse exited through the non-reflecting boundaries (negligible reflection)" :
        "  ✗ unexpected energy retained")
    return state
end

run_acoustics()
