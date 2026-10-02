using Printf
using StaticArrays

include("common.jl")

# ============================================================
# The O(N) model at large N as a conservation law, solved with nodal DG
# (Grossi & Wink, "Resolving phase transitions with Discontinuous Galerkin
# methods", arXiv:1903.09503, Sections II–III A).
#
# In the local potential approximation the flow of u(t, ρ) = ∂_ρ V(t, ρ) with
# RG time t = -ln(k/Λ) is
#
#     ∂_t u + ∂_ρ f(t, u) = 0,      f(t, u) = A_d (Λ e^{-t})^{d+2} / ((Λ e^{-t})² + u),
#
# a scalar conservation law with a time-dependent convex flux whose wave speed
# ∂_u f < 0 carries information towards smaller ρ. The Riemann problem with
# u_L > u_R develops a shock whose position follows the Rankine-Hugoniot
# condition dξ/dt = [f]/[u]; for d = 3, Λ = 1, u_R = 0 it integrates to
# equation (29) of the paper:
#
#     ξ(t) = ξ₀ + (1/6π²) [ e^{-t} − 1 + √u_L ( acot √u_L − acot(e^t √u_L) ) ].
#
# The script evolves the initial condition (32) with K elements of degree N
# and compares the numerical shock position against ξ(t): once through the
# conserved mass left of the shock (what the Rankine-Hugoniot condition
# really states), once through the 0.05 level crossing of the nodal solution
# (the discontinuity must stay within an element or two). The halo array
# holds one SVector of nodal values per element; the ghost elements are the
# neighbours a face flux needs.
# ============================================================

const DIM = 3
const Λ   = 1.0
const A_d = 1 / (6π^2)      # A_d = Ω_d (2π)^{-d} / d with Ω_3 = 4π

scale(t)   = Λ * exp(-t)
flux(t, u) = (k = scale(t); A_d * k^(DIM + 2) / (k^2 + u))
dflux(t, u) = (k = scale(t); -A_d * k^(DIM + 2) / (k^2 + u)^2)

# Equation (29): the shock position for d = 3, Λ = 1, u_R = 0.
acot(x) = atan(1 / x)
shock_position(t, ξ0, uL) =
    ξ0 + (exp(-t) - 1 + sqrt(uL) * (acot(sqrt(uL)) - acot(exp(t) * sqrt(uL)))) / (6π^2)

# Initial condition (32): a shock-forming jump at ρ = 0.02 and a rarefaction-
# forming jump at ρ = 0.05.
const U_L, ξ0, ρ_MAX = 0.1, 0.02, 0.08
u_initial(ρ) = (ρ < 0.02 || ρ > 0.05) ? U_L : 0.0

# ---- diagnostics -----------------------------------------------------------

# Shock position from conservation: left of the cut the solution is u_L up to
# the shock and 0 beyond it (the rarefaction fan from ρ = 0.05 stays right of
# 0.035 for t ≤ 3), so ∫₀ᶜ u dρ = u_L ξ. The integral is the exact LGL
# quadrature of the DG solution.
function shock_from_mass(u, g, ref; cut=0.03)
    U = parent(u)
    mass = 0.0
    for I in CartesianIndices(interior_range(u))
        c, h = cell_center(Cartesian(), g, I)[1], cell_width(Cartesian(), g, I)[1]
        c < cut || continue
        mass += h * element_average(ref, U[I])
    end
    return mass / U_L
end

# Shock position from the first downward crossing of u_L / 2 in the nodal
# solution, linearly interpolated between the two nodes that bracket it.
function shock_from_crossing(u, g, ref)
    U = parent(u)
    level = U_L / 2
    for I in CartesianIndices(interior_range(u))
        c, h = cell_center(Cartesian(), g, I)[1], cell_width(Cartesian(), g, I)[1]
        x, v = node_positions(ref, c, h), U[I]
        for i in 1:length(v)-1
            if v[i] >= level > v[i + 1]
                return x[i] + (x[i + 1] - x[i]) * (v[i] - level) / (v[i] - v[i + 1])
            end
        end
    end
    return NaN
end

# ---- driver ------------------------------------------------------------------

function run_riemann(; K=400, N=3, t_end=3.0, cfl=0.5, report_at=(0.5, 1.0, 2.0, 3.0))
    P   = N + 1
    ref = ReferenceElement(N)
    # one element per site, nodal values as the element type; the inflow
    # boundary (large ρ) keeps the initial state, the outflow boundary copies
    # its neighbour — both far from the waves for t ≤ 3
    inflow = FunctionBC((ghost, edge, side, dim, hw, origin) -> (ghost .= Ref(SVector{P}(fill(U_L, P))); nothing))
    u = LocalHaloArray(SVector{P,Float64}, (K,), 1;
        boundary_condition=((Repeating(), inflow),))
    g = cell_geometry(u, UniformAxis(0, ρ_MAX))
    # (32) is piecewise constant with its jumps on element edges (K a multiple
    # of 80), so the element-wise value at the centre represents it exactly —
    # sampling the jump at a shared face node would misplace h/12 of mass
    K % 80 == 0 || throw(ArgumentError("K must be a multiple of 80 so the jumps fall on element edges"))
    fill_from_global_indices!(u) do I
        c = cell_center(Cartesian(), g, CartesianIndex(I[1] + 1))[1]   # storage index = global + halo
        SVector{P}(fill(u_initial(c), P))
    end
    work = (similar(u), similar(u), similar(u))

    λmax = maximum(abs(dflux(0.0, v)) for v in (0.0, U_L))      # speeds only decrease with t
    dt   = dg_stable_dt(u, g, ref, λmax; cfl)
    h    = ρ_MAX / K

    rows = Tuple{Float64,Float64,Float64,Float64}[]
    t = 0.0
    for t_report in report_at
        while t < t_report - 1e-12
            step = min(dt, t_report - t)
            ssp_rk3!(u, g, ref, flux, dflux, t, step, work)
            t += step
        end
        push!(rows, (t, shock_position(t, ξ0, U_L), shock_from_mass(u, g, ref), shock_from_crossing(u, g, ref)))
    end
    return rows, h, dt
end

function main()
    rows, h, dt = run_riemann()
    @printf("DG, K=400 elements of degree 3 on [0, %.2f], dt=%.2e; shock of the Riemann problem (32)\n", ρ_MAX, dt)
    @printf("%6s  %12s  %12s  %12s  %10s  %10s\n", "t", "analytic", "from mass", "from crossing", "err mass", "err cross")
    for (t, ξ, ξm, ξc) in rows
        @printf("%6.2f  %12.7f  %12.7f  %12.7f  %10.1e  %10.1e\n", t, ξ, ξm, ξc, ξm - ξ, ξc - ξ)
    end
    for (t, ξ, ξm, ξc) in rows
        abs(ξm - ξ) < 1e-6 || error("t=$t: shock position from mass off by $(ξm - ξ) (Rankine-Hugoniot violated)")
        abs(ξc - ξ) < 2h   || error("t=$t: level crossing off by $(ξc - ξ), more than two elements")
    end
    println("shock position matches equation (29): conserved mass to 1e-6, level crossing within 2h = ", 2h)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
