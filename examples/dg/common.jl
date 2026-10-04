using LinearAlgebra
using StaticArrays
using HaloArrays

# ============================================================
# Nodal discontinuous Galerkin in one dimension: the reference element.
#
# Each element carries the solution at the P = N + 1 Legendre-Gauss-Lobatto
# nodes of the reference interval [-1, 1] (a Lagrange basis of degree N). The
# element-local operators are built once from the Legendre Vandermonde matrix
# (Hesthaven & Warburton, Nodal Discontinuous Galerkin Methods, ch. 3):
#
#     M = (V Vᵀ)⁻¹          mass matrix          ∫ lᵢ lⱼ dr
#     D = Vᵣ V⁻¹            differentiation      ∂ᵣ lⱼ (rᵢ)
#     S = M D               stiffness            ∫ lᵢ ∂ᵣ lⱼ dr
#
# On a physical element of width h the mass matrix scales by h/2 and S is
# invariant, so the strong-form semi-discrete update of a conservation law
# ∂ₜu + ∂ₓf = 0 reads, with the numerical flux f* on the two faces,
#
#     ∂ₜ u = (2/h) M⁻¹ ( -S f + e_P (f_P - f*_R) - e_1 (f_1 - f*_L) ).
#
# Everything is a static array, so a kernel over elements is allocation-free.
# ============================================================

# Legendre polynomial P_n and its derivative at x, by the three-term recurrence
# (the derivative via P'_n = P'_{n-2} + (2n - 1) P_{n-1}, regular at x = ±1).
function legendre(n::Int, x)
    p0, p1 = one(x), x
    d0, d1 = zero(x), one(x)
    n == 0 && return p0, d0
    for k in 2:n
        p0, p1 = p1, ((2k - 1) * x * p1 - (k - 1) * p0) / k
        d0, d1 = d1, d0 + (2k - 1) * p0      # p0 is now P_{k-1}
    end
    return p1, d1
end

# Legendre-Gauss-Lobatto nodes (the zeros of (1 - x²) P'_N) and weights, by
# Newton iteration on q(x) = P_{N+1}(x) - P_{N-1}(x) from a Chebyshev guess.
function lgl_nodes(N::Int)
    N >= 1 || throw(ArgumentError("need polynomial degree N ≥ 1"))
    x = [-cos(π * i / N) for i in 0:N]
    for i in 1:N-1
        for _ in 1:50
            qp1, dp1 = legendre(N + 1, x[i + 1])
            qm1, dm1 = legendre(N - 1, x[i + 1])
            q, dq = qp1 - qm1, dp1 - dm1
            dx = q / dq
            x[i + 1] -= dx
            abs(dx) < 4eps() && break
        end
    end
    w = [2 / (N * (N + 1) * legendre(N, xi)[1]^2) for xi in x]
    return x, w
end

"""
    ReferenceElement(N)

Operators of the degree-`N` nodal DG reference element on `[-1, 1]` at the
`P = N + 1` Legendre-Gauss-Lobatto nodes `r` with quadrature weights `w`:
the mass matrix `M` and its inverse `Minv`, the differentiation matrix `D`,
the stiffness matrix `S = M D`, and the unit vectors `e1`, `eP` that pick the
two face nodes.
"""
struct ReferenceElement{P,T,L}
    r::SVector{P,T}
    w::SVector{P,T}
    M::SMatrix{P,P,T,L}
    Minv::SMatrix{P,P,T,L}
    D::SMatrix{P,P,T,L}
    S::SMatrix{P,P,T,L}
    e1::SVector{P,T}
    eP::SVector{P,T}
end

function ReferenceElement(N::Int)
    P = N + 1
    r, w = lgl_nodes(N)
    V  = [legendre(j, ri)[1] * sqrt((2j + 1) / 2) for ri in r, j in 0:N]   # orthonormal Legendre
    Vr = [legendre(j, ri)[2] * sqrt((2j + 1) / 2) for ri in r, j in 0:N]
    M  = inv(V * V')
    D  = Vr / V
    S  = M * D
    e1 = SVector{P}(i == 1 ? 1.0 : 0.0 for i in 1:P)
    eP = SVector{P}(i == P ? 1.0 : 0.0 for i in 1:P)
    return ReferenceElement(SVector{P}(r), SVector{P}(w), SMatrix{P,P}(M), SMatrix{P,P}(inv(M)),
        SMatrix{P,P}(D), SMatrix{P,P}(S), e1, eP)
end

Base.length(::ReferenceElement{P}) where {P} = P

"Physical positions of the nodes of the element with centre `xc` and width `h`."
@inline node_positions(ref::ReferenceElement, xc, h) = xc .+ (h / 2) .* ref.r

"Cell average of the nodal values `u` of one element (exact LGL quadrature)."
@inline element_average(ref::ReferenceElement, u) = dot(ref.w, u) / 2

# ---- numerical flux ----------------------------------------------------------

"""
    lax_friedrichs(f, df, t, uL, uR)

Local Lax-Friedrichs flux across a face with left state `uL` and right state
`uR`, for a flux `f(t, u)` with derivative `df(t, u)`:
`{{f}} - (C/2)(uR - uL)`, `C = max(|f'(uL)|, |f'(uR)|)`.
"""
@inline function lax_friedrichs(f::F, df::G, t, uL, uR) where {F,G}
    C = max(abs(df(t, uL)), abs(df(t, uR)))
    return (f(t, uL) + f(t, uR)) / 2 - (C / 2) * (uR - uL)
end

# ---- semi-discrete DG operator on a 1-D halo array of nodal vectors ----------

"""
    dg_rhs!(du, u, g, ref, f, df, t)

Strong-form nodal DG right-hand side of `∂ₜu + ∂ₓ f(t, u) = 0` for a 1-D
`LocalHaloArray` whose elements are `SVector{P}` nodal values, with the
element widths from the geometry `g`, the local Lax-Friedrichs flux on the
faces, and the neighbours read from the (one-element) halo.
"""
function dg_rhs!(du, u, g, ref::ReferenceElement{P}, f::F, df::G, t) where {P,F,G}
    synchronize_halo!(u)
    U, dU = parent(u), parent(du)
    e = unit_vector(Val(1), 1)
    @inbounds for I in CartesianIndices(interior_range(u))
        uk, ul, ur = U[I], U[I - e], U[I + e]
        h  = cell_width(g, I)[1]
        fk = f.(t, uk)
        fL = lax_friedrichs(f, df, t, ul[P], uk[1])        # face shared with the left neighbour
        fR = lax_friedrichs(f, df, t, uk[P], ur[1])        # face shared with the right neighbour
        surface = ref.eP * (fk[P] - fR) - ref.e1 * (fk[1] - fL)
        dU[I] = (2 / h) * (ref.Minv * (surface - ref.S * fk))
    end
    return du
end

"""
    ssp_rk3!(u, g, ref, f, df, t, dt, work)

One third-order strong-stability-preserving Runge-Kutta step from `t` to
`t + dt`; `work` is a tuple of three arrays like `u`.
"""
function ssp_rk3!(u, g, ref, f::F, df::G, t, dt, work) where {F,G}
    u1, u2, du = work
    dg_rhs!(du, u, g, ref, f, df, t)
    u1 .= u .+ dt .* du
    dg_rhs!(du, u1, g, ref, f, df, t + dt)
    u2 .= 0.75 .* u .+ 0.25 .* (u1 .+ dt .* du)
    dg_rhs!(du, u2, g, ref, f, df, t + dt / 2)
    u .= (1 / 3) .* u .+ (2 / 3) .* (u2 .+ dt .* du)
    return u
end

"Explicit time step from the CFL condition: `cfl · h_min / (λ_max (2N + 1))`."
function dg_stable_dt(u, g, ref::ReferenceElement{P}, λmax; cfl=0.5) where {P}
    hmin = minimum(cell_width(g, I)[1] for I in CartesianIndices(interior_range(u)))
    return cfl * hmin / (λmax * (2P - 1))
end
