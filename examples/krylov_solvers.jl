# ============================================================
# Coordinate-free Krylov solvers for the HaloArrays examples
#
# These reimplement the standard algorithms (Saad, "Iterative Methods for
# Sparse Linear Systems") — they are NOT copied from Krylov.jl. They are
# written to be coordinate-free: they touch the unknown only through
#
#     similar, copy, copyto!, fill!, broadcasting,
#     LinearAlgebra.dot, LinearAlgebra.norm, and mul!(y, A, x)
#
# so they run unchanged on a plain Array, a LocalHaloArray, an MPI
# HaloArray, or a ThreadedHaloArray. Because HaloArrays.jl defines dot/norm
# as GLOBAL reductions (MPI Allreduce / threaded tile reduce), every solve
# here is automatically correct across ranks and tiles — no flat vector and
# no special distributed handling. `A` may be a matrix or any object with
# mul! + size (e.g. a SciMLOperators.FunctionOperator).
# ============================================================

using LinearAlgebra: dot, norm, mul!

# ---- Conjugate Gradient — symmetric positive definite ----
function cg!(x, A, b; tol=1e-10, maxiter=length(b))
    Ap = similar(b)
    mul!(Ap, A, x)
    r = copy(b); r .-= Ap          # r = b - A*x
    p = copy(r)
    rsold = real(dot(r, r))
    bnorm = max(norm(b), eps(float(real(eltype(b)))))
    res = sqrt(rsold); iters = 0
    for k in 1:maxiter
        iters = k
        mul!(Ap, A, p)
        α = rsold / real(dot(p, Ap))
        x .+= α .* p
        r .-= α .* Ap
        rsnew = real(dot(r, r))
        res = sqrt(rsnew)
        res ≤ tol * bnorm && break
        p .= r .+ (rsnew / rsold) .* p
        rsold = rsnew
    end
    return x, iters, res
end

# ---- BiCGStab — general square systems (short recurrence) ----
function bicgstab!(x, A, b; tol=1e-10, maxiter=length(b))
    r = copy(b); tmp = similar(b)
    mul!(tmp, A, x); r .-= tmp     # r = b - A*x
    rhat = copy(r)
    p = similar(b); fill!(p, 0)
    v = similar(b); fill!(v, 0)
    s = similar(b); t = similar(b)
    ρ_old = α = ω = one(real(eltype(b)))
    bnorm = max(norm(b), eps(float(real(eltype(b)))))
    res = norm(r); iters = 0
    for k in 1:maxiter
        iters = k
        ρ = dot(rhat, r)
        β = (ρ / ρ_old) * (α / ω)
        p .-= ω .* v               # p = r + β(p - ω v), in simple steps
        p .*= β
        p .+= r
        mul!(v, A, p)
        α = ρ / dot(rhat, v)
        s .= r .- α .* v
        res = norm(s)
        if res ≤ tol * bnorm
            x .+= α .* p
            break
        end
        mul!(t, A, s)
        ω = dot(t, s) / dot(t, t)
        x .+= α .* p
        x .+= ω .* s
        r .= s .- ω .* t
        res = norm(r)
        res ≤ tol * bnorm && break
        ρ_old = ρ
    end
    return x, iters, res
end

# ---- GMRES with restart — general square systems ----
function gmres!(x, A, b; tol=1e-10, restart=30, maxiter=length(b))
    T = float(real(eltype(b)))
    bnorm = max(norm(b), eps(T))
    m = restart
    V  = [similar(b) for _ in 1:m+1]      # Arnoldi basis (the only "big" storage)
    H  = zeros(T, m + 1, m)               # Hessenberg — small & dense
    g  = zeros(T, m + 1)
    cs = zeros(T, m); sn = zeros(T, m)
    w  = similar(b); Ax = similar(b)
    res = bnorm; total = 0
    for _cycle in 1:cld(maxiter, m)
        mul!(Ax, A, x)
        r = V[1]; copyto!(r, b); r .-= Ax  # reuse V[1] as the residual
        β = norm(r)
        res = β
        res ≤ tol * bnorm && break
        V[1] .*= inv(β)
        fill!(g, 0); g[1] = β
        k = 0
        for j in 1:m
            k = j; total += 1
            mul!(w, A, V[j])
            for i in 1:j                   # modified Gram-Schmidt
                H[i, j] = real(dot(V[i], w))
                w .-= H[i, j] .* V[i]
            end
            H[j+1, j] = norm(w)
            H[j+1, j] > eps(T) && (V[j+1] .= w .* inv(H[j+1, j]))
            for i in 1:j-1                 # apply previous Givens rotations
                τ          =  cs[i] * H[i, j] + sn[i] * H[i+1, j]
                H[i+1, j]  = -sn[i] * H[i, j] + cs[i] * H[i+1, j]
                H[i, j]    = τ
            end
            cs[j], sn[j] = _givens(H[j, j], H[j+1, j])
            H[j, j]   = cs[j] * H[j, j] + sn[j] * H[j+1, j]
            H[j+1, j] = 0
            g[j+1] = -sn[j] * g[j]
            g[j]   =  cs[j] * g[j]
            res = abs(g[j+1])
            res ≤ tol * bnorm && break
        end
        y = zeros(T, k)                    # back-substitution: H[1:k,1:k] y = g[1:k]
        for i in k:-1:1
            acc = g[i]
            for l in i+1:k
                acc -= H[i, l] * y[l]
            end
            y[i] = acc / H[i, i]
        end
        for i in 1:k
            x .+= y[i] .* V[i]
        end
        res ≤ tol * bnorm && break
    end
    return x, total, res
end

@inline function _givens(a, b)
    b == 0 && return (one(a), zero(a))
    if abs(b) > abs(a)
        τ = a / b; s = inv(sqrt(1 + τ^2)); return (s * τ, s)
    else
        τ = b / a; c = inv(sqrt(1 + τ^2)); return (c, c * τ)
    end
end
