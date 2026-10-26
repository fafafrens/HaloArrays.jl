using Test
using HaloArrays
using HaloArrays.Krylov
using LinearAlgebra

struct ScaleOp{T}
    d::T
end
LinearAlgebra.mul!(y, D::ScaleOp, x) = (y .= D.d .* x; y)

# The kernels need no LinearSolve: plain `mul!`/`dot`/`norm`/`ldiv!` and
# broadcasting. Checked on a Vector with a Matrix, and on a 2-D halo array with
# a Diagonal-of-halo-array operator (mul!/ldiv! from vector_space.jl).
@testset "HaloArrays.Krylov kernels without LinearSolve" begin
    n = 30
    Q = Matrix(qr(randn(n, n)).Q)
    Aspd = Q * Diagonal(range(1.0, 50.0; length = n)) * transpose(Q); Aspd = (Aspd + Aspd') / 2
    S = randn(n, n); Asym = Symmetric(S + S')                       # indefinite
    Agen = Diagonal(range(3.0, 6.0; length = n)) + 0.2 * randn(n, n)
    b = randn(n)
    tol = (; abstol = 0.0, reltol = 1e-10)

    for (name, f, ws, A) in (
            ("cg!",       cg!,       CGWorkspace(b),                 Aspd),
            ("minres!",   minres!,   MINRESWorkspace(b),             Asym),
            ("bicgstab!", bicgstab!, BiCGStabWorkspace(b),           Agen),
            ("gmres!",    gmres!,    GMRESWorkspace(b; restart = n), Agen),
            ("fgmres!",   fgmres!,   FGMRESWorkspace(b; restart = n), Agen))
        x = zeros(n)
        r = f(x, A, b, ws; tol..., maxiter = 10n)
        @test r isa Result
        @test r.converged
        @test norm(b - A * x) / norm(b) < 1e-9
        @test x ≈ A \ b
        # the workspace is reusable
        fill!(x, 0)
        @test f(x, A, b, ws; tol..., maxiter = 10n).iters == r.iters
        # maxiter is honoured and reported
        fill!(x, 0)
        r1 = f(x, A, b, ws; tol..., maxiter = 1)
        @test !r1.converged && r1.iters == 1
    end

    # Preconditioners: Jacobi on either side where accepted; `nothing` is none.
    M = Diagonal(diag(Aspd))
    x = zeros(n); @test cg!(x, Aspd, b, CGWorkspace(b); Pl = M, tol..., maxiter = 10n).converged && x ≈ Aspd \ b
    Mg = Diagonal(diag(Agen))
    for (f, ws) in ((bicgstab!, BiCGStabWorkspace(b)), (gmres!, GMRESWorkspace(b; restart = n)), (fgmres!, FGMRESWorkspace(b; restart = n)))
        for kw in ((; Pl = Mg), (; Pr = Mg), (; Pl = Mg, Pr = Mg))
            x = zeros(n)
            @test f(x, Agen, b, ws; kw..., tol..., maxiter = 10n).converged
            @test x ≈ Agen \ b
        end
    end
    # Breakdowns are reported, not spun on: CG on an indefinite operator,
    # BiCGStab with a shadow residual orthogonal to A r.
    Aind = Diagonal([1.0, -1.0]); bi = [1.0, 1.0]
    ri = cg!(zeros(2), Aind, bi, CGWorkspace(bi); abstol = 0.0, reltol = 1e-10, maxiter = 50)
    @test !ri.converged && ri.iters == 0 && isfinite(ri.residual)
    # A negative-definite pair (A, Pl) is the same iteration as (-A, -Pl): converges.
    xn = zeros(n)
    rn = cg!(xn, -Aspd, -b, CGWorkspace(b); Pl = -Diagonal(diag(Aspd)), abstol = 0.0, reltol = 1e-10, maxiter = 10n)
    @test rn.converged && xn ≈ Aspd \ b
    Askew = [0.0 1.0; -1.0 0.0]; bs = [1.0, 0.0]            # ⟨r̂, A r̂⟩ = 0 at the first step
    rs = bicgstab!(zeros(2), Askew, bs, BiCGStabWorkspace(bs); abstol = 0.0, reltol = 1e-10, maxiter = 50)
    @test !rs.converged && rs.iters == 0 && isfinite(rs.residual)

    # fgmres! without Pr leaves Z untouched (the correction reads V).
    wsz = FGMRESWorkspace(b; restart = n); foreach(z -> fill!(z, 7.0), wsz.Z)
    xz = zeros(n); @test fgmres!(xz, Agen, b, wsz; tol..., maxiter = 10n).converged
    @test all(z -> all(==(7.0), z), wsz.Z)
    @test xz ≈ Agen \ b

    @test_throws ArgumentError GMRESWorkspace(b; restart = 0)
    @test_throws ArgumentError FGMRESWorkspace(b; restart = 0)

    # Orthogonalisation variants: same solution and count as :mgs on a
    # well-conditioned system; an unknown choice throws; :auto is :mgs here.
    @test Krylov.resolve_orth(:auto, b) === :mgs
    @test_throws ArgumentError gmres!(zeros(n), Agen, b, GMRESWorkspace(b; restart = n); orth = :cgs, tol..., maxiter = n)
    for (f, mk) in ((gmres!, GMRESWorkspace), (fgmres!, FGMRESWorkspace))
        xm = zeros(n); rm = f(xm, Agen, b, mk(b; restart = n); orth = :mgs,  tol..., maxiter = 10n)
        xc = zeros(n); rc = f(xc, Agen, b, mk(b; restart = n); orth = :cgs2, tol..., maxiter = 10n)
        @test rc.converged && rc.iters == rm.iters
        @test isapprox(xc, xm; rtol = 1e-8)
    end
    # Orthogonality of the Arnoldi basis after one full cycle on an ill-conditioned
    # system (κ = 1e8). MGS loses it in proportion to the conditioning (here
    # ~1e-8); CGS2 keeps it at working precision — what the second pass buys.
    Qi = Matrix(qr(randn(n, n)).Q)
    Aill = Qi * Diagonal(10.0 .^ range(0, 8; length = n)) * transpose(Qi)
    loss = Dict{Symbol,Float64}()
    for orth in (:mgs, :cgs2)
        ws = GMRESWorkspace(b; restart = n)
        r = gmres!(zeros(n), Aill, b, ws; orth, abstol = 0.0, reltol = 1e-14, maxiter = n)
        Vk = reduce(hcat, ws.V[1:r.iters])
        loss[orth] = norm(Vk' * Vk - I)
    end
    @test loss[:cgs2] < 1e-12
    @test loss[:cgs2] < loss[:mgs]

    # 2-D halo array: D x = b with D a pointwise scaling by a halo array (any
    # operator with `mul!` will do).
    h = LocalHaloArray(Float64, (6, 5), 1; boundary_condition = :periodic)
    fill_from_global_indices!(I -> 1.0 + 0.3 * sin(I[1] + 2I[2]), h)
    d = 2.0 .+ 0.5 .* h
    D = ScaleOp(d)
    for (f, ws) in ((cg!, CGWorkspace(h)), (minres!, MINRESWorkspace(h)), (bicgstab!, BiCGStabWorkspace(h)),
                    (gmres!, GMRESWorkspace(h; restart = 10)), (fgmres!, FGMRESWorkspace(h; restart = 10)))
        x = zero(h)
        r = f(x, D, h, ws; tol..., maxiter = 200)
        @test r.converged
        @test x ≈ h ./ d
    end
end
