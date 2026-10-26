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
    @test_throws ArgumentError GMRESWorkspace(b; restart = 0)
    @test_throws ArgumentError FGMRESWorkspace(b; restart = 0)

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
