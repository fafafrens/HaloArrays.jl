using Test
using HaloArrays
using LinearSolve
using KrylovKit
using OrdinaryDiffEq
using LinearAlgebra
using LinearSolve.SciMLOperators: FunctionOperator

# HaloArraysKrylovKitExt: KrylovKit as the `linsolve` of an implicit integrator,
# the halo array itself as the ODE state (KrylovKit is VectorInterface-generic,
# so no workspace hook is needed). Stiff Fisher–KPP reaction–diffusion; every
# method must match a high-accuracy explicit reference.
@testset "HaloKrylovKit LinearSolve extension" begin
    @test !isnothing(Base.get_extension(HaloArrays, :HaloArraysLinearSolveExt))
    @test !isnothing(Base.get_extension(HaloArrays, :HaloArraysKrylovKitExt))

    Dc, Rc, E1 = 1.0, 8.0, CartesianIndex(1)
    ic(x) = 0.1 + 0.8 * exp(-50 * (x - 0.5)^2)
    function rhs!(du, u, p, t)
        dx2inv = p
        synchronize_halo!(u)
        for tile in 1:tile_count(u)
            s = tile_parent(u, tile); d = tile_parent(du, tile)
            @inbounds for I in CartesianIndices(interior_range(u))
                d[I] = Dc * (s[I - E1] - 2s[I] + s[I + E1]) * dx2inv + Rc * s[I] * (1 - s[I])
            end
        end
        return nothing
    end

    nx = 64
    u0 = LocalHaloArray(Float64, (nx,), 1; boundary_condition = :periodic)
    fill_from_global_indices!(I -> ic((I[1] - 0.5) / nx), u0)
    prob = ODEProblem(rhs!, u0, (0.0, 0.2), inv((1.0 / nx)^2))
    ref = solve(prob, Tsit5(); reltol = 1e-10, abstol = 1e-10, save_everystep = false)
    solves(alg) = solve(prob, FBDF(linsolve = alg, concrete_jac = false);
                        reltol = 1e-7, abstol = 1e-7, save_everystep = false)

    @testset "HaloKrylovKit(:$m)" for m in (:gmres, :cg, :bicgstab)
        sol = solves(HaloKrylovKit(m))
        @test sol.retcode == ReturnCode.Success
        @test sol.u[end] isa LocalHaloArray{Float64,1}
        @test maximum(abs, sol.u[end] - ref.u[end]) < 1e-5
    end
    @test solves(HaloKrylovKit()).retcode == ReturnCode.Success           # default :gmres
    # Tiled state: KrylovKit's VectorInterface fallbacks `map`/`eachindex` a
    # ThreadedHaloArray (no tile-less interior → MethodError); the extension's
    # VectorInterface methods route through the interior broadcasts instead.
    ntiles = Threads.nthreads()        # per-tile size × tile count = the same 64-cell grid
    @test nx % ntiles == 0
    t0 = ThreadedHaloArray(Float64, (nx ÷ ntiles,), 1; dims = (ntiles,), boundary_condition = :periodic)
    fill_from_global_indices!(I -> ic((I[1] - 0.5) / nx), t0)
    tsol = solve(ODEProblem(rhs!, t0, (0.0, 0.2), inv((1.0 / nx)^2)), FBDF(linsolve = HaloKrylovKit(:gmres), concrete_jac = false);
                 reltol = 1e-7, abstol = 1e-7, save_everystep = false)
    @test tsol.retcode == ReturnCode.Success
    @test tsol.u[end] isa ThreadedHaloArray
    @test maximum(abs, Array(tsol.u[end]) .- Array(ref.u[end])) < 1e-5
    # Forwarded kwargs reach the KrylovKit algorithm.
    @test solves(HaloKrylovKit(:gmres; krylovdim = 10)).retcode == ReturnCode.Success
    @test_throws ArgumentError HaloKrylovKit(:minres)
    @test_throws ArgumentError HaloKrylovKit(:fgmres)

    # LinearProblem on a dense system, with a nonzero initial guess; a
    # preconditioner is refused rather than silently ignored.
    n = 40; S = randn(n, n); A = S * S' + n * I; b = randn(n)
    sol = solve(LinearProblem(A, b; u0 = randn(n)), HaloKrylovKit(:cg); reltol = 1e-10, abstol = 0.0)
    @test sol.retcode == ReturnCode.Success && sol.u ≈ A \ b
    @test_throws ArgumentError solve(LinearProblem(A, b; u0 = zeros(n)), HaloKrylovKit(:gmres); Pl = Diagonal(diag(A)))
end

# Matrix-free -∇² (SPD, homogeneous Dirichlet) on a 2-D halo array, as a
# SciMLOperators.FunctionOperator (re-exported by LinearSolve — no extra dep).
function _neg_laplacian!(y, x, _u, p, _t)
    synchronize_halo!(x)
    xd = parent(x); yd = parent(y)
    ex = CartesianIndex(1, 0); ey = CartesianIndex(0, 1)
    @inbounds for I in CartesianIndices(interior_range(x))
        lap = (xd[I + ex] - 2xd[I] + xd[I - ex]) + (xd[I + ey] - 2xd[I] + xd[I - ey])
        yd[I] = -lap * p.inv_h2
    end
    return y
end

@testset "Coordinate-free LinearSolve solvers on an N-D halo array" begin
    # 2-D Dirichlet Poisson with a manufactured solution u = x(1-x)y(1-y).
    n = 48; h = 1.0 / n; ctr(i) = (i - 0.5) * h
    dir = ((Antireflecting(), Antireflecting()), (Antireflecting(), Antireflecting()))
    uex = LocalHaloArray(Float64, (n, n), 1; boundary_condition = dir)
    fill_from_global_indices!(uex) do I
        cx, cy = ctr(I[1]), ctr(I[2]); cx * (1 - cx) * cy * (1 - cy)
    end

    @testset "$name" for (name, alg) in (
            ("HaloCG", HaloCG()), ("HaloMINRES", HaloMINRES()),
            ("HaloBiCGStab", HaloBiCGStab()), ("HaloGMRES", HaloGMRES(restart = 50)))
        rhs = LocalHaloArray(Float64, (n, n), 1; boundary_condition = dir)
        fill_from_global_indices!(rhs) do I
            cx, cy = ctr(I[1]), ctr(I[2]); 2 * (cx * (1 - cx) + cy * (1 - cy))
        end
        L = FunctionOperator(_neg_laplacian!, similar(rhs), similar(rhs);
            islinear = true, isconstant = true, issymmetric = true, isposdef = true,
            p = (inv_h2 = 1.0 / h^2,))
        sol = solve(LinearProblem(L, rhs; u0 = zero(rhs)), alg; reltol = 1e-10)
        @test sol.retcode == ReturnCode.Success
        # error is the O(h²) discretisation error, not the solver tolerance
        @test maximum(abs, interior_view(sol.u) .- interior_view(uex)) < 1e-4
    end
end

@testset "HaloGMRES robustness" begin
    @test_throws ArgumentError HaloGMRES(restart = 0)
    @test_throws ArgumentError HaloGMRES(restart = -3)

    # Reach the internal solver to check the numerical edge cases directly.
    ext = Base.get_extension(HaloArrays, :HaloArraysLinearSolveExt)
    run_gmres(x, A, b; restart = length(b), maxiter = 10 * length(b)) =
        ext._krylov!(x, A, b, ext.GMRESWorkspace(b; restart = restart), I, I;
                    abstol = 0.0, reltol = 1e-10, maxiter = maxiter)

    # Complex system: the Hessenberg/Givens stay in the complex field — A = i,
    # b = 1 must give x = -i (a real-truncating GMRES would record H[1,1] = 0).
    x1 = ComplexF64[0]; run_gmres(x1, fill(im, 1, 1), ComplexF64[1]; restart = 1)
    @test x1[1] ≈ -im

    # Happy breakdown: an exact 3-dimensional Krylov space converges (and stops)
    # without using an un-formed basis vector.
    A = Diagonal([1.0, 2, 3, 4, 5]); b = [1.0, 1, 1, 0, 0]
    x = zeros(5); _, iters, _, conv = run_gmres(x, A, b; restart = 5, maxiter = 20)
    @test conv && iters == 3 && x ≈ A \ b

    # Iteration budget: maxiter = 1 must perform at most one Arnoldi step
    # (not a full restart cycle of `restart` matvecs).
    xb = zeros(8); _, itb, _, _ = run_gmres(xb, randn(8, 8) + 8I, randn(8); restart = 30, maxiter = 1)
    @test itb ≤ 1
end

@testset "left preconditioning (Pl) is applied, not ignored" begin
    n = 12
    Q = Matrix(qr(randn(n, n)).Q)
    Aspd = Q * Diagonal(range(1.0, 40.0; length = n)) * transpose(Q)
    Aspd = (Aspd + transpose(Aspd)) / 2                      # SPD, ill-conditioned
    # well-conditioned, diagonally dominant non-symmetric matrix for GMRES/BiCGStab
    Agen = Diagonal(range(3.0, 6.0; length = n)) + 0.2 * randn(n, n)
    b = randn(n)

    @testset "$name converges with a Jacobi preconditioner" for (name, alg, A) in (
            ("HaloCG", HaloCG(), Aspd),
            ("HaloGMRES", HaloGMRES(restart = n), Agen),
            ("HaloBiCGStab", HaloBiCGStab(), Agen))
        xref = A \ b
        M = Diagonal(diag(A))
        sol = solve(LinearProblem(copy(A), copy(b)), alg; Pl = M, reltol = 1e-10, maxiters = 500)
        @test sol.retcode == ReturnCode.Success
        @test isapprox(sol.u, xref; rtol = 1e-6)

        # A perfect preconditioner (M = A on a diagonal system, so M⁻¹A = I) must
        # converge in a single iteration — proof the preconditioner is applied.
        d = collect(range(2.0, 5.0; length = n)); Ad = Diagonal(d); bd = randn(n)
        exact = solve(LinearProblem(Ad, copy(bd)), alg; Pl = Diagonal(d), reltol = 1e-10, maxiters = 50)
        @test exact.iters == 1
        @test exact.u ≈ Ad \ bd
    end

    # The stopping test is relative to b in the PRECONDITIONED norm: a strongly
    # scaled Pl (c·I, which leaves the iterates unchanged) must neither report
    # convergence early (c = 1e6: true residual far above reltol) nor over-solve
    # (c = 1e-6: many extra iterations). Both must match the unpreconditioned solve.
    @testset "preconditioned stopping test is scale-invariant: $name" for (name, alg) in (
            ("HaloCG", HaloCG()), ("HaloBiCGStab", HaloBiCGStab()), ("HaloGMRES", HaloGMRES(restart = 10)))
        relres(sol) = norm(b - Aspd * sol.u) / norm(b)
        base = solve(LinearProblem(Aspd, b; u0 = zeros(n)), alg; reltol = 1e-10, abstol = 0.0, maxiters = 500)
        @test relres(base) < 1e-9
        for c in (1e6, 1e-6), Pl in (Diagonal(fill(c, n)), c * I)
            sol = solve(LinearProblem(Aspd, b; u0 = zeros(n)), alg; Pl, reltol = 1e-10, abstol = 0.0, maxiters = 500)
            @test sol.retcode == ReturnCode.Success
            @test relres(sol) < 1e-9
            @test abs(sol.iters - base.iters) ≤ 1
        end
    end

    # Unsupported preconditioner requests are rejected, not silently dropped.
    M = Diagonal(diag(Aspd))
    @test_throws ArgumentError solve(LinearProblem(Aspd, b), HaloCG(); Pr = M)
    @test_throws ArgumentError solve(LinearProblem(Aspd, b), HaloMINRES(); Pl = M)
    @test_throws ArgumentError solve(LinearProblem(Aspd, b), HaloMINRES(); Pr = M)
end

@testset "right preconditioning (Pr) on HaloGMRES / HaloBiCGStab" begin
    n = 12
    # Badly scaled rows/columns: Jacobi on either side takes most of the iterations out.
    d = collect(10.0 .^ range(0, 4; length = n))
    Agen = Diagonal(d) + 0.1 * randn(n, n)
    b = randn(n)
    xref = Agen \ b
    relres(sol) = norm(b - Agen * sol.u) / norm(b)
    M = Diagonal(diag(Agen))

    @testset "$name" for (name, alg) in (("HaloGMRES", HaloGMRES(restart = n)), ("HaloBiCGStab", HaloBiCGStab()))
        prob() = LinearProblem(copy(Agen), copy(b); u0 = zeros(n))
        base = solve(prob(), alg; reltol = 1e-10, abstol = 0.0, maxiters = 500)
        right = solve(prob(), alg; Pr = M, reltol = 1e-10, abstol = 0.0, maxiters = 500)
        @test right.retcode == ReturnCode.Success
        @test relres(right) < 1e-9
        @test isapprox(right.u, xref; rtol = 1e-6)
        @test right.iters < base.iters                   # the preconditioner is applied

        # Both sides at once (split Jacobi): still converges to the same solution.
        both = solve(prob(), alg; Pl = Diagonal(sqrt.(d)), Pr = Diagonal(sqrt.(d)),
                     reltol = 1e-10, abstol = 0.0, maxiters = 500)
        @test both.retcode == ReturnCode.Success
        @test isapprox(both.u, xref; rtol = 1e-6)

        # A perfect right preconditioner on a diagonal system (A Pr⁻¹ = I) converges
        # in a single iteration — proof the direction is mapped back through Pr⁻¹.
        Ad = Diagonal(d); bd = randn(n)
        exact = solve(LinearProblem(Ad, copy(bd); u0 = zeros(n)), alg; Pr = Diagonal(d), reltol = 1e-10, maxiters = 50)
        @test exact.iters == 1
        @test exact.u ≈ Ad \ bd

        # Pr = c·I leaves the iterates unchanged (y = c x spans the same space),
        # and the monitored residual is the TRUE residual, so the stopping test
        # must be scale-invariant without any rescaling. (Slack for rounding: on
        # this ill-conditioned system BiCGStab's count moves by an iteration or two.)
        for c in (1e6, 1e-6), Pr in (Diagonal(fill(c, n)), c * I)
            sol = solve(prob(), alg; Pr, reltol = 1e-10, abstol = 0.0, maxiters = 500)
            @test sol.retcode == ReturnCode.Success
            @test relres(sol) < 1e-9
            @test abs(sol.iters - base.iters) ≤ max(2, base.iters ÷ 10)
        end
    end

    # On an N-D halo array, with a halo-array-valued diagonal preconditioner.
    h = LocalHaloArray(Float64, (6, 5), 1; boundary_condition = :periodic)
    fill_from_global_indices!(I -> 1.0 + 0.3 * sin(I[1] + 2I[2]), h)
    dh = 2.0 .+ 0.5 .* h                              # positive diagonal, a halo array
    Aop = FunctionOperator((w, v, u, p, t) -> (w .= dh .* v; w), h, h; isinplace = true, islinear = true)
    Mop = FunctionOperator((w, v, u, p, t) -> (w .= v ./ dh; w), h, h; isinplace = true, islinear = true)
    # LinearSolve applies Pr through `ldiv!`; FunctionOperator exposes the inverse
    # through `\`, so wrap the inverse as the operator: Pr⁻¹ v = v ./ dh.
    for alg in (HaloGMRES(restart = 10), HaloBiCGStab())
        sol = solve(LinearProblem(Aop, h; u0 = zero(h)), alg; Pr = LinearSolve.InvPreconditioner(Mop),
                    reltol = 1e-10, abstol = 0.0, maxiters = 100)
        @test sol.retcode == ReturnCode.Success
        @test sol.u ≈ h ./ dh
    end
end

# A 2-D / 3-D halo array as the state of an implicit OrdinaryDiffEq solve.
# OrdinaryDiffEq `vec`s the state around every `mul!` with the matrix-free `W`
# operator (and `zero(vec(u))`s it to build `W`); `vec` is the identity on a halo
# array, so the N-D state reaches the Jacobian-vector product and the coordinate-
# free solvers untouched. Heat equation, Dirichlet, vs. a tight explicit reference.
@testset "Implicit OrdinaryDiffEq on an N-D halo array state" begin
    for dims in ((4,), (4, 3), (4, 3, 2))
        h = LocalHaloArray(Float64, dims, 1; boundary_condition = :periodic)
        @test vec(h) === h
        m = MultiHaloArray(LocalHaloArray, Float64, dims, 1; fields = (:a, :b), boundary_condition = :periodic)
        @test vec(m) === m
    end
    function heat!(du, u, p, t)
        synchronize_halo!(u)
        s = parent(u); d = parent(du)
        @inbounds for I in CartesianIndices(interior_range(u))
            acc = zero(eltype(s))
            for v in HaloArrays.versors(u)
                e = CartesianIndex(v)
                acc += s[I - e] - 2s[I] + s[I + e]
            end
            d[I] = acc * p
        end
        return nothing
    end
    @testset "$(length(dims))-D, $name" for dims in ((24, 20), (10, 9, 8)),
                                            (name, alg) in (("HaloCG", HaloCG()), ("HaloGMRES", HaloGMRES()),
                                                            ("HaloKrylovKit(:cg)", HaloKrylovKit(:cg)))
        n = dims[1]
        u0 = LocalHaloArray(Float64, dims, 1; boundary_condition = :antireflecting)
        fill_from_global_indices!(I -> exp(-sum(d -> (I[d] - dims[d] / 2)^2, 1:length(dims)) / 6), u0)
        prob = ODEProblem(heat!, u0, (0.0, 0.05), Float64(n^2))
        ref = solve(prob, Tsit5(); reltol = 1e-10, abstol = 1e-10, save_everystep = false)
        sol = solve(prob, FBDF(linsolve = alg, concrete_jac = false);
                    reltol = 1e-7, abstol = 1e-7, save_everystep = false)
        @test sol.retcode == ReturnCode.Success
        @test sol.u[end] isa LocalHaloArray{Float64,length(dims)}
        @test maximum(abs, sol.u[end] - ref.u[end]) < 1e-6
    end
end

# A field collection as the implicit state. Its `size` is `(fields..., spatial...)`,
# so even a 1-D-in-space collection is a 2-D array to the integrator and would be
# `vec`-wrapped; the identity keeps the named fields reachable inside `rhs!`.
# Two-species reaction–diffusion (nonsymmetric Jacobian → GMRES / BiCGStab).
# A user-supplied Jacobian-vector product. OrdinaryDiffEq's matrix-free path
# ignores `ODEFunction(; jvp)`; the route is `jac_prototype` as a SciMLOperators
# `FunctionOperator` with `jvp!(Jv, v, u, p, t)` in place, and the halo array
# flows through it unflattened. Linear 2-D heat: J is the Laplacian, so the
# solution must match the autodiff path, while `rhs!` is called far less often.
@testset "User Jacobian-vector product via jac_prototype on a 2-D halo array" begin
    dims = (20, 16); p = Float64(dims[1]^2)
    calls = Ref(0)
    function heat!(du, u, p, t)
        calls[] += 1
        synchronize_halo!(u)
        s = parent(u); d = parent(du)
        @inbounds for I in CartesianIndices(interior_range(u))
            acc = zero(eltype(s))
            for v in HaloArrays.versors(u)
                e = CartesianIndex(v); acc += s[I - e] - 2s[I] + s[I + e]
            end
            d[I] = acc * p
        end
        return nothing
    end
    jvp!(Jv, v, u, p, t) = (heat!(Jv, v, p, t); Jv)   # linear: J v = L v
    u0 = LocalHaloArray(Float64, dims, 1; boundary_condition = :antireflecting)
    fill_from_global_indices!(I -> exp(-sum(d -> (I[d] - dims[d] / 2)^2, 1:2) / 6), u0)
    J = FunctionOperator(jvp!, similar(u0), similar(u0); u = u0, p = p, t = 0.0, isinplace = true, islinear = true)
    tspan = (0.0, 0.05)
    solve_with(f) = (calls[] = 0; sol = solve(ODEProblem(f, u0, tspan, p), FBDF(linsolve = HaloCG(), concrete_jac = false);
                                               reltol = 1e-7, abstol = 1e-7, save_everystep = false); (sol, calls[]))
    ref = solve(ODEProblem(heat!, u0, tspan, p), Tsit5(); reltol = 1e-10, abstol = 1e-10, save_everystep = false)
    sol_ad, n_ad = solve_with(ODEFunction{true}(heat!))
    sol_op, n_op = solve_with(ODEFunction{true}(heat!; jac_prototype = J))
    @test sol_op.retcode == ReturnCode.Success
    @test sol_op.u[end] isa LocalHaloArray{Float64,2}
    @test maximum(abs, sol_op.u[end] - ref.u[end]) < 1e-6
    @test sol_op.u[end] ≈ sol_ad.u[end] rtol = 1e-6
    @test n_op < n_ad ÷ 2      # the operator replaced the autodiff JVP's rhs! evaluations
end

@testset "Implicit OrdinaryDiffEq on a MultiHaloArray state" begin
    function rd!(du, u, p, _t)
        DA, DB, k, inv_dx2 = p
        synchronize_halo!(u)
        dA = parent(du.A); dB = parent(du.B); A = parent(u.A); B = parent(u.B)
        @inbounds for I in CartesianIndices(interior_range(u.A))
            lapA = zero(eltype(A)); lapB = zero(eltype(B))
            for v in HaloArrays.versors(u.A)
                e = CartesianIndex(v)
                lapA += A[I + e] - 2A[I] + A[I - e]
                lapB += B[I + e] - 2B[I] + B[I - e]
            end
            r = k * A[I] * B[I]
            dA[I] = DA * lapA * inv_dx2 - r
            dB[I] = DB * lapB * inv_dx2 - r
        end
        return du
    end
    @testset "$(length(n))-D, $name" for n in ((32,), (16, 12)),
                                         (name, alg) in (("HaloGMRES", HaloGMRES()), ("HaloBiCGStab", HaloBiCGStab()),
                                                         ("HaloKrylovKit(:gmres)", HaloKrylovKit(:gmres)))
        bc = ntuple(_ -> (Periodic(), Periodic()), length(n))
        u0 = MultiHaloArray(LocalHaloArray, Float64, n, 1; boundary_conditions = (A = bc, B = bc))
        fill_from_global_indices!(I -> 0.5 + 0.1sin(2π * I[1] / n[1]), u0.A)
        fill_from_global_indices!(I -> 0.5 + 0.1cos(2π * I[end] / n[end]), u0.B)
        prob = ODEProblem{true}(rd!, u0, (0.0, 0.1), (1.0, 0.5, 50.0, Float64(n[1]^2)))
        ref = solve(prob, Tsit5(); reltol = 1e-10, abstol = 1e-10, save_everystep = false)
        sol = solve(prob, FBDF(linsolve = alg, concrete_jac = false);
                    reltol = 1e-7, abstol = 1e-7, save_everystep = false)
        @test sol.retcode == ReturnCode.Success
        @test sol.u[end] isa MultiHaloArray
        @test maximum(abs, sol.u[end] - ref.u[end]) < 1e-6
    end
end

@testset "HaloMINRES on a symmetric indefinite system" begin
    ext = Base.get_extension(HaloArrays, :HaloArraysLinearSolveExt)
    n = 40; S = randn(n, n); A = Symmetric(S + S')   # eigenvalues straddle 0 → CG breaks down
    b = randn(n); x = zeros(n)
    _, _, _, conv = ext._krylov!(x, A, b, ext.MINRESWorkspace(b), I, I;
                                 abstol = 0.0, reltol = 1e-10, maxiter = 5n)
    @test conv && x ≈ A \ b
end

@testset "LinearSolve algorithm interface contract" begin
    # LinearSolve's own checker: a release that adds a required hook fails here
    # instead of warning at solve time.
    # (The checker arrived in LinearSolve 5.17; compat still allows older 5.x.)
    for alg in (HaloCG(), HaloGMRES(), HaloBiCGStab(), HaloMINRES(), HaloKrylovKit())
        if isdefined(LinearSolve, :algorithm_interface_issues)
            @test isempty(LinearSolve.algorithm_interface_issues(alg))
        else
            @test_skip isempty(LinearSolve.algorithm_interface_issues(alg))
        end
    end
end
