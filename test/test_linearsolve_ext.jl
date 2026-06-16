using Test
using HaloArrays
using LinearSolve
using Krylov
using OrdinaryDiffEq
using LinearAlgebra

# Exercises HaloArraysLinearSolveExt: the `init_cacheval` override that lets
# LinearSolve's KrylovJL build its workspace via `KrylovConstructor` on a halo
# array (so the stock `KrylovJL_*` algorithms — and `HaloKrylov(:method)` — solve
# matrix-free with the HaloArray itself as the ODE state). Stiff Fisher–KPP
# reaction–diffusion; every supported method must match a high-accuracy explicit
# reference.

@testset "HaloKrylov / KrylovJL LinearSolve extension" begin
    @test !isnothing(Base.get_extension(HaloArrays, :HaloArraysLinearSolveExt))

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

    # Stock KrylovJL_* work directly on a halo array (via the init_cacheval override).
    @testset "$(nameof(typeof(alg)))-$i" for (i, alg) in enumerate((
            KrylovJL_GMRES(), KrylovJL_CG(), KrylovJL_BICGSTAB(), KrylovJL_MINRES()))
        sol = solves(alg)
        @test sol.retcode == ReturnCode.Success
        @test maximum(abs, sol.u[end] - ref.u[end]) < 1e-5
    end

    # HaloKrylov(:method) — symbol alias for KrylovJL, incl. methods without a
    # named KrylovJL_* wrapper.
    @testset "HaloKrylov(:$m)" for m in (:gmres, :cg, :bicgstab, :minres, :dqgmres,
                                         :diom, :fom, :cgs, :minares, :minres_qlp, :symmlq)
        @test HaloKrylov(m) isa KrylovJL
        sol = solves(HaloKrylov(m))
        @test sol.retcode == ReturnCode.Success
        @test maximum(abs, sol.u[end] - ref.u[end]) < 1e-5
    end

    # Forwarded kwargs reach KrylovJL / the Krylov solver.
    @test solves(HaloKrylov(:gmres; atol = 1e-10, rtol = 1e-8)).retcode == ReturnCode.Success

    # Unsupported methods fail fast with a helpful error (not deep in the solve).
    @test_throws ArgumentError HaloKrylov(:car)
    @test_throws ArgumentError HaloKrylov(:fgmres)
    @test_throws ArgumentError HaloKrylov(:cg_lanczos)
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
            ("HaloCG", HaloCG()), ("HaloBiCGStab", HaloBiCGStab()), ("HaloGMRES", HaloGMRES(restart = 50)))
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
