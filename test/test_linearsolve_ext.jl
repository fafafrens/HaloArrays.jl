using Test
using HaloArrays
using LinearSolve
using Krylov
using OrdinaryDiffEq
using LinearAlgebra

# Exercises the HaloArraysLinearSolveExt: `HaloKrylov(:method)` as the linear
# solver of an implicit (matrix-free) SciML integrator, with the HaloArray itself
# as the ODE state. A stiff Fisher–KPP reaction–diffusion problem; every working
# method must reach the same solution as a high-accuracy explicit reference.

@testset "HaloKrylov LinearSolve extension" begin
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

    # The matrix-free, up-front-allocating methods that work on a halo array.
    @testset "$m" for m in (:gmres, :bicgstab, :cg, :minres, :cgs, :dqgmres, :car, :cg_lanczos)
        sol = solve(prob, FBDF(linsolve = HaloKrylov(m), concrete_jac = false);
                    reltol = 1e-7, abstol = 1e-7, save_everystep = false)
        @test sol.retcode == ReturnCode.Success
        @test maximum(abs, sol.u[end] - ref.u[end]) < 1e-5
    end

    # Forwarded solver kwargs are accepted.
    sol = solve(prob, FBDF(linsolve = HaloKrylov(:gmres; atol = 1e-10, rtol = 1e-8),
                           concrete_jac = false);
                reltol = 1e-7, abstol = 1e-7, save_everystep = false)
    @test sol.retcode == ReturnCode.Success
end
