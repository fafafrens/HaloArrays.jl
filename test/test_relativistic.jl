using Test
using HaloArrays

# ============================================================
# Relativistic hydro example kernels
#
# The examples are self-contained scripts that auto-run a driver at the end.
# Load each into its own module with the trailing `run_*()` calls stripped, so
# the physics kernels (EOS, cons↔prim recovery, fluxes, RHS, stepper) are
# testable and short runs can be driven with test-sized parameters.
#
# Covers all three conserved→primitive recovery schemes:
#   - mass-based (D, S, τ): 1-D Newton on pressure   (relativistic_common.jl)
#   - μ = 0 conformal:      closed form              (relativistic_hydro_mu0_1d.jl)
#   - (T, μ) with charge:   2×2 Newton               (relativistic_hydro_Tmu_1d.jl / _2d.jl)
# ============================================================

function _include_example!(mod::Module, relpath::AbstractString)
    path = joinpath(@__DIR__, "..", "examples", relpath)
    src = read(path, String)
    # strip top-level driver calls like `run_mu0_sod()` or `u = run_mu0_blast_3d()`
    src = replace(src, r"^\s*(?:[A-Za-z_][A-Za-z0-9_!]*\s*=\s*)?run_[A-Za-z0-9_]+\([^)]*\)\s*$"m => "")
    Base.include_string(mod, src, relpath)
    return mod
end

module RelCommon end
module RelMu0 end
module RelTmu1D end
module RelTmu2D end

_include_example!(RelCommon, joinpath("finite_volume", "relativistic_common.jl"))
_include_example!(RelMu0,    joinpath("finite_volume", "relativistic_hydro_mu0_1d.jl"))
_include_example!(RelTmu1D,  joinpath("finite_volume", "relativistic_hydro_Tmu_1d.jl"))
_include_example!(RelTmu2D,  joinpath("finite_volume", "relativistic_hydro_Tmu_2d.jl"))

@testset "Relativistic hydro kernels" begin

    @testset "mass-based (D,S,τ): recovery round-trip" begin
        eos = RelCommon.IdealGas(5.0 / 3.0)
        for ρ in (0.1, 1.0, 10.0), p in (0.05, 1.0, 5.0), v in (-0.9, -0.5, 0.0, 0.3, 0.9)
            U = RelCommon.cons_from_prim(eos, ρ, v, p)
            ρ2, v2, p2 = RelCommon.prim_from_cons(eos, U)
            @test isapprox(ρ2, ρ; rtol=1e-7)
            @test isapprox(v2, v; atol=1e-8)
            @test isapprox(p2, p; rtol=1e-7)
        end
    end

    @testset "mass-based (D,S,τ): flux & wave-speed identities" begin
        eos = RelCommon.IdealGas(5.0 / 3.0)
        # at rest: F = (0, p, 0)
        U0 = RelCommon.cons_from_prim(eos, 1.0, 0.0, 2.5)
        F0 = RelCommon.physical_flux(eos, U0)
        @test isapprox(F0[1], 0.0; atol=1e-12)
        @test isapprox(F0[2], 2.5; rtol=1e-10)
        @test isapprox(F0[3], 0.0; atol=1e-10)
        for v in (-0.8, -0.2, 0.0, 0.4, 0.9)
            U = RelCommon.cons_from_prim(eos, 1.0, v, 1.0)
            s = RelCommon.max_wave_speed(eos, U)
            @test abs(v) - 1e-10 <= s < 1.0          # subluminal, at least |v|
            # consistency: Rusanov of identical states is the physical flux
            @test RelCommon.rusanov_flux(eos, U, U) ≈ RelCommon.physical_flux(eos, U)
        end
    end

    @testset "mass-based (D,S,τ): periodic conservation" begin
        eos = RelCommon.IdealGas(5.0 / 3.0)
        nx = 128
        dx = 1.0 / nx
        u  = LocalMultiHaloArray(Float64, (nx,), 1;
            fields=(:D, :S, :tau), boundary_condition=:periodic)
        for i in 1:nx
            x = (i - 0.5) * dx
            ρ = 1.0 + 0.3 * sinpi(2x)
            p = 1.0 + 0.2 * cospi(2x)
            v = 0.2 * sinpi(2x)
            U = RelCommon.cons_from_prim(eos, ρ, v, p)
            interior_view(u.D)[i]   = U[1]
            interior_view(u.S)[i]   = U[2]
            interior_view(u.tau)[i] = U[3]
        end
        synchronize_halo!(u)
        nobc = _ -> nothing                       # periodic: exchange does it all
        D0 = sum(interior_view(u.D)); S0 = sum(interior_view(u.S)); τ0 = sum(interior_view(u.tau))

        u1 = similar(u); du = similar(u)
        for _ in 1:30
            dt = RelCommon.cfl_dt(u, eos, dx, 0.4)
            RelCommon.ssprk2_step!(u, u1, du, eos, nobc, dt, dx)
        end
        @test all(isfinite, interior_view(u.D))
        @test isapprox(sum(interior_view(u.D)),   D0; rtol=1e-12, atol=1e-10)
        @test isapprox(sum(interior_view(u.S)),   S0; rtol=1e-10, atol=1e-10)
        @test isapprox(sum(interior_view(u.tau)), τ0; rtol=1e-12, atol=1e-10)
        _, _, vmax = RelCommon.diagnostics(u, eos, dx)
        @test 0.0 < vmax < 1.0
    end

    @testset "μ=0 conformal: closed-form recovery round-trip" begin
        eos = RelMu0.ConformalGas(1.0)
        for T in (0.3, 1.0, 2.5), v in (-0.95, -0.5, 0.0, 0.7, 0.95)
            U = RelMu0.cons_from_prim(eos, T, v)
            T2, v2 = RelMu0.prim_from_cons(eos, U)
            @test isapprox(T2, T; rtol=1e-12)     # closed form: exact to roundoff
            @test isapprox(v2, v; atol=1e-12)
        end
        # energy flux is the momentum: F_E = M
        U = RelMu0.cons_from_prim(eos, 1.2, 0.6)
        @test RelMu0.physical_flux(eos, U)[2] == U[1]
        @test RelMu0.rusanov_flux(eos, U, U) ≈ RelMu0.physical_flux(eos, U)
    end

    @testset "(T,μ) with charge: 2×2 Newton recovery round-trip" begin
        eos = RelTmu1D.UltraRelGas(1.0)
        for T in (0.5, 1.0, 2.0), μ in (-1.0, 0.0, 1.0), v in (-0.8, 0.0, 0.5, 0.9)
            U = RelTmu1D.cons_from_prim(eos, T, μ, v)
            T2, μ2, v2 = RelTmu1D.prim_from_cons(eos, U)
            @test isapprox(T2, T; rtol=1e-7)
            @test isapprox(μ2, μ; atol=1e-6)
            @test isapprox(v2, v; atol=1e-8)
        end
        # EOS identity n = p/T and flux consistency
        @test RelTmu1D.charge_density(eos, 1.3, 0.4) ≈ RelTmu1D.pressure(eos, 1.3, 0.4) / 1.3
        U = RelTmu1D.cons_from_prim(eos, 1.0, 0.5, 0.4)
        @test RelTmu1D.rusanov_flux(eos, U, U) ≈ RelTmu1D.physical_flux(eos, U)
    end

    @testset "(T,μ) 2-D: directional fluxes + periodic conservation" begin
        eos = RelTmu2D.UltraRelGas(1.0)
        # recovery sees the momentum only through |M|²
        Ua = RelTmu2D.cons_from_prim(eos, 1.0, 0.2, 0.5, 0.0)
        Ub = RelTmu2D.cons_from_prim(eos, 1.0, 0.2, 0.0, 0.5)
        Ta, μa, _, _ = RelTmu2D.prim_from_cons(eos, Ua)
        Tb, μb, _, _ = RelTmu2D.prim_from_cons(eos, Ub)
        @test isapprox(Ta, Tb; rtol=1e-10)
        @test isapprox(μa, μb; atol=1e-9)
        # wave speeds subluminal in both directions
        @test RelTmu2D.max_wave_speed(eos, Ua, 1) < 1.0
        @test RelTmu2D.max_wave_speed(eos, Ua, 2) < 1.0

        n = 24
        dx = 1.0 / n; dy = dx
        u = LocalMultiHaloArray(Float64, (n, n), 1;
            fields=(:N, :Mx, :My, :E), boundary_condition=:periodic)
        for j in 1:n, i in 1:n
            x = (i - 0.5) * dx; y = (j - 0.5) * dy
            T = 1.0 + 0.2 * sinpi(2x) * cospi(2y)
            U = RelTmu2D.cons_from_prim(eos, T, 0.1 * sinpi(2y), 0.0, 0.0)
            interior_view(u.N)[i, j]  = U[1]
            interior_view(u.Mx)[i, j] = U[2]
            interior_view(u.My)[i, j] = U[3]
            interior_view(u.E)[i, j]  = U[4]
        end
        synchronize_halo!(u)
        q0 = sum(interior_view(u.N)); e0 = sum(interior_view(u.E))

        u1 = similar(u); du = similar(u)
        for _ in 1:10
            dt = RelTmu2D.cfl_dt(u, eos, dx, dy, 0.3)
            RelTmu2D.ssprk2_step!(u, u1, du, eos, dt, dx, dy)
        end
        @test all(isfinite, interior_view(u.E))
        @test isapprox(sum(interior_view(u.N)), q0; rtol=1e-12, atol=1e-10)   # charge conserved
        @test isapprox(sum(interior_view(u.E)), e0; rtol=1e-12, atol=1e-10)   # energy conserved
        _, _, vmax = RelTmu2D.diagnostics(u, eos, dx, dy)
        @test 0.0 < vmax < 1.0
    end
end
