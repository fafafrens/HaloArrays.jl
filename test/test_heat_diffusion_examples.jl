using Test
using HaloArrays

include(joinpath(@__DIR__, "..", "examples", "heat", "common.jl"))

@testset "Heat diffusion local examples" begin
    for dims in ((16,), (8, 7), (5, 4, 3))
        u = LocalHaloArray(Float64, dims, 1; boundary_condition=:periodic)
        dx = ntuple(i -> 1 / dims[i], Val(length(dims)))
        dt = stable_heat_dt(1.0, 0.2, dx)

        fill_centered_gaussian!(u; baseline=1.0, amplitude=1.0)
        initial_sum = sum(interior_view(u))
        initial_max = maximum(interior_view(u))

        solve_heat!(u; alpha=1.0, dt, dx, nt=4)

        @test all(isfinite, interior_view(u))
        @test sum(interior_view(u)) ≈ initial_sum
        @test maximum(interior_view(u)) <= initial_max
    end
end
