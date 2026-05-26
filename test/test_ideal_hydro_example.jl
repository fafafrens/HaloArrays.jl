module LocalIdealHydroExampleTest

using Test
using HaloArrays

include(joinpath(@__DIR__, "..", "examples", "ideal_hydro_local_2d.jl"))

function run_tests()
    @testset "Local ideal hydro 2D example" begin
        u, info, initial, final =
            run_local_ideal_hydro_2d(; nx=16, ny=12, steps=4, gamma=1.4, cfl=0.25)

        @test u isa MultiHaloArray
        @test u[:rho] isa LocalHaloArray
        @test global_size(u[:rho]) == (16, 12)
        @test info.time > 0
        @test info.dt > 0
        @test info.adaptive
        @test final.min_rho > 0
        @test final.min_pressure > 0
        @test isfinite(final.max_speed)
        @test final.mass ≈ initial.mass
        @test final.energy ≈ initial.energy
    end
end

end

module ThreadedIdealHydroExampleTest

using Test
using HaloArrays

include(joinpath(@__DIR__, "..", "examples", "ideal_hydro_threaded_2d.jl"))

function run_tests()
    @testset "Threaded ideal hydro 2D example" begin
        u, info, initial, final =
            run_threaded_ideal_hydro_2d(; nx=16, ny=12, steps=4, gamma=1.4, cfl=0.25)

        @test u isa MultiHaloArray
        @test u[:rho] isa ThreadedHaloArray
        @test global_size(u[:rho]) == (16, 12)
        @test tile_count(u[:rho]) == Base.Threads.nthreads()
        @test info.time > 0
        @test info.dt > 0
        @test info.adaptive
        @test final.min_rho > 0
        @test final.min_pressure > 0
        @test isfinite(final.max_speed)
        @test final.mass ≈ initial.mass
        @test final.energy ≈ initial.energy
    end
end

end

LocalIdealHydroExampleTest.run_tests()
ThreadedIdealHydroExampleTest.run_tests()
