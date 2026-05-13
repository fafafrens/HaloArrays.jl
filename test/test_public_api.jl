using Test
using MPI
using HaloArrays

@testset "public API exports" begin
    ha = HaloArray(Float64, (5,), 2; boundary_condition=:repeating)

    @test ha isa HaloArray
    @test interior_size(ha) == (5,)
    @test full_size(ha) == (9,)
    @test halo_width(ha) == 2
    @test size(ha) == (5,)
    @test length(ha) == 5

    fill_interior(ha, 3.0)
    @test all(interior_view(ha) .== 3.0)

    interior_view(ha)[1] = 4.0
    @test interior_view(ha)[1] == 4.0
    @test parent(ha)[3] == 4.0

    boundary_condition!(ha)
    if ha.topology.neighbors[1][1] == MPI.PROC_NULL
        @test parent(ha)[1:2] == [4.0, 4.0]
    end
    if ha.topology.neighbors[1][2] == MPI.PROC_NULL
        @test parent(ha)[8:9] == [3.0, 3.0]
    end

    resized = similar(ha, Float32, (3,))
    @test resized isa HaloArray
    @test eltype(resized) === Float32
    @test size(resized) == (3,)
    @test full_size(resized) == (7,)

    local_ha = LocalHaloArray(Float64, (3,), 1; boundary_condition=:repeating)
    interior_view(local_ha) .= [1.0, 2.0, 3.0]
    @test start_halo_exchange!(local_ha) === local_ha
    @test finish_halo_exchange!(local_ha) === local_ha
    @test synchronize_halo!(local_ha) === local_ha
    @test parent(local_ha) == [1.0, 1.0, 2.0, 3.0, 3.0]
end
