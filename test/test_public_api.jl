using Test
using MPI
using HaloArrays

@testset "public API exports" begin
    ha = HaloArray(Float64, (5,), 2; boundary_condition=:repeating)

    @test ha isa HaloArray
    @test ha isa AbstractDistributedHaloArray
    @test ha isa AbstractSingleHaloArray
    @test ha isa AbstractHaloArray
    @test !(ha isa AbstractArray)
    @test interior_size(ha) == (5,)
    @test local_size(ha) == (5,)
    @test full_size(ha) == (9,)
    @test halo_width(ha) == 2
    @test size(ha) == global_size(ha)
    @test axes(ha) == map(Base.OneTo, global_size(ha))
    @test local_axes(ha) == axes(interior_view(ha))
    @test length(axes(ha, 1)) == size(ha, 1)
    @test length(local_axes(ha, 1)) == local_size(ha, 1)
    @test length(ha) == prod(global_size(ha))

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
    @test size(resized) == global_size(resized)
    @test local_size(resized) == (3,)
    @test full_size(resized) == (7,)

    local_ha = LocalHaloArray(Float64, (3,), 1; boundary_condition=:repeating)
    @test local_ha isa AbstractSerialHaloArray
    @test local_ha isa AbstractSingleHaloArray
    @test local_ha isa AbstractHaloArray
    @test !(local_ha isa AbstractArray)
    @test local_size(local_ha) == (3,)
    @test global_size(local_ha) == local_size(local_ha)
    @test axes(local_ha) == map(Base.OneTo, global_size(local_ha))
    @test local_axes(local_ha) == axes(interior_view(local_ha))
    interior_view(local_ha) .= [1.0, 2.0, 3.0]
    @test start_halo_exchange!(local_ha) === local_ha
    @test finish_halo_exchange!(local_ha) === local_ha
    @test synchronize_halo!(local_ha) === local_ha
    @test parent(local_ha) == [1.0, 1.0, 2.0, 3.0, 3.0]
end
