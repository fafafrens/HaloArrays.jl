using Test
using MPI
using HaloArrays

function _self_topology(dims::NTuple{N,Int}; periodic=ntuple(_ -> false, Val(N))) where {N}
    return CartesianTopology(MPI.COMM_SELF, dims; periodic=periodic)
end

@testset "Boundary conditions" begin
    @testset "1D halo fill" begin
        topology = _self_topology((1,))
        ha = HaloArray(
            Int,
            (4,),
            2,
            topology;
            boundary_condition=((Repeating(), Reflecting()),),
        )

        fill!(parent(ha), -1)
        interior_view(ha) .= [10, 20, 30, 40]

        boundary_condition!(ha)

        @test parent(ha)[1:2] == [10, 10]
        @test parent(ha)[3:6] == [10, 20, 30, 40]
        @test parent(ha)[7:8] == [40, 30]
    end

    @testset "antireflecting mirrors the boundary side" begin
        topology = _self_topology((1,))
        ha = HaloArray(
            Int,
            (4,),
            2,
            topology;
            boundary_condition=((Antireflecting(), Antireflecting()),),
        )

        fill!(parent(ha), -1)
        interior_view(ha) .= [10, 20, 30, 40]

        boundary_condition!(ha)

        @test parent(ha)[1:2] == [-20, -10]
        @test parent(ha)[7:8] == [-40, -30]
    end

    @testset "2D per-dimension boundary modes" begin
        topology = _self_topology((1, 1))
        ha = HaloArray(
            Int,
            (3, 4),
            1,
            topology;
            boundary_condition=((Reflecting(), Repeating()), (Antireflecting(), Reflecting())),
        )

        fill!(parent(ha), -1)
        interior = interior_view(ha)
        for i in 1:size(ha, 1), j in 1:size(ha, 2)
            interior[i, j] = 10 * i + j
        end

        boundary_condition!(ha)

        @test collect(get_recv_view(Side(1), Dim(1), ha)) == reshape([11, 12, 13, 14], 1, 4)
        @test collect(get_recv_view(Side(2), Dim(1), ha)) == reshape([31, 32, 33, 34], 1, 4)
        @test collect(get_recv_view(Side(1), Dim(2), ha)) == reshape([-11, -21, -31], 3, 1)
        @test collect(get_recv_view(Side(2), Dim(2), ha)) == reshape([14, 24, 34], 3, 1)
        @test collect(interior_view(ha)) == [10 * i + j for i in 1:3, j in 1:4]
    end

    @testset "periodicity validation" begin
        nonperiodic = _self_topology((1,))
        periodic = _self_topology((1,); periodic=(true,))

        @test_throws ErrorException HaloArray(Int, (3,), 1, nonperiodic; boundary_condition=:periodic)
        @test_throws ErrorException HaloArray(Int, (3,), 1, periodic; boundary_condition=:repeating)
        @test HaloArray(Int, (3,), 1, periodic; boundary_condition=:periodic) isa HaloArray
    end
end
