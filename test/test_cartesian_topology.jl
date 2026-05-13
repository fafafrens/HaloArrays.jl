using Test
using MPI
using HaloArrays

@testset "CartesianTopology" begin
    @testset "inactive topology" begin
        topology = HaloArrays.inactive_cartesian_topology((2, 3))

        @test !isactive(topology)
        @test topology.dims == (2, 3)
        @test topology.global_rank == MPI.PROC_NULL
        @test topology.cart_coords == (MPI.PROC_NULL, MPI.PROC_NULL)
        @test topology.neighbors == ((MPI.PROC_NULL, MPI.PROC_NULL), (MPI.PROC_NULL, MPI.PROC_NULL))
        @test topology.comm == MPI.COMM_NULL
        @test topology.cart_comm == MPI.COMM_NULL
    end

    @testset "single-rank topology" begin
        topology = CartesianTopology(MPI.COMM_SELF, (1, 1, 1); periodic=(false, true, false))

        @test isactive(topology)
        @test topology.nprocs == 1
        @test topology.dims == (1, 1, 1)
        @test topology.global_rank == 0
        @test topology.cart_coords == (0, 0, 0)
        @test topology.periodic_boundary_condition == (false, true, false)
        @test topology.neighbors[1] == (MPI.PROC_NULL, MPI.PROC_NULL)
        @test topology.neighbors[2] == (0, 0)
        @test topology.neighbors[3] == (MPI.PROC_NULL, MPI.PROC_NULL)
    end

    @testset "MPI integer dimensions" begin
        topology = CartesianTopology(MPI.COMM_SELF, (Int32(1), Int32(1)); periodic=(true, false))

        @test isactive(topology)
        @test topology.dims == (1, 1)
        @test topology.dims isa NTuple{2,Int}
        @test topology.periodic_boundary_condition == (true, false)
    end

    @testset "slice coloring" begin
        @test HaloArrays.coords_to_color_multi((1, 2, 3), (2, 3, 4), (2,)) == 7
        @test HaloArrays.coords_to_color_multi((1, 2, 3), (2, 3, 4), (1, 3)) == 2
    end
end
