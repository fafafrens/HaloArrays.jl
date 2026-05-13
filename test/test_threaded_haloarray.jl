@testset "ThreadedHaloArray" begin
    @testset "precomputed local Cartesian neighbors" begin
        topology = ThreadedCartesianTopology((2, 3); periodic=(false, true))

        @test tile_count(topology) == 6
        @test tile_coordinates(topology, 1) == (1, 1)
        @test tile_coordinates(topology, 6) == (2, 3)
        @test neighbor_tile_id(topology, 1, 1, 1) == 0
        @test neighbor_tile_id(topology, 1, 1, 2) == 2
        @test neighbor_tile_id(topology, 1, 2, 1) == 5
        @test neighbor_tile_id(topology, 1, 2, 2) == 3
    end

    @testset "constructor uses tile size and derives owned size" begin
        halo = ThreadedHaloArray(Int, (4, 5), 2; dims=(3, 2), boundary_condition=:repeating)

        @test tile_size(halo) == (4, 5)
        @test size(halo) == (12, 10)
        @test global_size(halo) == (12, 10)
        @test halo_width(halo) == 2
        @test tile_count(halo) == 6
        @test full_size(halo) == (8, 9)
        @test size(tile_parent(halo, 1)) == (8, 9)
        @test size(interior_view(halo, 1)) == (4, 5)
        @test_throws ErrorException ThreadedHaloArray(Int, (4,), 1; dims=(2,),
            boundary_condition=((Periodic(), Repeating()),))
    end

    @testset "nonperiodic tile exchange and physical boundaries" begin
        halo = ThreadedHaloArray(Int, (3,), 1; dims=(2,), boundary_condition=:repeating)
        interior_view(halo, 1) .= [11, 12, 13]
        interior_view(halo, 2) .= [21, 22, 23]

        synchronize_halo!(halo)

        @test tile_parent(halo, 1) == [11, 11, 12, 13, 21]
        @test tile_parent(halo, 2) == [13, 21, 22, 23, 23]
    end

    @testset "reflecting physical boundaries only apply on exterior tile sides" begin
        halo = ThreadedHaloArray(Int, (3,), 2; dims=(2,), boundary_condition=:reflecting)
        interior_view(halo, 1) .= [11, 12, 13]
        interior_view(halo, 2) .= [21, 22, 23]

        synchronize_halo!(halo)

        @test tile_parent(halo, 1) == [12, 11, 11, 12, 13, 21, 22]
        @test tile_parent(halo, 2) == [12, 13, 21, 22, 23, 23, 22]
    end

    @testset "antireflecting physical boundaries only apply on exterior tile sides" begin
        halo = ThreadedHaloArray(Int, (3,), 2; dims=(2,), boundary_condition=:antireflecting)
        interior_view(halo, 1) .= [11, 12, 13]
        interior_view(halo, 2) .= [21, 22, 23]

        synchronize_halo!(halo)

        @test tile_parent(halo, 1) == [-12, -11, 11, 12, 13, 21, 22]
        @test tile_parent(halo, 2) == [12, 13, 21, 22, 23, -23, -22]
    end

    @testset "different lower and upper physical boundary modes" begin
        halo = ThreadedHaloArray(Int, (3,), 2; dims=(2,),
            boundary_condition=((Repeating(), Antireflecting()),))
        interior_view(halo, 1) .= [11, 12, 13]
        interior_view(halo, 2) .= [21, 22, 23]

        synchronize_halo!(halo)

        @test tile_parent(halo, 1) == [11, 11, 11, 12, 13, 21, 22]
        @test tile_parent(halo, 2) == [12, 13, 21, 22, 23, -23, -22]
    end

    @testset "periodic tile exchange" begin
        halo = ThreadedHaloArray(Int, (3,), 1; dims=(2,), boundary_condition=:periodic)
        interior_view(halo, 1) .= [11, 12, 13]
        interior_view(halo, 2) .= [21, 22, 23]

        synchronize_halo!(halo)

        @test tile_parent(halo, 1) == [23, 11, 12, 13, 21]
        @test tile_parent(halo, 2) == [13, 21, 22, 23, 11]
    end

    @testset "single periodic tile wraps to itself" begin
        halo = ThreadedHaloArray(Int, (4,), 2; dims=(1,), boundary_condition=:periodic)
        interior_view(halo, 1) .= [1, 2, 3, 4]

        synchronize_halo!(halo)

        @test tile_parent(halo, 1) == [3, 4, 1, 2, 3, 4, 1, 2]
    end

    @testset "2D copies use neighboring tile faces" begin
        halo = ThreadedHaloArray(Int, (2, 3), 1; dims=(2, 2), boundary_condition=:repeating)

        for tile_id in 1:tile_count(halo)
            interior = interior_view(halo, tile_id)
            for I in CartesianIndices(interior)
                i, j = Tuple(I)
                interior[I] = 100 * tile_id + 10 * i + j
            end
        end

        synchronize_halo!(halo)

        @test collect(get_recv_view(Side(2), Dim(1), halo, 1)) ==
            collect(get_send_view(Side(1), Dim(1), halo, 2))
        @test collect(get_recv_view(Side(2), Dim(2), halo, 1)) ==
            collect(get_send_view(Side(1), Dim(2), halo, 3))
        @test collect(get_recv_view(Side(1), Dim(1), halo, 1)) ==
            collect(get_send_view(Side(1), Dim(1), halo, 1))
        @test collect(get_recv_view(Side(2), Dim(2), halo, 4)) ==
            collect(get_send_view(Side(2), Dim(2), halo, 4))
    end

    @testset "2D mixed boundary modes fill only exterior faces" begin
        halo = ThreadedHaloArray(Int, (2, 3), 1; dims=(1, 1),
            boundary_condition=((Reflecting(), Antireflecting()), (Repeating(), Reflecting())))
        interior = interior_view(halo, 1)
        for I in CartesianIndices(interior)
            i, j = Tuple(I)
            interior[I] = 10 * i + j
        end

        synchronize_halo!(halo)

        @test collect(get_recv_view(Side(1), Dim(1), halo, 1)) == reshape([11, 12, 13], 1, 3)
        @test collect(get_recv_view(Side(2), Dim(1), halo, 1)) == reshape([-21, -22, -23], 1, 3)
        @test collect(get_recv_view(Side(1), Dim(2), halo, 1)) == reshape([11, 21], 2, 1)
        @test collect(get_recv_view(Side(2), Dim(2), halo, 1)) == reshape([13, 23], 2, 1)
    end
end
