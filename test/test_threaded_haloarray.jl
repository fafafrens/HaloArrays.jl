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
        @test halo isa AbstractArray{Int,2}
        @test size(halo) == (12, 10)
        @test owned_size(halo) == (12, 10)
        @test global_size(halo) == (12, 10)
        @test axes(halo) == map(Base.OneTo, global_size(halo))
        @test owned_axes(halo) == map(Base.OneTo, owned_size(halo))
        @test halo_width(halo) == 2
        @test tile_count(halo) == 6
        @test storage_size(halo) == (8, 9)
        @test size(tile_parent(halo, 1)) == (8, 9)
        @test size(interior_view(halo, 1)) == (4, 5)
        @test_throws ErrorException ThreadedHaloArray(Int, (4,), 1; dims=(2,),
            boundary_condition=((Periodic(), Repeating()),))

        resized = similar(halo, Float32, (15, 12))
        @test resized isa ThreadedHaloArray
        @test eltype(resized) === Float32
        @test size(resized) == (15, 12)
        @test tile_size(resized) == (5, 6)
        @test storage_size(resized) == (9, 10)
        @test_throws DimensionMismatch similar(halo, Float32, (14, 12))

        resized_same_eltype = similar(halo, (15, 12))
        @test eltype(resized_same_eltype) === Int
        @test size(resized_same_eltype) == (15, 12)
        @test tile_size(resized_same_eltype) == (5, 6)
    end

    @testset "nonperiodic tile exchange and physical boundaries" begin
        halo = ThreadedHaloArray(Int, (3,), 1; dims=(2,), boundary_condition=:repeating)
        interior_view(halo, 1) .= [11, 12, 13]
        interior_view(halo, 2) .= [21, 22, 23]

        @test eltype(typeof(halo)) === Int
        @test halo[1] == 11
        @test halo[4] == 21
        halo[5] = 99
        @test halo[5] == 99
        @test interior_view(halo, 2)[2] == 99
        halo[5] = 22
        @test_throws BoundsError halo[0]
        @test_throws BoundsError setindex!(halo, 1, 7)

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

    @testset "copyto! copies threaded tile storage" begin
        src = ThreadedHaloArray(Int, (2, 3), 1; dims=(2, 2), boundary_condition=:repeating)
        dest = similar(src)

        for tile_id in 1:tile_count(src)
            tile_parent(src, tile_id) .= tile_id
            tile_parent(dest, tile_id) .= -1
        end

        @test copyto!(dest, src) === dest
        for tile_id in 1:tile_count(src)
            @test tile_parent(dest, tile_id) == tile_parent(src, tile_id)
        end

        wrong_size = similar(src, (6, 6))
        @test_throws DimensionMismatch copyto!(wrong_size, src)
    end

    @testset "explicit threaded halo synchronization variants match default path" begin
        function make_sync_test_halo()
            halo = ThreadedHaloArray(Int, (2, 3), 1; dims=(2, 2), boundary_condition=:repeating)
            fill!(halo, -1)
            for tile_id in 1:tile_count(halo)
                interior = interior_view(halo, tile_id)
                for I in CartesianIndices(interior)
                    i, j = Tuple(I)
                    interior[I] = 100 * tile_id + 10 * i + j
                end
            end
            return halo
        end

        same_storage(a, b) = all(tile_id -> tile_parent(a, tile_id) == tile_parent(b, tile_id), 1:tile_count(a))

        default_exchange = make_sync_test_halo()
        threaded_exchange = copy(default_exchange)
        halo_exchange!(default_exchange)
        @test halo_exchange_threads!(threaded_exchange) === threaded_exchange
        @test same_storage(threaded_exchange, default_exchange)

        default_boundary = make_sync_test_halo()
        threaded_boundary = copy(default_boundary)
        boundary_condition!(default_boundary)
        @test boundary_condition_threads!(threaded_boundary) === nothing
        @test same_storage(threaded_boundary, default_boundary)

        default_sync = make_sync_test_halo()
        threaded_sync = copy(default_sync)
        synchronize_halo!(default_sync)
        @test synchronize_halo_threads!(threaded_sync) === threaded_sync
        @test same_storage(threaded_sync, default_sync)
    end

    @testset "threaded multi halo array fieldwise exchange and boundary conditions" begin
        fields = ThreadedMultiHaloArray(
            Int,
            (3,),
            1;
            dims=(2,),
            boundary_conditions=(;
                rho=:repeating,
                mom=:antireflecting,
            ),
        )

        interior_view(fields.arrays.rho, 1) .= [11, 12, 13]
        interior_view(fields.arrays.rho, 2) .= [21, 22, 23]
        interior_view(fields.arrays.mom, 1) .= [31, 32, 33]
        interior_view(fields.arrays.mom, 2) .= [41, 42, 43]

        synchronize_halo!(fields)

        @test size(fields) == (2, 6)
        @test tile_size(fields) == (3,)
        @test tile_count(fields) == 2
        @test tile_coordinates(fields, 2) == (2,)
        @test keys(interior_view(fields, 1)) == (:rho, :mom)
        @test interior_view(fields, 1).rho == interior_view(fields.arrays.rho, 1)

        @test tile_parent(fields.arrays.rho, 1) == [11, 11, 12, 13, 21]
        @test tile_parent(fields.arrays.rho, 2) == [13, 21, 22, 23, 23]
        @test tile_parent(fields.arrays.mom, 1) == [-31, 31, 32, 33, 41]
        @test tile_parent(fields.arrays.mom, 2) == [33, 41, 42, 43, -43]

        copied = copy(fields)
        interior_view(copied.arrays.rho, 1)[1] = -1
        @test interior_view(fields.arrays.rho, 1)[1] == 11

        similar_fields = similar(fields, Float64)
        @test eltype(similar_fields) == Float64
        @test tile_size(similar_fields) == tile_size(fields)

        resized_fields = similar(fields, Float32, (2, 8))
        @test resized_fields isa MultiHaloArray
        @test eltype(resized_fields) === Float32
        @test size(resized_fields) == (2, 8)
        @test tile_size(resized_fields) == (4,)
        @test_throws DimensionMismatch similar(fields, Float32, (3, 8))
    end

    @testset "threaded halo array broadcast" begin
        a = ThreadedHaloArray(Int, (3,), 1; dims=(2,), boundary_condition=:repeating)
        b = similar(a)

        interior_view(a, 1) .= [1, 2, 3]
        interior_view(a, 2) .= [4, 5, 6]
        interior_view(b, 1) .= [10, 20, 30]
        interior_view(b, 2) .= [40, 50, 60]

        shifted = a .+ 2
        @test shifted isa ThreadedHaloArray
        @test collect(interior_view(shifted, 1)) == [3, 4, 5]
        @test collect(interior_view(shifted, 2)) == [6, 7, 8]

        dest = similar(a)
        dest .= 2 .* a .+ b
        @test collect(interior_view(dest, 1)) == [12, 24, 36]
        @test collect(interior_view(dest, 2)) == [48, 60, 72]
    end

    @testset "threaded multi halo array broadcast" begin
        fields = ThreadedMultiHaloArray(
            Int,
            (3,),
            1;
            dims=(2,),
            boundary_conditions=(;
                rho=:repeating,
                mom=:repeating,
            ),
        )

        interior_view(fields.arrays.rho, 1) .= [1, 2, 3]
        interior_view(fields.arrays.rho, 2) .= [4, 5, 6]
        interior_view(fields.arrays.mom, 1) .= [10, 20, 30]
        interior_view(fields.arrays.mom, 2) .= [40, 50, 60]

        shifted = fields .+ 5
        @test shifted isa MultiHaloArray
        @test shifted.arrays.rho isa ThreadedHaloArray
        @test shifted.arrays.mom isa ThreadedHaloArray
        @test collect(interior_view(shifted.arrays.rho, 1)) == [6, 7, 8]
        @test collect(interior_view(shifted.arrays.rho, 2)) == [9, 10, 11]
        @test collect(interior_view(shifted.arrays.mom, 1)) == [15, 25, 35]
        @test collect(interior_view(shifted.arrays.mom, 2)) == [45, 55, 65]

        dest = similar(fields)
        dest .= 3 .* fields .- 1
        @test collect(interior_view(dest.arrays.rho, 1)) == [2, 5, 8]
        @test collect(interior_view(dest.arrays.rho, 2)) == [11, 14, 17]
        @test collect(interior_view(dest.arrays.mom, 1)) == [29, 59, 89]
        @test collect(interior_view(dest.arrays.mom, 2)) == [119, 149, 179]
    end

    @testset "broadcast between different halo backends fails" begin
        threaded = ThreadedHaloArray(Int, (3,), 1; dims=(2,), boundary_condition=:repeating)
        local_halo = LocalHaloArray(Int, (6,), 1; boundary_condition=:repeating)

        @test size(threaded) == size(local_halo)
        @test_throws ArgumentError threaded .+ local_halo
        @test_throws ArgumentError local_halo .+ threaded

        threaded_dest = similar(threaded)
        local_dest = similar(local_halo)
        @test_throws ArgumentError threaded_dest .= threaded .+ local_halo
        @test_throws ArgumentError local_dest .= local_halo .+ threaded

        threaded_fields = ThreadedMultiHaloArray(
            Int,
            (3,),
            1;
            dims=(2,),
            boundary_conditions=(;
                rho=:repeating,
                mom=:repeating,
            ),
        )
        local_fields = LocalMultiHaloArray((;
            rho=LocalHaloArray(Int, (6,), 1; boundary_condition=:repeating),
            mom=LocalHaloArray(Int, (6,), 1; boundary_condition=:repeating),
        ))

        @test size(threaded_fields) == size(local_fields)
        @test_throws ArgumentError threaded_fields .+ local_fields
        @test_throws ArgumentError local_fields .+ threaded_fields

        threaded_fields_dest = similar(threaded_fields)
        local_fields_dest = similar(local_fields)
        @test_throws ArgumentError threaded_fields_dest .= threaded_fields .+ local_fields
        @test_throws ArgumentError local_fields_dest .= local_fields .+ threaded_fields
    end

    @testset "fill_from_global_indices! matches manual index loop" begin
        nthreads = max(1, Threads.nthreads())
        tsz = (4, 3)
        u = ThreadedHaloArray(Float64, tsz, 1;
            dims=(nthreads, 1), boundary_condition=:repeating)

        fill_from_global_indices!(u) do I
            Float64(I[1] * 10 + I[2])
        end

        nx, ny = tsz[1] * nthreads, tsz[2]
        expected = [Float64(i * 10 + j) for i in 1:nx, j in 1:ny]
        @test collect(reshape([u[i, j] for i in 1:nx, j in 1:ny], nx, ny)) == expected
    end

    @testset "fill_from_local_indices! fills per-tile interior" begin
        nthreads = max(1, Threads.nthreads())
        tsz = (3,)
        u = ThreadedHaloArray(Float64, tsz, 1;
            dims=(nthreads,), boundary_condition=:repeating)

        fill_from_local_indices!(u) do i
            Float64(i)
        end

        for tile_id in 1:tile_count(u)
            @test collect(interior_view(u, tile_id)) == [1.0, 2.0, 3.0]
        end
    end

    @testset "threaded multi halo array compatibility checks" begin
        rho = ThreadedHaloArray(Int, (3,), 1; dims=(2,), boundary_condition=:repeating)
        bad_tile_size = ThreadedHaloArray(Int, (4,), 1; dims=(2,), boundary_condition=:repeating)
        bad_halo = ThreadedHaloArray(Int, (3,), 2; dims=(2,), boundary_condition=:repeating)
        bad_topology = ThreadedHaloArray(Int, (3,), 1; dims=(3,), boundary_condition=:repeating)

        @test ThreadedMultiHaloArray((; rho)) isa MultiHaloArray
        @test_throws DimensionMismatch ThreadedMultiHaloArray((; rho, bad_tile_size))
        @test_throws DimensionMismatch ThreadedMultiHaloArray((; rho, bad_halo))
        @test_throws DimensionMismatch ThreadedMultiHaloArray((; rho, bad_topology))
        @test_throws ArgumentError ThreadedMultiHaloArray((; rho, local_halo=LocalHaloArray(Int, (3,), 1)))
    end
end
