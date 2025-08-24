using MPI, BenchmarkTools, Test
include("cartesian_topology.jl") 
include("haloarray.jl")
include("haloarrays.jl")
include("boundary.jl")        # <<-- boundary prima
include("interior_broadcast.jl")
include("halo_exchange.jl") 

MPI.Init()

function test_halo_exchange_3d_correctness()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size_n = MPI.Comm_size(comm)

    dims = (0, 0, 0)
    topo = CartesianTopology(comm, dims; periodic=(true, true, true))

    if rank == 0
        println("Topology neighbors per rank:")
    end
    MPI.Barrier(comm)
    println("Rank $rank coords=$(topo.cart_coords), neighbors: x=$(topo.neighbors[1]), y=$(topo.neighbors[2]), z=$(topo.neighbors[3])")
    MPI.Barrier(comm)

    halo = 1
    local_nx, local_ny, local_nz = 4, 4, 4
    bd = ntuple(_ -> (Periodic(), Periodic()), 3)  # 3D periodic boundary

    A = HaloArray(Int, (local_nx, local_ny, local_nz), halo, topo, bd)

    # Fill interior with predictable values
    for i in 1:local_nx, j in 1:local_ny, k in 1:local_nz
        interior_view(A)[i, j, k] = rank * 100_000 + k + (j-1)*local_nz + (i-1)*local_ny*local_nz
    end

    MPI.Barrier(comm)
    halo_exchange_async_unsafe!(A)
    MPI.Barrier(comm)

    # Ordered printing
    for r in 0:size_n-1
        MPI.Barrier(comm)
        if rank == r
            println("====== Rank $rank ======")
            println("Coords: ", topo.cart_coords)
            println("Neighbors: x=$(topo.neighbors[1]), y=$(topo.neighbors[2]), z=$(topo.neighbors[3])")
            println("Interior slice z=1:")
            for i in local_nx+2:-1:1
                println(join(A.data[i, :, 2], ", "))
            end
            println()
            println("Interior slice z=1:")
            for i in local_nx:-1:1
                println(join(interior_view(A)[i, :, 1], ", "))
            end
            println()
        end
    end

    # Neighbor ranks
    left_rank   = topo.neighbors[1][1]
    right_rank  = topo.neighbors[1][2]
    down_rank   = topo.neighbors[2][1]
    up_rank     = topo.neighbors[2][2]
    back_rank   = topo.neighbors[3][1]
    front_rank  = topo.neighbors[3][2]

    # Expected halos (corrette dimensioni e ordine per h=1)
    # left/right: face di dimensione (local_ny, local_nz) proveniente da i = local_nx / i = 1 del vicino
    expected_left = [
        left_rank * 100_000 + k + (j-1)*local_nz + (local_nx-1)*local_ny*local_nz
        for j in 1:local_ny, k in 1:local_nz
    ]

    expected_right = [
        right_rank * 100_000 + k + (j-1)*local_nz + (0)*local_ny*local_nz
        for j in 1:local_ny, k in 1:local_nz
    ]

    # down/up: face di dimensione (local_nx, local_nz) proveniente da j = local_ny / j = 1 del vicino
    expected_down = [
        down_rank * 100_000 + k + (local_ny-1)*local_nz + (i-1)*local_ny*local_nz
        for i in 1:local_nx, k in 1:local_nz
    ]

    expected_up = [
        up_rank * 100_000 + k + (0)*local_nz + (i-1)*local_ny*local_nz
        for i in 1:local_nx, k in 1:local_nz
    ]

    # back/front: face di dimensione (local_nx, local_ny) proveniente da k = local_nz / k = 1 del vicino
    expected_back = [
        back_rank * 100_000 + local_nz + (j-1)*local_nz + (i-1)*local_ny*local_nz
        for i in 1:local_nx, j in 1:local_ny
    ]

    expected_front = [
        front_rank * 100_000 + 1 + (j-1)*local_nz + (i-1)*local_ny*local_nz
        for i in 1:local_nx, j in 1:local_ny
    ]

    # Actual halos (fixed: dim1=x, dim2=y, dim3=z)
    # left/right are faces at i = halo / i = halo+local_nx+1
    actual_left   = vec(A.data[halo,                 halo+1:halo+local_ny, halo+1:halo+local_nz])
    actual_right  = vec(A.data[halo+local_nx+1,     halo+1:halo+local_ny, halo+1:halo+local_nz])

    # down/up are faces at j = halo / j = halo+local_ny+1
    actual_down   = vec(A.data[halo+1:halo+local_nx, halo,                 halo+1:halo+local_nz])
    actual_up     = vec(A.data[halo+1:halo+local_nx, halo+local_ny+1,     halo+1:halo+local_nz])

    # front/back are faces at k = halo / k = halo+local_nz+1
    actual_back   = vec(A.data[halo+1:halo+local_nx, halo+1:halo+local_ny, halo])
    actual_front  = vec(A.data[halo+1:halo+local_nx, halo+1:halo+local_ny, halo+local_nz+1])

    # Tests per rank (ordered)
    for r in 0:size_n-1
        MPI.Barrier(comm)
        if rank == r
            println("====== Rank $rank halos ======")
            println("Left:    ", actual_left[1:8], " ... Expected: ", expected_left[1:8])
            println("Right:   ", actual_right[1:8], " ... Expected: ", expected_right[1:8])
            println("Down:    ", actual_down[1:8], " ... Expected: ", expected_down[1:8])
            println("Up:      ", actual_up[1:8], " ... Expected: ", expected_up[1:8])
            println("Back:    ", actual_back[1:8], " ... Expected: ", expected_back[1:8])
            println("Front:   ", actual_front[1:8], " ... Expected: ", expected_front[1:8])
            println()
        end
    end

    # Vector comparisons
    @test actual_left   == vec(expected_left)
    @test actual_right  == vec(expected_right)
    @test actual_down   == vec(expected_down)
    @test actual_up     == vec(expected_up)
    @test actual_back   == vec(expected_back)
    @test actual_front  == vec(expected_front)

    if rank == 0
        println("✅ 3D halo exchange correctness test passed.")
    end

    MPI.Barrier(comm)
end

test_halo_exchange_3d_correctness()

MPI.Barrier(MPI.COMM_WORLD)
MPI.Finalize()
