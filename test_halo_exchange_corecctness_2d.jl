using MPI, BenchmarkTools, Test
include("cartesian_topology.jl") 
include("haloarray.jl")
include("haloarrays.jl")
include("interior_broadcast.jl")
include("halo_exchange.jl") 


function test_halo_exchange_2d_correctness()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size_n = MPI.Comm_size(comm)
    N = 2
    topo = CartesianTopology(comm, (0,0),periodic=(true, true ))  # 2D
    if rank == 0
    println("Topology neighbors per rank:")
    end
MPI.Barrier(comm)
println("Rank $rank coords=$(topo.cart_coords), neighbors: left=$(topo.neighbors[1][1]), right=$(topo.neighbors[1][2]), down=$(topo.neighbors[2][1]), up=$(topo.neighbors[2][2])")
MPI.Barrier(comm)   
    halo = 1
    local_nx, local_ny = 4, 4
    bd = ((Periodic(), Periodic()), (Periodic(), Periodic()))
    #bd = ((Reflecting(), Reflecting()), ( Reflecting(),  Reflecting()))  # Use reflecting boundaries for testing
    A = HaloArray(Int, (local_nx, local_ny), halo, topo, bd)

    # Fill interior with predictable values: rank*1000 + linear index
    for i in 1:local_nx, j in 1:local_ny
       interior_view(A)[i,j] = rank * 1000 + j + (i-1)*local_ny
    end

    MPI.Barrier(comm)
    halo_exchange_async_unsafe!(A)
    MPI.Barrier(comm)

    # Ordered printing by rank
    for r in 0:size_n-1
        MPI.Barrier(comm)
        if rank == r
            println("====== Rank $rank ======")
            println("Coords: ", topo.cart_coords)
            println("neighbors: ", topo.neighbors)
            println("Neighbors: left=$(topo.neighbors[1][1]), right=$(topo.neighbors[1][2]), down=$(topo.neighbors[2][1]), up=$(topo.neighbors[2][2])")
            @show myrank = MPI.Comm_rank(topo.cart_comm)
            @show topo.cart_coords
            @show topo.neighbors
            println("Full A.data including halos:")
            for i in size(A.data,1):-1:1
                println(join(A.data[i, :], ", "))
            end
            println("\nInterior (without halos):")
            for i in local_nx:-1:1
                println(join(interior_view(A)[i, :], ", "))
            end
            println()
        end
        MPI.Barrier(comm)
    end

    # Compute expected halos values (left/right/down/up)
    #=
    left_rank  = topo.neighbors[1][1]
    right_rank = topo.neighbors[1][2]
    down_rank  = topo.neighbors[2][1]
    up_rank    = topo.neighbors[2][2]
    =#

    left_rank  = topo.neighbors[2][1]
    right_rank = topo.neighbors[2][2]
    up_rank    = topo.neighbors[1][2]
    down_rank  = topo.neighbors[1][1]
    #interior_view(A)[i,j] = rank * 1000 + j + (i-1)*local_ny
    expected_left  = [left_rank  * 1000 + local_ny + (i-1)*local_ny for i in 1:local_nx]
    expected_right = [right_rank * 1000 + 1 + (i-1)*local_ny for i in 1:local_nx]
    expected_down  = [down_rank  * 1000 + j + (local_nx - 1)*local_ny for j in 1:local_ny]
    expected_up    = [up_rank    * 1000 + j for j in 1:local_ny]

    # Extract halos from the data buffer (accounting for halo offset)
    actual_left   = A.data[halo+1 : halo+local_nx, halo]      # left halo column
    actual_right  = A.data[halo+1 : halo+local_nx, halo+local_ny+1]  # right halo column
    actual_down   = A.data[halo, halo+1 : halo+local_ny]     # bottom halo row
    actual_up     = A.data[halo+local_nx+1, halo+1 : halo+local_ny]  # top halo row
MPI.Barrier(comm)
for r in 0:size_n-1
        MPI.Barrier(comm)
        if rank == r
            println("====== Rank $rank halos ======")
            println("Left halo:  ", actual_left, " Expected: ", expected_left)
            println("Right halo: ", actual_right, " Expected: ", expected_right)
            println("Down halo:  ", actual_down, " Expected: ", expected_down)
            println("Up halo:    ", actual_up, " Expected: ", expected_up)
            println()
        end
    end
MPI.Barrier(comm)

    @test actual_left == expected_left
    @test actual_right == expected_right
    @test actual_down == expected_down
    @test actual_up == expected_up

    if rank == 0
        println("✅ 2D halo exchange correctness test passed.")
    end

    MPI.Barrier(comm)
    MPI.Finalize()
end

test_halo_exchange_2d_correctness()