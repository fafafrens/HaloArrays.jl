using MPI, BenchmarkTools, Test
include("cartesian_topology.jl") 
include("haloarray.jl")
include("haloarrays.jl")
include("boundary.jl")        # <<-- boundary prima
include("interior_broadcast.jl")
include("halo_exchange.jl") 


function test_halo_exchange_1d_correctness()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    topo = CartesianTopology(comm, (0,))  # 1D
    halo = 1
    local_len = 4
    bd = ((Periodic(), Periodic()),)  # periodic left/right

    A = HaloArray(Int, (local_len,), halo, topo, bd)

    # Fill interior with predictable values: rank*100 + i
    for i in eachindex(interior_view(A))
        interior_view(A)[i] = rank * 100 + i
    end

    MPI.Barrier(comm)
    halo_exchange_async_unsafe!(A)
    MPI.Barrier(comm)

    for r in 0:MPI.Comm_size(comm)-1
    MPI.Barrier(comm)
    if rank == r
        println("====== Rank $rank ======")
        println("Left neighbor:  ", (rank - 1 + size) % size)
        println("Right neighbor: ", (rank + 1) % size)
        println("Full A.data:    ", collect(A.data))
        println("Interior:       ", interior_view(A))
        println("Left halo:      ", collect(@view A.data[1:halo]))
        println("Right halo:     ", collect(@view A.data[end-halo+1:end]))
        println()
    end
end
MPI.Barrier(comm)

    # Neighbor ranks (periodic)
left_rank  = (rank - 1 + size) % size
right_rank = (rank + 1) % size

# Expected halos
left_vals  = [(left_rank * 100 + local_len) for i in 1:halo]
right_vals = [(right_rank * 100 + 1) for i in 1:halo]

# Extract halos from data buffer
left_halo  = A.data[1:halo]
right_halo = A.data[end - halo + 1 : end]

@test left_halo == left_vals
@test right_halo == right_vals

println("✅ Rank $rank: halo exchange correctness verified.")
MPI.Barrier(comm)
MPI.Barrier(comm)
end

test_halo_exchange_1d_correctness()
