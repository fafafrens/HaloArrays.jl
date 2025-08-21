using MPI
using Test
include("cartesian_topology.jl")
include("haloarray.jl")
include("haloarrays.jl")
include("interior_broadcast.jl")
include("interior_broadcast_marray.jl")
include("reduction.jl")
include("halo_exchange.jl")
include("boundary.jl")
include("reduce_dim.jl")
include("gather.jl")

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# Create 2×1 topology (2 ranks in x-direction)
topo = CartesianTopology(comm, (4, 1); periodic=(false, false))

# Local array shape (3×4) — same for each rank
local_size = (10, 4)
halo = HaloArray(Int, local_size, 1, topo)

# Fill based on global coordinates
fill_from_local_indices!(halo) do i, j
    rank * 10 + (j - 1) * size(halo, 1) + i
end

# Reduce along dim=2 (columns)
reduced = reduce_dim_distributed(halo, +; dim=2)

# Gather reduced results to sub_comm root
local_result = interior_view(reduced)

# Gather full reduced result on rank 0
# Only rank 0 can test against the global expected result
gathered_result = gather_haloarray(reduced)
    # Reconstruct the global halo for reference
 global_input = gather_haloarray(halo)

    MPI.Barrier(MPI.COMM_WORLD)
if rank == 0

    # Apply expected reduction manually
    expected = dropdims(mapreduce(identity, +, global_input; dims=2); dims=2)

    @test gathered_result == expected
end

MPI.Barrier(comm)
if rank == 0
    println("✓ Test passed for reduce_dim_distributed")
end

MPI.Finalize()