# to import MPIManager
#using MPIClusterManagers

# need to also import Distributed to use addprocs()
#using Distributed

# specify, number of mpi workers, launch cmd, etc.
#manager=MPIManager(np=4)

# start mpi workers and add them as julia workers too.
#addprocs(manager)

#@mpi_do manager begin
using MPI
using HDF5

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
include("save_hdf5.jl")

# -- MAIN SCRIPT --

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

dims = (2, 2) # 2x2 process grid for example
topo = CartesianTopology(comm, dims)



local_size = (4, 4)
halo_size = 1
bd=((Periodic(),Periodic()),(Periodic(),Periodic()))
halo_arr = HaloArray(Float64, local_size, halo_size, topo, bd)

# Initialize interior (without halo)
interior = interior_view(halo_arr)
interior .= rank + 1.0

MPI.Barrier(comm)

halo_exchange_waitall!(halo_arr)

MPI.Barrier(comm)

for rank in 0:topo.nprocs-1
    if rank == topo.global_rank
        println("%%%%%%%%%%%%%%%%%%%%%%")
        println("Rank $rank after halo exchange:")
        @show(halo_arr.data)
    end
    MPI.Barrier(topo.comm)
end


# --- Broadcast tests ---
halo_arr3 = halo_arr .+ 1000

halo_arr4 = similar(halo_arr)
halo_arr4 .= halo_arr .+ halo_arr .+ 200

halo_arr5= sin.(halo_arr)

MPI.Barrier(comm)

for r in 0:topo.nprocs - 1
    if r == rank
        println("%%%%%%%%%%%%%%%%%%%%%%")
        println("Rank $rank after broadcasting operations:")
        @show halo_arr3.data
        println("%%%%%%%%%%%%%%%%%%%%%%")
        @show halo_arr4.data
        println("%%%%%%%%%%%%%%%%%%%%%%")
        @show halo_arr5.data
    end
    MPI.Barrier(comm)
end


local_idx = (2, 2)

# Convert to global
global_idx = local_to_global_index(halo_arr, local_idx)

# Convert back to local
local_idx_back = global_to_local_index(halo_arr, global_idx)

for r in 0:topo.nprocs - 1
    if r == rank
println("%%%%%%%%%%%%%%%%%%%%%%")
println("Rank $rank:")
println("  Local index     = $local_idx")
println("  Global index    = $global_idx")
println("  Local back from global = $local_idx_back")
    end
    MPI.Barrier(comm)
end



MPI.Barrier(comm)
MPI.Finalize()




