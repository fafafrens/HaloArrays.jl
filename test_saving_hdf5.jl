using MPI
using HDF5
using Test


# Assumes your full `HaloArray` and `CartesianTopology` definitions are loaded
include("cartesian_topology.jl")
include("haloarray.jl")
include("haloarrays.jl") 
include("boundary.jl")
include("halo_exchange.jl")  
include("interior_broadcast.jl") 
include("interior_broadcast_marray.jl")
include("reduction.jl")  # Import MultiHaloArray and related functions
include("save_hdf5.jl")

# Simple test script
function test_append()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    local_inner_size = (4, 4)
    halo = 1
    boundary = ((Periodic(), Periodic()), (Periodic(), Periodic()))
    halo = HaloArray(Float64, local_inner_size, halo, boundary_condition=boundary)

    filename = "halo_out.h5"
    fid = h5open(filename, "r+", comm, MPI.Info())
    dset = create_dataset_from_haloarray(fid, "field", halo)

    for t in 0:4
        fill_interior(halo, rank + t * 0.1)
        append_haloarray!(dset, halo)
    end
    size(dset)
    close(fid)
    MPI.Barrier(comm)
    if rank == 0
        println("✅ File saved to $filename")
    end
    MPI.Finalize()

    fid = h5open(filename, comm, MPI.Info())

    size(fid["field"]) 

    for t in 1:5
        data =fid["field"][t, :, :]
        @test all(data .== (rank + (t - 1) * 0.1))
    end
end



function test_fixedsize_hdf5()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    # Example: 2D grid (2x2 processes) assumed
    if nprocs != 4
        if rank == 0
            println("Please run with 4 MPI processes for this test")
        end
        MPI.Barrier(comm)
        MPI.Finalize()
        return  
    end

    # Create a HaloArray with local data and topology
    local_inner_size = (4, 4)
    # Assign coordinates manually for simplicity; in your real code, use CartesianTopology
    
    
    halo = HaloArray(local_inner_size, 1)

    fill_interior(halo,rank + 1.0)
    # Number of timesteps
    num_timesteps = 5
    filename = "halo_fixedsize.h5"

    # Open file with MPI support
    fid = h5open(filename, "w", comm, MPI.Info())

    # Create fixed-size dataset
    dset = create_fixedsize_dataset_from_haloarray(fid, "field", halo, num_timesteps)

    # Write data for each timestep
    for t in 0:num_timesteps-1
        # update data arbitrarily for demo
        fill!(halo.data, rank + 1.0 + t*0.1)
        write_haloarray_timestep!(dset, halo, t)
    end

    close(fid)
    MPI.Barrier(comm)

    # Rank 0 reads back and prints summary
    if rank == 0
        fid = h5open(filename, "r")
        dset = fid["field"]
        println("Dataset size: ", size(dset))
        # Read entire dataset
        all_data = read(dset)
        println("Sample data slice at time 1, block (0,0):")
        println(all_data[2, 1:4, 1:4])  # time=1 (index 2 in Julia 1-based)
        close(fid)
    end

    #MPI.Finalize()
end

test_fixedsize_hdf5()
test_append()

MPI.Finalize()
# Run the test
