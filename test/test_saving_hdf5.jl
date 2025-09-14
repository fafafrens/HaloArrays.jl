using MPI
using HDF5
using Test
using HaloArrays

# inizializza MPI una sola volta
MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)

# Simple test script
function test_append()
     local_inner_size = (4, 4)
     nprocs = MPI.Comm_size(COMM)
     # Assuming a 2D decomposition with equal processes per dimension
     pdims = (isqrt(nprocs), isqrt(nprocs))
     rank_coords = (RANK % pdims[1], RANK ÷ pdims[1])

     start_x = rank_coords[2] * local_inner_size[1] + 1
     stop_x = start_x + local_inner_size[1] - 1
     start_y = rank_coords[1] * local_inner_size[2] + 1
     stop_y = start_y + local_inner_size[2] - 1

     start = (start_x, start_y)
     stop = (stop_x, stop_y)
     halo = 1
     boundary = ((Periodic(), Periodic()), (Periodic(), Periodic()))
     halo = HaloArray(Float64, local_inner_size, halo, boundary_condition=boundary)
 
     filename = "halo_out.h5"
     # apri in modo robusto: se non esiste crea con "w"
     fid = try
         h5open(filename, "r+", COMM, MPI.Info())
     catch
         h5open(filename, "w", COMM, MPI.Info())
     end
     dset = create_dataset_from_haloarray(fid, "field", halo)
 
     for t in 0:4
         fill_interior(halo, RANK + t * 0.1)
         append_haloarray!(dset, halo)
     end
     size(dset)
     close(fid)
     MPI.Barrier(COMM)
     if RANK == 0
         println("✅ File saved to $filename")
     end
     
     # verifica leggendo il file collettivamente in modalità lettura
     fidr = h5open(filename, "r", COMM, MPI.Info())
     for t in 1:5
         slab = fidr["field"][t, start[1]:stop[1], start[2]:stop[2]]
         @test all(slab .== (RANK + (t - 1) * 0.1))
     end
     close(fidr)
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
    # Number of timesteps = 5
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
