
function create_dataset_from_haloarray(g, name::String, halo::HaloArray)
    local_size = size(halo)
    T = eltype(halo)
    N = ndims(halo)
    global_dims = global_size(halo)

    # Dataset dimensions: (time, global_dims...)
    initial_size = (0, global_dims...)
    max_size = (-1, global_dims...)
    chunk = (1, local_size...)

    # Create dataspace from dims
    
    dspace = dataspace(initial_size; max_dims=max_size)
    
    if haskey(g, name)
        # Open existing dataset
        return HDF5.open_dataset(g, name)
    end
    # Create dataset and specify chunk size as a keyword argument
    dset = HDF5.create_dataset(g, name, T, dspace; chunk=chunk)

    return dset
end


function append_haloarray!(dset::HDF5.Dataset, halo::HaloArray)
    global_dims = global_size(halo)
    local_data = interior_view(halo)
    local_dims = size(local_data)

    # Current dataset size (dims)
    curr_dims = size(dset)

    # New size after appending one time slice
    new_dims = (curr_dims[1] + 1, curr_dims[2:end]...)

    # Extend the dataset along the time dimension
    HDF5.set_extent_dims(dset, new_dims)

    # Compute the offset index in the dataset for this rank (process)
    coords = halo.topology.cart_coords
    offset_spatial = ntuple(i -> coords[i] * local_dims[i] + 1, length(coords))

    # Hyperslab slices for new timestep = last index of first dimension
    timestep_index = new_dims[1]
    slices = (timestep_index, ntuple(i -> offset_spatial[i]:(offset_spatial[i] + local_dims[i] - 1), length(offset_spatial))...)

    # Write local data into the hyperslab
    dset[slices...] = local_data

    return nothing
end



function append_haloarray_to_file!(file::String, dataset_name::String, halo::HaloArray)

    file *= ".h5"
    # Ensure the file exists and is opened with the correct communicator
    if !isfile(file)
        h5open(file, "w") do _ end  # Create an empty file if it doesn't exist
    end

    comm = get_comm(halo)

    h5open(file, "r+", comm) do fid

    # Create or open the dataset
        dset = haskey(fid, dataset_name) ? fid[dataset_name] :
            create_dataset_from_haloarray(fid, dataset_name, halo)

    # Append the current interior halo data to the dataset
        append_haloarray!(dset, halo)

    end 

    return nothing
end



function create_fixedsize_dataset_from_haloarray(g, name::String, halo, num_timesteps::Int)
    local_size = size(halo)
    T = eltype(halo)
    global_dims = global_size(halo)

    initial_size = (num_timesteps, global_dims...)
    max_size = initial_size  # fixed size, no extension
    chunk = (1, local_size...)

    dspace = HDF5.dataspace(initial_size; max_dims=max_size)
    dset = HDF5.create_dataset(g, name, T, dspace; chunk=chunk)
    return dset
end



function write_haloarray_timestep!(dset, halo, timestep)
    local_data = interior_view(halo)
    local_dims = size(local_data)

    coords = halo.topology.cart_coords
    offset_spatial = ntuple(i -> coords[i] * local_dims[i] + 1, length(coords))
    slices = (timestep + 1, ntuple(i -> offset_spatial[i]:(offset_spatial[i] + local_dims[i] - 1), length(offset_spatial))...)

    dset[slices...] = local_data
    return nothing
end


function create_haloarray_output_file(filename::String, dataset_name::String,
                                      halo::HaloArray, num_timesteps::Int)
    comm = get_comm(halo)
    rank = MPI.Comm_rank(comm)

    # Rank 0 creates the file (if not existing)
    if rank == 0 && !isfile(filename)
        h5open(filename, "w",comm) do _ end
    end
    #MPI.Barrier(comm)  # Synchronize all ranks before proceeding

    # Open the file in parallel mode
    fid = h5open(filename, "r+",comm)

    # Create dataset if it doesn't already exist
    dset = haskey(fid, dataset_name) ?
            fid[dataset_name] :
            create_fixedsize_dataset_from_haloarray(fid, dataset_name, halo, num_timesteps)

    return fid, dset
end


"""
    save_array_hdf5(filename::String, data::AbstractArray, comm::MPI.Comm; root::Int=0)

Save the array `data` to an HDF5 file with path `filename` on the MPI `root` process.
Only the `root` rank writes the file; others do nothing.
"""
function save_array_hdf5(filename::String, data, comm::MPI.Comm; root::Int=0)
    rank = MPI.Comm_rank(comm)
    if rank == root
        h5open(filename*".h5", "w") do file
            write(file, "dataset", data)
        end
        println("Rank $rank wrote data to $filename")
    else
        # Non-root ranks do nothing
        nothing
    end
end

function gather_and_save_haloarray(filename::String, halo::HaloArray; root::Int=0)
    comm = halo.topology.cart_comm
    gathered = gather_haloarray(halo; root=root)
    if MPI.Comm_rank(comm) == root
        save_array_hdf5(filename, gathered, comm; root=root)
    end
end


"""
    gather_and_append_haloarray!(filename::String, dataset::String, halo::HaloArray, comm::MPI.Comm; root::Int=0)

Effettua il gather del `HaloArray` sul root e lo aggiunge come nuovo timestep al dataset HDF5, indipendentemente dalla dimensione.
"""
function gather_and_append_haloarray!(filename::String, dataset::String, halo::HaloArray; root::Int=0)
    comm= halo.topology.cart_comm
    rank = MPI.Comm_rank(comm)
    gathered = gather_haloarray(halo; root=root)
    #@show sum(gathered) ,size(gathered)
    filename_h5 = filename * ".h5"
    
    if rank == root

        if !isfile(filename_h5)
            h5open(filename_h5, "w") do _ end
        end

        h5open(filename*".h5", "r+") do file
            if haskey(file, dataset)
                dset = file[dataset]
                curr_dims = size(dset)
                new_dims = (curr_dims[1] + 1, curr_dims[2:end]...)
                HDF5.set_extent_dims(dset, new_dims)
                # Costruisci la tupla di indici: (nuovo_t, :, :, ...)
                #@show new_dims 
                inds = (new_dims[1], ntuple(_ -> Colon(), ndims(gathered))...)
                #@show inds
                #@show inds
                #@show dset[inds...], gathered
                dset[inds...] = gathered
                #@show dset[inds...] , gathered
            else
                # Crea dataset con dimensione temporale estendibile
                global_dims = size(gathered)
                dspace = HDF5.dataspace((1, global_dims...); max_dims=(-1, global_dims...))
                dset = HDF5.create_dataset(file, dataset, eltype(gathered), dspace; chunk=(1, global_dims...))
                
                inds = (1, ntuple(_ -> Colon(), ndims(gathered))...)
                #@show inds
                #@show dset[inds...], gathered
                dset[inds...] = gathered
                #@show dset[inds...] , gathered
            end
        end
    end
    MPI.Barrier(comm)
end