@inline _hdf5_dataset_dims(halo::AbstractSingleHaloArray) =
    global_size(halo)
@inline _hdf5_dataset_dims(halo::ArrayOfHaloArray) =
    (field_shape(halo)..., _hdf5_dataset_dims(first(parent(halo)))...)

@inline _hdf5_chunk_dims(halo::HaloArray) = interior_size(halo)
@inline _hdf5_chunk_dims(halo::AbstractSerialHaloArray) =
    _hdf5_dataset_dims(halo)
@inline _hdf5_chunk_dims(halo::ArrayOfHaloArray) =
    (field_shape(halo)..., _hdf5_chunk_dims(first(parent(halo)))...)

@inline _hdf5_comm(halo::HaloArray) = get_comm(halo)
@inline _hdf5_comm(::AbstractSerialHaloArray) = nothing
@inline _hdf5_comm(halo::AbstractHaloCollection) = _hdf5_comm(_first_field(halo))

@inline _hdf5_field_name(name::Symbol) = String(name)
@inline _hdf5_field_name(name) = string(name)

function _hdf5_open_or_create_group(parent, name::String)
    return haskey(parent, name) ? HDF5.open_group(parent, name) : HDF5.create_group(parent, name)
end

function _hdf5_snapshot(halo::LocalHaloArray)
    return Array(interior_view(halo))
end

function _hdf5_snapshot(halo::ThreadedHaloArray{T,N}) where {T,N}
    data = Array{T}(undef, global_size(halo))
    owned_tile_size = tile_size(halo)

    for tile_id in 1:tile_count(halo)
        coords = tile_coordinates(halo, tile_id)
        inds = ntuple(Val(N)) do d
            first_owned = (coords[d] - 1) * owned_tile_size[d] + 1
            last_owned = coords[d] * owned_tile_size[d]
            first_owned:last_owned
        end
        data[inds...] .= interior_view(halo, tile_id)
    end

    return data
end

function _hdf5_snapshot(halo::ArrayOfHaloArray)
    first(parent(halo)) isa HaloArray &&
        throw(ArgumentError("snapshot assembly for MPI ArrayOfHaloArray is not supported; write it collectively with append_haloarray! or write_haloarray_timestep!"))

    data = Array{eltype(halo)}(undef, _hdf5_dataset_dims(halo))
    for I in CartesianIndices(parent(halo))
        field_data = _hdf5_snapshot(parent(halo)[I])
        inds = (Tuple(I)..., ntuple(_ -> Colon(), ndims(field_data))...)
        data[inds...] .= field_data
    end

    return data
end

function _hdf5_snapshot(halo::MultiHaloArray)
    _hdf5_comm(halo) === nothing ||
        throw(ArgumentError("snapshot assembly for MPI MultiHaloArray is not supported; use gather_and_save_haloarray or write it collectively with append_haloarray!"))

    fields = map(_hdf5_snapshot, values(halo.arrays))
    return NamedTuple{keys(halo.arrays)}(fields)
end

function _hdf5_gather_snapshot(halo::HaloArray; root::Int=0)
    return gather_haloarray(halo; root=root)
end

function _hdf5_gather_snapshot(halo::AbstractSerialHaloArray; root::Int=0)
    return _hdf5_snapshot(halo)
end

function _hdf5_gather_snapshot(halo::ArrayOfHaloArray; root::Int=0)
    comm = _hdf5_comm(halo)
    comm === nothing && return _hdf5_snapshot(halo)

    rank = MPI.Comm_rank(comm)
    data = nothing
    for I in CartesianIndices(parent(halo))
        field_data = gather_haloarray(parent(halo)[I]; root=root)
        if rank == root
            if data === nothing
                data = Array{eltype(halo)}(undef, (field_shape(halo)..., size(field_data)...))
            end
            inds = (Tuple(I)..., ntuple(_ -> Colon(), ndims(field_data))...)
            data[inds...] .= field_data
        end
    end

    return data
end

function _hdf5_gather_snapshot(halo::MultiHaloArray; root::Int=0)
    fields = map(values(halo.arrays)) do field
        _hdf5_gather_snapshot(field; root=root)
    end
    return NamedTuple{keys(halo.arrays)}(fields)
end

function _hdf5_write_snapshot!(parent, name::String, data::AbstractArray)
    write(parent, name, data)
    return nothing
end

function _hdf5_write_snapshot!(parent, name::String, data::NamedTuple)
    group = HDF5.create_group(parent, name)
    for (field_name, field_data) in pairs(data)
        _hdf5_write_snapshot!(group, _hdf5_field_name(field_name), field_data)
    end
    return nothing
end

function _hdf5_save_snapshot(filename::String, data; dataset::String="dataset")
    h5open(filename*".h5", "w") do file
        _hdf5_write_snapshot!(file, dataset, data)
    end
    return nothing
end

function _hdf5_open(filename::String, mode::String, comm)
    return comm === nothing ? h5open(filename, mode) : h5open(filename, mode, comm, MPI.Info())
end

function _hdf5_open(f::Function, filename::String, mode::String, comm)
    fid = _hdf5_open(filename, mode, comm)
    try
        return f(fid)
    finally
        close(fid)
    end
end

function _hdf5_write_timestep!(dset, halo::HaloArray, time_index::Integer)
    local_data = interior_view(halo)
    local_dims = size(local_data)
    coords = halo.topology.cart_coords
    offset_spatial = ntuple(i -> coords[i] * local_dims[i] + 1, length(coords))
    slices = (time_index, ntuple(i -> offset_spatial[i]:(offset_spatial[i] + local_dims[i] - 1), length(offset_spatial))...)

    dset[slices...] = local_data
    return nothing
end

function _hdf5_write_timestep!(dset, halo::AbstractSerialHaloArray, time_index::Integer)
    data = _hdf5_snapshot(halo)
    slices = (time_index, ntuple(_ -> Colon(), ndims(data))...)
    dset[slices...] = data
    return nothing
end

function _hdf5_write_field_timestep!(dset, field::HaloArray, time_index::Integer, field_index)
    local_data = interior_view(field)
    local_dims = size(local_data)
    coords = field.topology.cart_coords
    offset_spatial = ntuple(i -> coords[i] * local_dims[i] + 1, length(coords))
    spatial_slices = ntuple(i -> offset_spatial[i]:(offset_spatial[i] + local_dims[i] - 1), length(offset_spatial))
    slices = (time_index, field_index..., spatial_slices...)

    dset[slices...] = local_data
    return nothing
end

function _hdf5_write_field_timestep!(dset, field::AbstractSerialHaloArray, time_index::Integer, field_index)
    field_data = _hdf5_snapshot(field)
    slices = (time_index, field_index..., ntuple(_ -> Colon(), ndims(field_data))...)
    dset[slices...] = field_data
    return nothing
end

function _hdf5_write_timestep!(dset, halo::ArrayOfHaloArray, time_index::Integer)
    for I in CartesianIndices(parent(halo))
        _hdf5_write_field_timestep!(dset, parent(halo)[I], time_index, Tuple(I))
    end
    return nothing
end

function _hdf5_write_timestep!(group::HDF5.Group, halo::MultiHaloArray, time_index::Integer)
    for (field_name, field) in pairs(halo.arrays)
        child = group[_hdf5_field_name(field_name)]
        _hdf5_write_timestep!(child, field, time_index)
    end
    return nothing
end

function _create_hdf5_dataset_from_haloarray(g, name::String, halo)
    T = eltype(halo)
    global_dims = _hdf5_dataset_dims(halo)

    initial_size = (0, global_dims...)
    max_size = (-1, global_dims...)
    chunk = (1, _hdf5_chunk_dims(halo)...)

    dspace = dataspace(initial_size; max_dims=max_size)

    if haskey(g, name)
        return HDF5.open_dataset(g, name)
    end

    dset = HDF5.create_dataset(g, name, T, dspace; chunk=chunk)
    return dset
end

function create_dataset_from_haloarray(g, name::String, halo::AbstractSingleHaloArray)
    return _create_hdf5_dataset_from_haloarray(g, name, halo)
end

function create_dataset_from_haloarray(g, name::String, halo::ArrayOfHaloArray)
    return _create_hdf5_dataset_from_haloarray(g, name, halo)
end

function create_dataset_from_haloarray(g, name::String, halo::MultiHaloArray)
    group = _hdf5_open_or_create_group(g, name)
    for (field_name, field) in pairs(halo.arrays)
        create_dataset_from_haloarray(group, _hdf5_field_name(field_name), field)
    end
    return group
end

function create_dataset_from_haloarray(g, name::String, halo::MaybeHaloArray)
    isactive(halo) || return nothing
    return create_dataset_from_haloarray(g, name, getdata(halo))
end

function _append_hdf5_dataset!(dset::HDF5.Dataset, halo)
    curr_dims = size(dset)
    new_dims = (curr_dims[1] + 1, curr_dims[2:end]...)
    HDF5.set_extent_dims(dset, new_dims)

    _hdf5_write_timestep!(dset, halo, new_dims[1])
    return nothing
end

function append_haloarray!(dset::HDF5.Dataset, halo::AbstractSingleHaloArray)
    return _append_hdf5_dataset!(dset, halo)
end

function append_haloarray!(dset::HDF5.Dataset, halo::ArrayOfHaloArray)
    return _append_hdf5_dataset!(dset, halo)
end

function append_haloarray!(group::HDF5.Group, halo::MultiHaloArray)
    for (field_name, field) in pairs(halo.arrays)
        child_name = _hdf5_field_name(field_name)
        child = haskey(group, child_name) ? group[child_name] :
            create_dataset_from_haloarray(group, child_name, field)
        append_haloarray!(child, field)
    end
    return nothing
end

function append_haloarray!(dset::HDF5.Dataset, halo::MaybeHaloArray)
    isactive(halo) || return nothing
    return append_haloarray!(dset, getdata(halo))
end

function append_haloarray!(group::HDF5.Group, halo::MaybeHaloArray)
    isactive(halo) || return nothing
    return append_haloarray!(group, getdata(halo))
end

"""
    append_haloarray_to_file!(file, dataset_name, halo)

Append the interior data of `halo` to an HDF5 file (the `.h5` suffix is added to
`file`), creating it on first write and growing an extendable `dataset_name`
dataset by one slab per call — the idiom for streaming successive timesteps to a
single dataset. For a distributed `HaloArray` each rank writes its own block
collectively (parallel HDF5). See also [`gather_and_save_haloarray`](@ref) for
gathering to a single dense array on the root rank instead.
"""
function append_haloarray_to_file!(file::String, dataset_name::String, halo::AbstractHaloArray)
    file *= ".h5"
    comm = _hdf5_comm(halo)
    mode = isfile(file) ? "r+" : "w"

    _hdf5_open(file, mode, comm) do fid
        dset = haskey(fid, dataset_name) ? fid[dataset_name] :
            create_dataset_from_haloarray(fid, dataset_name, halo)

        append_haloarray!(dset, halo)
    end

    return nothing
end

function append_haloarray_to_file!(file::String, dataset_name::String, halo::MaybeHaloArray)
    isactive(halo) || return nothing
    return append_haloarray_to_file!(file, dataset_name, getdata(halo))
end

function _create_fixedsize_hdf5_dataset_from_haloarray(g, name::String, halo, num_timesteps::Int)
    T = eltype(halo)
    global_dims = _hdf5_dataset_dims(halo)

    initial_size = (num_timesteps, global_dims...)
    max_size = initial_size
    chunk = (1, _hdf5_chunk_dims(halo)...)

    dspace = HDF5.dataspace(initial_size; max_dims=max_size)
    dset = HDF5.create_dataset(g, name, T, dspace; chunk=chunk)
    return dset
end

function create_fixedsize_dataset_from_haloarray(g, name::String, halo::AbstractSingleHaloArray, num_timesteps::Int)
    return _create_fixedsize_hdf5_dataset_from_haloarray(g, name, halo, num_timesteps)
end

function create_fixedsize_dataset_from_haloarray(g, name::String, halo::ArrayOfHaloArray, num_timesteps::Int)
    return _create_fixedsize_hdf5_dataset_from_haloarray(g, name, halo, num_timesteps)
end

function create_fixedsize_dataset_from_haloarray(g, name::String, halo::MultiHaloArray, num_timesteps::Int)
    group = _hdf5_open_or_create_group(g, name)
    for (field_name, field) in pairs(halo.arrays)
        create_fixedsize_dataset_from_haloarray(group, _hdf5_field_name(field_name), field, num_timesteps)
    end
    return group
end

function create_fixedsize_dataset_from_haloarray(g, name::String, halo::MaybeHaloArray, num_timesteps::Int)
    isactive(halo) || return nothing
    return create_fixedsize_dataset_from_haloarray(g, name, getdata(halo), num_timesteps)
end

"""
    write_haloarray_timestep!(dset, halo, timestep)

Write the interior (ghost-free) data of `halo` into the time-resolved dataset
`dset` (from [`create_haloarray_output_file`](@ref)) at index `timestep`
(0-based). Under MPI each rank writes its own subdomain collectively.
"""
function write_haloarray_timestep!(dset, halo::AbstractHaloArray, timestep)
    _hdf5_write_timestep!(dset, halo, timestep + 1)
    return nothing
end

function write_haloarray_timestep!(dset, halo::MaybeHaloArray, timestep)
    isactive(halo) || return nothing
    return write_haloarray_timestep!(dset, getdata(halo), timestep)
end

"""
    create_haloarray_output_file(filename, dataset_name, halo, num_timesteps) -> (file, dataset)

Open (or create) the HDF5 file `filename` and return it with a fixed-size
`dataset` shaped to hold `num_timesteps` snapshots of `halo`. Write each step
with [`write_haloarray_timestep!`](@ref); `close` the returned file when done.
Under MPI the file is opened collectively across the array's communicator.
"""
function create_haloarray_output_file(filename::String, dataset_name::String,
                                      halo::AbstractHaloArray, num_timesteps::Int)
    comm = _hdf5_comm(halo)
    mode = isfile(filename) ? "r+" : "w"
    fid = _hdf5_open(filename, mode, comm)

    dset = haskey(fid, dataset_name) ?
            fid[dataset_name] :
            create_fixedsize_dataset_from_haloarray(fid, dataset_name, halo, num_timesteps)

    return fid, dset
end

function create_haloarray_output_file(filename::String, dataset_name::String,
                                      halo::MaybeHaloArray, num_timesteps::Int)
    isactive(halo) || return nothing, nothing
    return create_haloarray_output_file(filename, dataset_name, getdata(halo), num_timesteps)
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
        nothing
    end
end

function save_array_hdf5(filename::String, data; dataset::String="dataset")
    return _hdf5_save_snapshot(filename, data; dataset=dataset)
end

"""
    gather_and_save_haloarray(filename, halo; root=0)

Gather `halo` onto `root` and write the assembled global array to the HDF5 file
`filename` as a single snapshot. Distributed arrays write only on `root`; serial
arrays write directly. See [`gather_and_append_haloarray!`](@ref) to append
successive timesteps to one dataset instead.
"""
function gather_and_save_haloarray(filename::String, halo::HaloArray; root::Int=0)
    comm = halo.topology.cart_comm
    gathered = _hdf5_gather_snapshot(halo; root=root)
    if MPI.Comm_rank(comm) == root
        save_array_hdf5(filename, gathered)
    end
    MPI.Barrier(comm)
    return nothing
end

function gather_and_save_haloarray(filename::String, halo::AbstractSerialHaloArray; root::Int=0)
    gathered = _hdf5_gather_snapshot(halo; root=root)
    save_array_hdf5(filename, gathered)
    return nothing
end

function gather_and_save_haloarray(filename::String, halo::ArrayOfHaloArray; root::Int=0)
    comm = _hdf5_comm(halo)
    gathered = _hdf5_gather_snapshot(halo; root=root)
    if comm === nothing || MPI.Comm_rank(comm) == root
        save_array_hdf5(filename, gathered)
    end
    comm === nothing || MPI.Barrier(comm)
    return nothing
end

function gather_and_save_haloarray(filename::String, halo::MultiHaloArray; root::Int=0)
    comm = _hdf5_comm(halo)
    gathered = _hdf5_gather_snapshot(halo; root=root)
    if comm === nothing || MPI.Comm_rank(comm) == root
        _hdf5_save_snapshot(filename, gathered)
    end
    comm === nothing || MPI.Barrier(comm)
    return nothing
end

function gather_and_save_haloarray(filename::String, halo::MaybeHaloArray; root::Int=0)
    isactive(halo) || return nothing
    return gather_and_save_haloarray(filename, getdata(halo); root=root)
end

"""
    gather_and_append_haloarray!(filename::String, dataset::String, halo::HaloArray; root::Int=0)

Gather `halo` on `root` and append it as a new timestep in `dataset`.
"""
function gather_and_append_haloarray!(filename::String, dataset::String, halo::HaloArray; root::Int=0)
    comm = halo.topology.cart_comm
    rank = MPI.Comm_rank(comm)
    gathered = _hdf5_gather_snapshot(halo; root=root)
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
                inds = (new_dims[1], ntuple(_ -> Colon(), ndims(gathered))...)
                dset[inds...] = gathered
            else
                global_dims = size(gathered)
                dspace = HDF5.dataspace((1, global_dims...); max_dims=(-1, global_dims...))
                dset = HDF5.create_dataset(file, dataset, eltype(gathered), dspace; chunk=(1, global_dims...))
                inds = (1, ntuple(_ -> Colon(), ndims(gathered))...)
                dset[inds...] = gathered
            end
        end
    end

    MPI.Barrier(comm)
    return nothing
end

function gather_and_append_haloarray!(filename::String, dataset::String, halo::AbstractHaloArray; root::Int=0)
    append_haloarray_to_file!(filename, dataset, halo)
    return nothing
end

function gather_and_append_haloarray!(filename::String, dataset::String, halo::MaybeHaloArray; root::Int=0)
    isactive(halo) || return nothing
    return gather_and_append_haloarray!(filename, dataset, getdata(halo); root=root)
end
