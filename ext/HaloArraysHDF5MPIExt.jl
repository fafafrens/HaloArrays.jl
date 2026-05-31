module HaloArraysHDF5MPIExt

# Loaded when BOTH HDF5 and MPI are available.
# Provides the MPI-aware HDF5 helpers: parallel file open,
# gather-and-save for HaloArray, and save_array_hdf5 with a comm argument.

using HDF5
using MPI
import HaloArrays
using HaloArrays:
    HaloArray, ArrayOfHaloArray, MultiHaloArray, MaybeHaloArray,
    AbstractHaloArray, AbstractSerialHaloArray,
    interior_view, field_shape, isactive, getdata,
    gather_haloarray, _hdf5_comm

# ---- parallel file open -----------------------------------------------

function _hdf5_open_mpi(filename::String, mode::String, comm)
    comm === nothing ? h5open(filename, mode) : h5open(filename, mode, comm, MPI.Info())
end

function _hdf5_open_mpi(f::Function, filename::String, mode::String, comm)
    fid = _hdf5_open_mpi(filename, mode, comm)
    try
        return f(fid)
    finally
        close(fid)
    end
end

# ---- save_array_hdf5 with MPI.Comm ------------------------------------

function HaloArrays.save_array_hdf5(filename::String, data, comm::MPI.Comm; root::Int=0)
    if MPI.Comm_rank(comm) == root
        h5open(filename*".h5", "w") do file
            write(file, "dataset", data)
        end
    end
end

# ---- gather snapshot (MPI path) ---------------------------------------

function _hdf5_gather_snapshot_mpi(halo::HaloArray; root::Int=0)
    gather_haloarray(halo; root=root)
end

function _hdf5_gather_snapshot_mpi(halo::ArrayOfHaloArray; root::Int=0)
    comm = _hdf5_comm(halo)
    comm === nothing && return _hdf5_snapshot_serial(halo)
    rank = MPI.Comm_rank(comm)
    data = nothing
    for I in CartesianIndices(parent(halo))
        fd = gather_haloarray(parent(halo)[I]; root=root)
        if rank == root
            if data === nothing
                data = Array{eltype(halo)}(undef, (field_shape(halo)..., size(fd)...))
            end
            inds = (Tuple(I)..., ntuple(_ -> Colon(), ndims(fd))...)
            data[inds...] .= fd
        end
    end
    return data
end

function _hdf5_gather_snapshot_mpi(halo::MultiHaloArray; root::Int=0)
    fields = map(values(halo.arrays)) do field
        _hdf5_gather_snapshot_mpi(field; root=root)
    end
    NamedTuple{keys(halo.arrays)}(fields)
end

# ---- gather_and_save_haloarray (MPI variants) -------------------------

function HaloArrays.gather_and_save_haloarray(filename::String, halo::HaloArray;
        root::Int=0)
    comm    = halo.topology.cart_comm
    gathered = _hdf5_gather_snapshot_mpi(halo; root=root)
    if MPI.Comm_rank(comm) == root
        HaloArrays.save_array_hdf5(filename, gathered)
    end
    MPI.Barrier(comm)
end

function HaloArrays.gather_and_save_haloarray(filename::String, halo::ArrayOfHaloArray;
        root::Int=0)
    comm    = _hdf5_comm(halo)
    gathered = _hdf5_gather_snapshot_mpi(halo; root=root)
    if comm === nothing || MPI.Comm_rank(comm) == root
        HaloArrays.save_array_hdf5(filename, gathered)
    end
    comm === nothing || MPI.Barrier(comm)
end

function HaloArrays.gather_and_save_haloarray(filename::String, halo::MultiHaloArray;
        root::Int=0)
    comm    = _hdf5_comm(halo)
    gathered = _hdf5_gather_snapshot_mpi(halo; root=root)
    if comm === nothing || MPI.Comm_rank(comm) == root
        h5open(filename*".h5", "w") do file
            for (fn, fd) in pairs(gathered)
                write(file, String(fn), fd)
            end
        end
    end
    comm === nothing || MPI.Barrier(comm)
end

# ---- gather_and_append_haloarray! (HaloArray variant) -----------------

function HaloArrays.gather_and_append_haloarray!(filename::String, dataset::String,
        halo::HaloArray; root::Int=0)
    comm    = halo.topology.cart_comm
    rank    = MPI.Comm_rank(comm)
    gathered = _hdf5_gather_snapshot_mpi(halo; root=root)
    filename_h5 = filename * ".h5"
    if rank == root
        isfile(filename_h5) || h5open(filename_h5, "w") do _ end
        h5open(filename_h5, "r+") do file
            if haskey(file, dataset)
                dset = file[dataset]
                curr_dims = size(dset)
                new_dims  = (curr_dims[1]+1, curr_dims[2:end]...)
                HDF5.set_extent_dims(dset, new_dims)
                inds = (new_dims[1], ntuple(_ -> Colon(), ndims(gathered))...)
                dset[inds...] = gathered
            else
                gdims  = size(gathered)
                dspace = HDF5.dataspace((1, gdims...); max_dims=(-1, gdims...))
                dset   = HDF5.create_dataset(file, dataset, eltype(gathered), dspace;
                             chunk=(1, gdims...))
                dset[(1, ntuple(_ -> Colon(), ndims(gathered))...)...] = gathered
            end
        end
    end
    MPI.Barrier(comm)
end

end # module HaloArraysHDF5MPIExt
