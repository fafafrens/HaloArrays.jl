module HaloArraysHDF5Ext

using HDF5
import HaloArrays
using HaloArrays:
    HaloArray, LocalHaloArray, ThreadedHaloArray, ArrayOfHaloArray, MultiHaloArray,
    MaybeHaloArray, AbstractSingleHaloArray, AbstractSerialHaloArray, AbstractHaloArray,
    interior_view, global_size, field_shape, tile_count, tile_size, tile_coordinates,
    tile_parent, isactive, getdata, _hdf5_comm

# ---- helpers ----------------------------------------------------------

@inline HaloArrays._hdf5_dataset_dims(halo::AbstractSingleHaloArray) = global_size(halo)
@inline HaloArrays._hdf5_dataset_dims(halo::ArrayOfHaloArray) =
    (field_shape(halo)..., HaloArrays._hdf5_dataset_dims(first(parent(halo)))...)

@inline HaloArrays._hdf5_chunk_dims(halo::HaloArray) = HaloArrays.interior_size(halo)
@inline HaloArrays._hdf5_chunk_dims(halo::AbstractSerialHaloArray) =
    HaloArrays._hdf5_dataset_dims(halo)
@inline HaloArrays._hdf5_chunk_dims(halo::ArrayOfHaloArray) =
    (field_shape(halo)..., HaloArrays._hdf5_chunk_dims(first(parent(halo)))...)

@inline HaloArrays._hdf5_comm(::AbstractSerialHaloArray) = nothing
@inline HaloArrays._hdf5_comm(halo::ArrayOfHaloArray)    = HaloArrays._hdf5_comm(first(parent(halo)))
@inline HaloArrays._hdf5_comm(halo::MultiHaloArray)      = HaloArrays._hdf5_comm(first(values(halo.arrays)))

@inline _hdf5_field_name(name::Symbol) = String(name)
@inline _hdf5_field_name(name) = string(name)

function _hdf5_open_or_create_group(parent, name::String)
    haskey(parent, name) ? HDF5.open_group(parent, name) : HDF5.create_group(parent, name)
end

function _hdf5_open(filename::String, mode::String, ::Nothing)
    h5open(filename, mode)
end

function _hdf5_open(f::Function, filename::String, mode::String, comm::Nothing)
    fid = _hdf5_open(filename, mode, comm)
    try
        return f(fid)
    finally
        close(fid)
    end
end

# ---- snapshots --------------------------------------------------------

function _hdf5_snapshot(halo::LocalHaloArray)
    Array(interior_view(halo))
end

function _hdf5_snapshot(halo::ThreadedHaloArray{T,N}) where {T,N}
    data = Array{T}(undef, global_size(halo))
    owned_ts = tile_size(halo)
    for tile_id in 1:tile_count(halo)
        coords = tile_coordinates(halo, tile_id)
        inds = ntuple(Val(N)) do d
            first_owned = (coords[d]-1)*owned_ts[d] + 1
            first_owned:(coords[d]*owned_ts[d])
        end
        data[inds...] .= interior_view(halo, tile_id)
    end
    return data
end

function _hdf5_snapshot(halo::ArrayOfHaloArray)
    first(parent(halo)) isa HaloArray &&
        throw(ArgumentError("snapshot for MPI ArrayOfHaloArray not supported; use append_haloarray! or write_haloarray_timestep!"))
    data = Array{eltype(halo)}(undef, HaloArrays._hdf5_dataset_dims(halo))
    for I in CartesianIndices(parent(halo))
        fd = _hdf5_snapshot(parent(halo)[I])
        inds = (Tuple(I)..., ntuple(_ -> Colon(), ndims(fd))...)
        data[inds...] .= fd
    end
    return data
end

function _hdf5_snapshot(halo::MultiHaloArray)
    HaloArrays._hdf5_comm(halo) === nothing ||
        throw(ArgumentError("snapshot for MPI MultiHaloArray not supported; use gather_and_save_haloarray"))
    fields = map(_hdf5_snapshot, values(halo.arrays))
    NamedTuple{keys(halo.arrays)}(fields)
end

# ---- gather snapshot (serial path) -----------------------------------

_hdf5_gather_snapshot(halo::AbstractSerialHaloArray; root::Int=0) = _hdf5_snapshot(halo)

function _hdf5_gather_snapshot(halo::ArrayOfHaloArray; root::Int=0)
    comm = HaloArrays._hdf5_comm(halo)
    comm === nothing || error("MPI HDF5 gather requires HaloArraysHDF5MPIExt (load both HDF5 and MPI)")
    _hdf5_snapshot(halo)
end

function _hdf5_gather_snapshot(halo::MultiHaloArray; root::Int=0)
    comm = HaloArrays._hdf5_comm(halo)
    comm === nothing || error("MPI HDF5 gather requires HaloArraysHDF5MPIExt (load both HDF5 and MPI)")
    _hdf5_snapshot(halo)
end

# ---- write helpers ----------------------------------------------------

function _hdf5_write_snapshot!(parent_node, name::String, data::AbstractArray)
    write(parent_node, name, data)
end

function _hdf5_write_snapshot!(parent_node, name::String, data::NamedTuple)
    group = HDF5.create_group(parent_node, name)
    for (fn, fd) in pairs(data)
        _hdf5_write_snapshot!(group, _hdf5_field_name(fn), fd)
    end
end

function _hdf5_save_snapshot(filename::String, data; dataset::String="dataset")
    h5open(filename*".h5", "w") do file
        _hdf5_write_snapshot!(file, dataset, data)
    end
end

function _hdf5_write_timestep!(dset, halo::AbstractSerialHaloArray, time_index::Integer)
    data = _hdf5_snapshot(halo)
    slices = (time_index, ntuple(_ -> Colon(), ndims(data))...)
    dset[slices...] = data
end

function _hdf5_write_field_timestep!(dset, field::AbstractSerialHaloArray,
        time_index::Integer, field_index)
    fd = _hdf5_snapshot(field)
    slices = (time_index, field_index..., ntuple(_ -> Colon(), ndims(fd))...)
    dset[slices...] = fd
end

function _hdf5_write_timestep!(dset, halo::ArrayOfHaloArray, time_index::Integer)
    for I in CartesianIndices(parent(halo))
        _hdf5_write_field_timestep!(dset, parent(halo)[I], time_index, Tuple(I))
    end
end

function _hdf5_write_timestep!(group::HDF5.Group, halo::MultiHaloArray, time_index::Integer)
    for (field_name, field) in pairs(halo.arrays)
        child = group[_hdf5_field_name(field_name)]
        _hdf5_write_timestep!(child, field, time_index)
    end
end

# ---- dataset creation -------------------------------------------------

function _create_hdf5_dataset_from_haloarray(g, name::String, halo)
    T = eltype(halo)
    global_dims  = HaloArrays._hdf5_dataset_dims(halo)
    initial_size = (0, global_dims...)
    max_size     = (-1, global_dims...)
    chunk        = (1, HaloArrays._hdf5_chunk_dims(halo)...)
    dspace = dataspace(initial_size; max_dims=max_size)
    haskey(g, name) && return HDF5.open_dataset(g, name)
    HDF5.create_dataset(g, name, T, dspace; chunk=chunk)
end

function HaloArrays.create_dataset_from_haloarray(g, name::String,
        halo::AbstractSingleHaloArray)
    _create_hdf5_dataset_from_haloarray(g, name, halo)
end

function HaloArrays.create_dataset_from_haloarray(g, name::String, halo::ArrayOfHaloArray)
    _create_hdf5_dataset_from_haloarray(g, name, halo)
end

function HaloArrays.create_dataset_from_haloarray(g, name::String, halo::MultiHaloArray)
    group = _hdf5_open_or_create_group(g, name)
    for (fn, field) in pairs(halo.arrays)
        HaloArrays.create_dataset_from_haloarray(group, _hdf5_field_name(fn), field)
    end
    return group
end

function HaloArrays.create_dataset_from_haloarray(g, name::String, halo::MaybeHaloArray)
    isactive(halo) || return nothing
    HaloArrays.create_dataset_from_haloarray(g, name, getdata(halo))
end

# ---- append -----------------------------------------------------------

function _append_hdf5_dataset!(dset::HDF5.Dataset, halo)
    curr_dims = size(dset)
    new_dims  = (curr_dims[1]+1, curr_dims[2:end]...)
    HDF5.set_extent_dims(dset, new_dims)
    _hdf5_write_timestep!(dset, halo, new_dims[1])
end

HaloArrays.append_haloarray!(dset::HDF5.Dataset, halo::AbstractSingleHaloArray) =
    _append_hdf5_dataset!(dset, halo)

HaloArrays.append_haloarray!(dset::HDF5.Dataset, halo::ArrayOfHaloArray) =
    _append_hdf5_dataset!(dset, halo)

function HaloArrays.append_haloarray!(group::HDF5.Group, halo::MultiHaloArray)
    for (fn, field) in pairs(halo.arrays)
        cn = _hdf5_field_name(fn)
        child = haskey(group, cn) ? group[cn] :
            HaloArrays.create_dataset_from_haloarray(group, cn, field)
        HaloArrays.append_haloarray!(child, field)
    end
end

HaloArrays.append_haloarray!(dset::HDF5.Dataset, halo::MaybeHaloArray) =
    (isactive(halo) && HaloArrays.append_haloarray!(dset, getdata(halo)); nothing)

HaloArrays.append_haloarray!(group::HDF5.Group, halo::MaybeHaloArray) =
    (isactive(halo) && HaloArrays.append_haloarray!(group, getdata(halo)); nothing)

function HaloArrays.append_haloarray_to_file!(file::String, dataset_name::String,
        halo::AbstractHaloArray)
    file_h5 = file * ".h5"
    comm    = HaloArrays._hdf5_comm(halo)
    mode    = isfile(file_h5) ? "r+" : "w"
    _hdf5_open(file_h5, mode, comm) do fid
        dset = haskey(fid, dataset_name) ? fid[dataset_name] :
            HaloArrays.create_dataset_from_haloarray(fid, dataset_name, halo)
        HaloArrays.append_haloarray!(dset, halo)
    end
end

HaloArrays.append_haloarray_to_file!(file::String, dn::String, halo::MaybeHaloArray) =
    (isactive(halo) && HaloArrays.append_haloarray_to_file!(file, dn, getdata(halo)); nothing)

# ---- fixed-size datasets ----------------------------------------------

function _create_fixedsize_hdf5_dataset(g, name, halo, nt::Int)
    T = eltype(halo)
    gdims = HaloArrays._hdf5_dataset_dims(halo)
    sz    = (nt, gdims...)
    dspace = HDF5.dataspace(sz; max_dims=sz)
    HDF5.create_dataset(g, name, T, dspace; chunk=(1, HaloArrays._hdf5_chunk_dims(halo)...))
end

HaloArrays.create_fixedsize_dataset_from_haloarray(g, name::String,
        halo::AbstractSingleHaloArray, nt::Int) =
    _create_fixedsize_hdf5_dataset(g, name, halo, nt)

HaloArrays.create_fixedsize_dataset_from_haloarray(g, name::String,
        halo::ArrayOfHaloArray, nt::Int) =
    _create_fixedsize_hdf5_dataset(g, name, halo, nt)

function HaloArrays.create_fixedsize_dataset_from_haloarray(g, name::String,
        halo::MultiHaloArray, nt::Int)
    group = _hdf5_open_or_create_group(g, name)
    for (fn, field) in pairs(halo.arrays)
        HaloArrays.create_fixedsize_dataset_from_haloarray(group, _hdf5_field_name(fn), field, nt)
    end
    return group
end

HaloArrays.create_fixedsize_dataset_from_haloarray(g, name::String,
        halo::MaybeHaloArray, nt::Int) =
    isactive(halo) ?
        HaloArrays.create_fixedsize_dataset_from_haloarray(g, name, getdata(halo), nt) :
        nothing

HaloArrays.write_haloarray_timestep!(dset, halo::AbstractHaloArray, ts) =
    (_hdf5_write_timestep!(dset, halo, ts+1); nothing)

HaloArrays.write_haloarray_timestep!(dset, halo::MaybeHaloArray, ts) =
    (isactive(halo) && HaloArrays.write_haloarray_timestep!(dset, getdata(halo), ts); nothing)

function HaloArrays.create_haloarray_output_file(filename::String, dn::String,
        halo::AbstractHaloArray, nt::Int)
    comm = HaloArrays._hdf5_comm(halo)
    mode = isfile(filename) ? "r+" : "w"
    fid  = _hdf5_open(filename, mode, comm)
    dset = haskey(fid, dn) ? fid[dn] :
        HaloArrays.create_fixedsize_dataset_from_haloarray(fid, dn, halo, nt)
    return fid, dset
end

HaloArrays.create_haloarray_output_file(filename::String, dn::String,
        halo::MaybeHaloArray, nt::Int) =
    isactive(halo) ?
        HaloArrays.create_haloarray_output_file(filename, dn, getdata(halo), nt) :
        (nothing, nothing)

# ---- save / gather-and-save (serial path) ----------------------------

HaloArrays.save_array_hdf5(filename::String, data; dataset::String="dataset") =
    _hdf5_save_snapshot(filename, data; dataset=dataset)

function HaloArrays.gather_and_save_haloarray(filename::String,
        halo::AbstractSerialHaloArray; root::Int=0)
    HaloArrays.save_array_hdf5(filename, _hdf5_gather_snapshot(halo; root=root))
end

HaloArrays.gather_and_save_haloarray(filename::String, halo::MaybeHaloArray; root::Int=0) =
    (isactive(halo) && HaloArrays.gather_and_save_haloarray(filename, getdata(halo); root=root); nothing)

function HaloArrays.gather_and_append_haloarray!(filename::String, dn::String,
        halo::AbstractHaloArray; root::Int=0)
    HaloArrays.append_haloarray_to_file!(filename, dn, halo)
end

HaloArrays.gather_and_append_haloarray!(filename::String, dn::String,
        halo::MaybeHaloArray; root::Int=0) =
    (isactive(halo) && HaloArrays.gather_and_append_haloarray!(filename, dn, getdata(halo); root=root); nothing)

end # module HaloArraysHDF5Ext
