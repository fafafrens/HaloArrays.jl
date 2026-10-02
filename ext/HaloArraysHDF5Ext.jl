module HaloArraysHDF5Ext

# HDF5 I/O for HaloArrays. Loaded only when the user has `using HDF5`. The public
# entry point is declared (with its docstring) as a stub in src/hdf5_api.jl; this
# extension provides the methods, so `using HaloArrays` alone does not pull in
# HDF5 (and its MPI-built JLLs, which clash with a system CUDA-aware MPI).
#
# One operation: append the interior of a halo array as the next step of a
# time-series dataset (time on the leading axis). A single array or an
# ArrayOfHaloArray is one dataset (field axes first for the latter); a
# MultiHaloArray is a group with one such dataset per field. Distributed arrays
# write their own block collectively; serial arrays write the assembled interior.

using HaloArrays
using HDF5
using MPI

import HaloArrays:
    AbstractHaloArray, AbstractSingleHaloArray, AbstractSerialHaloArray,
    AbstractHaloCollection, HaloArray, MultiHaloArray, ArrayOfHaloArray,
    MaybeHaloArray, field_shape, _container_shape, interior_size, interior_view,
    communicator, _first_field, gather_haloarray, is_active, getdata,
    append_haloarray!

const _Parent = Union{HDF5.File,HDF5.Group}

# ---- geometry helpers ---------------------------------------------------------

# An ArrayOfHaloArray is one dataset with the field axes first; a nested one
# recurses (the inner field axes follow the outer container's). A
# MultiHaloArray inside it has named fields and cannot be one dataset.
@inline _dataset_dims(halo::AbstractSingleHaloArray) = size(halo)
@inline _dataset_dims(halo::ArrayOfHaloArray) =
    (_container_shape(halo)..., _dataset_dims(first(parent(halo)))...)
_dataset_dims(::MultiHaloArray) = throw(ArgumentError(
    "a MultiHaloArray nested in an ArrayOfHaloArray cannot be written as one dataset; nest the other way round"))

@inline _chunk_dims(halo::HaloArray) = interior_size(halo)
@inline _chunk_dims(halo::AbstractSerialHaloArray) = _dataset_dims(halo)
@inline _chunk_dims(halo::ArrayOfHaloArray) =
    (_container_shape(halo)..., _chunk_dims(first(parent(halo)))...)

@inline _comm(halo::HaloArray) = communicator(halo)
@inline _comm(::AbstractSerialHaloArray) = nothing
@inline _comm(halo::AbstractHaloCollection) = _comm(_first_field(halo))
@inline _comm(halo::MaybeHaloArray) = _comm(getdata(halo))

_field_name(name) = string(name)

# The block a rank writes and where it lands in the global spatial axes.
@inline _block(halo::HaloArray) = interior_view(halo)
@inline _block(halo::AbstractSerialHaloArray) = gather_haloarray(halo)
@inline function _block_slices(halo::HaloArray)
    dims = interior_size(halo)
    coords = halo.topology.cart_coords
    return ntuple(d -> (coords[d] * dims[d] + 1):((coords[d] + 1) * dims[d]), length(dims))
end
@inline _block_slices(halo::AbstractSerialHaloArray) = ntuple(_ -> Colon(), ndims(halo))

# ---- dataset creation and validation ------------------------------------------

# Reusing an existing dataset must match the halo array that will be written: a
# SMALLER array would silently write a partial slab into each appended step (the
# rest stays stale); a larger one errors late inside HDF5.
function _assert_appendable(obj, ::Type{T}, global_dims, name::String) where {T}
    obj isa HDF5.Dataset || throw(ArgumentError(
        "HDF5: \"$name\" already exists as $(typeof(obj)), not an appendable dataset."))
    eltype(obj) === T || throw(ArgumentError(
        "HDF5: existing dataset \"$name\" has eltype $(eltype(obj)), expected $T."))
    dims = size(obj)
    (length(dims) == length(global_dims) + 1 && dims[2:end] == global_dims) ||
        throw(DimensionMismatch(
            "HDF5: existing dataset \"$name\" has size $dims but appending this halo " *
            "array needs (nsteps, $(join(global_dims, ", "))). Refusing to append to a " *
            "mismatched dataset — use a new file/name or delete the old dataset."))
    _, maxdims = HDF5.get_extent_dims(obj)
    maxdims[1] == -1 || throw(ArgumentError(
        "HDF5: existing dataset \"$name\" has a fixed time axis (max $(maxdims[1]) " *
        "step(s)); it was not created for appending — use a new file/name or delete it."))
    return obj
end

function _dataset(parent::_Parent, name::String, halo)
    T = eltype(halo)
    global_dims = _dataset_dims(halo)
    haskey(parent, name) && return _assert_appendable(parent[name], T, global_dims, name)
    dspace = dataspace((0, global_dims...); max_dims=(-1, global_dims...))
    return HDF5.create_dataset(parent, name, T, dspace; chunk=(1, _chunk_dims(halo)...))
end

function _group(parent::_Parent, name::String)
    return haskey(parent, name) ? HDF5.open_group(parent, name) : HDF5.create_group(parent, name)
end

# ---- writing one step -----------------------------------------------------------

function _write_step!(dset, halo::AbstractSingleHaloArray, step::Int)
    dset[step, _block_slices(halo)...] = _block(halo)
    return nothing
end

# Each leaf writes its own block under its field-index prefix; a nested
# ArrayOfHaloArray extends the prefix with its container index.
_write_field!(dset, step::Int, prefix::Tuple, field::AbstractSingleHaloArray) =
    (dset[step, prefix..., _block_slices(field)...] = _block(field); nothing)
function _write_field!(dset, step::Int, prefix::Tuple, c::ArrayOfHaloArray)
    fields = parent(c)
    for I in CartesianIndices(fields)
        _write_field!(dset, step, (prefix..., Tuple(I)...), fields[I])
    end
    return nothing
end
_write_step!(dset, halo::ArrayOfHaloArray, step::Int) = _write_field!(dset, step, (), halo)

function _append!(parent::_Parent, name::String, halo::Union{AbstractSingleHaloArray,ArrayOfHaloArray})
    dset = _dataset(parent, name, halo)
    step = size(dset, 1) + 1
    HDF5.set_extent_dims(dset, (step, size(dset)[2:end]...))
    _write_step!(dset, halo, step)
    return dset
end

function _append!(parent::_Parent, name::String, halo::MultiHaloArray)
    group = _group(parent, name)
    for (field_name, field) in pairs(halo.arrays)
        _append!(group, _field_name(field_name), field)
    end
    return group
end

# ---- public methods -------------------------------------------------------------

append_haloarray!(parent::_Parent, name::AbstractString, halo::AbstractHaloArray) =
    _append!(parent, String(name), halo)

function append_haloarray!(parent::_Parent, name::AbstractString, halo::MaybeHaloArray)
    is_active(halo) || return nothing
    return append_haloarray!(parent, name, getdata(halo))
end

function append_haloarray!(filename::AbstractString, name::AbstractString, halo::AbstractHaloArray)
    comm = _comm(halo)
    mode = isfile(filename) ? "r+" : "w"
    fid = comm === nothing ? h5open(filename, mode) : h5open(filename, mode, comm, MPI.Info())
    try
        _append!(fid, String(name), halo)
    finally
        close(fid)
    end
    return nothing
end

function append_haloarray!(filename::AbstractString, name::AbstractString, halo::MaybeHaloArray)
    is_active(halo) || return nothing
    return append_haloarray!(filename, name, getdata(halo))
end

end # module HaloArraysHDF5Ext
