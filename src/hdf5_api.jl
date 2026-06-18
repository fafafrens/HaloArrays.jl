# HDF5 I/O — public API stubs. The methods live in the `HaloArraysHDF5Ext` package
# extension, which loads only when the user has `using HDF5`. This keeps HDF5 (and
# its MPI-built JLLs, which clash with a system CUDA-aware MPI) out of the core
# dependency tree. Without HDF5 loaded these throw a MethodError, by design.

"""
    append_haloarray_to_file!(file, dataset_name, halo)

Append the interior data of `halo` to an HDF5 file (the `.h5` suffix is added to
`file`), creating it on first write and growing an extendable `dataset_name`
dataset by one slab per call — the idiom for streaming successive timesteps to a
single dataset. For a distributed `HaloArray` each rank writes its own block
collectively (parallel HDF5). See also [`gather_and_save_haloarray`](@ref) for
gathering to a single dense array on the root rank instead.

Requires `using HDF5` (provided by the `HaloArraysHDF5Ext` extension).
"""
function append_haloarray_to_file! end

"""
    write_haloarray_timestep!(dset, halo, timestep)

Write the interior (ghost-free) data of `halo` into the time-resolved dataset
`dset` (from [`create_haloarray_output_file`](@ref)) at index `timestep`
(0-based). Under MPI each rank writes its own subdomain collectively.

Requires `using HDF5`.
"""
function write_haloarray_timestep! end

"""
    create_haloarray_output_file(filename, dataset_name, halo, num_timesteps) -> (file, dataset)

Open (or create) the HDF5 file `filename` and return it with a fixed-size
`dataset` shaped to hold `num_timesteps` snapshots of `halo`. Write each step
with [`write_haloarray_timestep!`](@ref); `close` the returned file when done.
Under MPI the file is opened collectively across the array's communicator.

Requires `using HDF5`.
"""
function create_haloarray_output_file end

"""
    gather_and_save_haloarray(filename, halo; root=0)

Gather `halo` onto `root` and write the assembled global array to the HDF5 file
`filename` as a single snapshot. Distributed arrays write only on `root`; serial
arrays write directly. See [`gather_and_append_haloarray!`](@ref) to append
successive timesteps to one dataset instead.

Requires `using HDF5`.
"""
function gather_and_save_haloarray end

"""
    gather_and_append_haloarray!(filename, dataset, halo; root=0)

Gather `halo` on `root` and append it as a new timestep in `dataset`.

Requires `using HDF5`.
"""
function gather_and_append_haloarray! end

# Non-exported entry points (called as `HaloArrays.<name>`); methods in the extension.
"""
    save_array_hdf5(filename, data[, comm; root=0])

Write a plain `data` array to `filename.h5`. With an MPI `comm`, only `root` writes.
Requires `using HDF5`.
"""
function save_array_hdf5 end

function append_haloarray! end
function create_dataset_from_haloarray end
function create_fixedsize_dataset_from_haloarray end
