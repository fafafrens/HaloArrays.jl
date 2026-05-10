__precompile__()
module HaloArrays

using MPI
using HDF5
using StaticArrays

# Include library files
include("cartesian_topology.jl")
include("haloarray.jl")
include("multihaloarray.jl")
include("meybe.jl")
include("meybehaloarray.jl")
include("interior_broadcast.jl")
include("interior_broadcast_marray.jl")
include("interior_broadcast_maybe.jl")
include("boundary.jl")
include("halo_exchange.jl")
include("gather.jl")
include("reduction.jl")
include("save_hdf5.jl")

# Core types
export HaloArray,
    MultiHaloArray,
    MaybeHaloArray,
    CartesianTopology,
    AbstractBoundaryCondition,
    Reflecting,
    Antireflecting,
    Repeating,
    Periodic,
    Side,
    Dim

# Basic API
export interior_view,
    interior_range,
    full_size,
    interior_size,
    halo_width,
    get_comm,
    global_size,
    versors,
    local_to_global_index,
    global_to_local_index,
    fill_interior,
    fill_from_global_indices!,
    fill_from_local_indices!

# Boundary conditions
export boundary_condition!,
    register_bc,
    normalize_boundary_condition,
    to_bc

# Halo exchange
export halo_exchange!,
    halo_exchange_wait!,
    halo_exchange_waitall!,
    halo_exchange_waitall_unsafe!,
    halo_exchange_async!,
    halo_exchange_async_unsafe!,
    start_halo_exchange_async!,
    halo_exchange_async_wait!,
    halo_exchange_async_wait_unsafe!,
    start_halo_exchange_async_unsafe!,
    end_halo_exchange_async_wait_unsafe!,
    get_send_view,
    get_recv_view

# Reductions and gather
export mapreduce_haloarray_dims,
    mapreduce_mhaloarray_dims,
    gather_haloarray

# Maybe helpers
export isactive,
    getdata,
    unwrap,
    active,
    inactive,
    unsafe_get

# HDF5 I/O helpers
export create_dataset_from_haloarray,
    append_haloarray!,
    append_haloarray_to_file!,
    create_fixedsize_dataset_from_haloarray,
    write_haloarray_timestep!,
    create_haloarray_output_file,
    save_array_hdf5,
    gather_and_save_haloarray,
    gather_and_append_haloarray!

end
