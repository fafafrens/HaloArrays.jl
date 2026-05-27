__precompile__()
module HaloArrays

using MPI
using HDF5
using LinearAlgebra
using OhMyThreads: tforeach, tmap, tmapreduce
using StaticArrays

# Include library files
include("abstract_haloarray.jl")
include("cartesian_topology.jl")
include("haloarray.jl")
include("local_haloarray.jl")
include("threaded_haloarray.jl")
include("ArrayOfHaloArray.jl")
include("threaded_multihaloarray.jl")
include("multihaloarray.jl")
include("local_multihaloarray.jl")
include("face_ranges.jl")
include("maybe.jl")
include("maybehaloarray.jl")
include("interior_broadcast.jl")
include("interior_broadcast_marray.jl")
include("interior_broadcast_maybe.jl")
include("boundary.jl")
include("halo_exchange.jl")
include("flux_contribution_exchange.jl")
include("gather.jl")
include("reduction.jl")
include("save_hdf5.jl")

# Core types
export HaloArray,
    AbstractHaloArray,
    AbstractSingleHaloArray,
    AbstractDistributedHaloArray,
    AbstractSerialHaloArray,
    AbstractHaloCollection,
    AbstractHaloBackend,
    MPIHaloBackend,
    LocalHaloBackend,
    ThreadedHaloBackend,
    LocalHaloArray,
    ThreadedHaloArray,
    ThreadedMultiHaloArray,
    MultiHaloArray,
    LocalMultiHaloArray,
    ArrayOfHaloArray,
    MaybeHaloArray,
    CartesianTopology,
    ThreadedCartesianTopology,
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
    storage_size,
    interior_size,
    owned_size,
    owned_axes,
    is_root,
    halo_backend,
    field_shape,
    halo_width,
    get_comm,
    global_size,
    versors,
    tile_size,
    tile_count,
    tile_parent,
    tile_coordinates,
    neighbor_tile_id,
    face_offset,
    left_face_range,
    internal_face_range,
    right_face_range,
    FaceRanges,
    get_left_face,
    get_internal_face,
    get_right_face,
    get_unit_vector,
    owned_to_global_index,
    global_to_storage_index,
    fill_interior,
    fill_from_global_indices!,
    fill_from_local_indices!

# Boundary conditions
export boundary_condition!,
    boundary_condition_threads!,
    normalize_boundary_condition,
    to_bc

# Halo exchange
export halo_exchange!,
    halo_exchange_threads!,
    start_halo_exchange!,
    finish_halo_exchange!,
    synchronize_halo!,
    synchronize_halo_threads!,
    synchronize_flux_contributions!,
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
