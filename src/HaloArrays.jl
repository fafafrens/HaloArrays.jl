module HaloArrays

using MPI
using HDF5
using LinearAlgebra
using OhMyThreads: tforeach, tmapreduce
using StaticArrays

# Include library files
include("abstract_haloarray.jl")
include("thread_backend.jl")
include("cartesian_topology.jl")
include("haloarray.jl")
include("local_haloarray.jl")
include("threaded_haloarray.jl")
include("field_collection.jl")
include("ArrayOfHaloArray.jl")
include("multihaloarray.jl")
include("face_ranges.jl")
include("cell_ranges.jl")
include("face_kernel_regions.jl")
include("cell_kernel_regions.jl")
include("maybehaloarray.jl")
include("interior_broadcast.jl")
include("interior_broadcast_marray.jl")
include("interior_broadcast_maybe.jl")
include("boundary.jl")
include("halo_exchange.jl")
include("gather.jl")
include("reduction.jl")
include("vector_space.jl")
include("mpi_support.jl")
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
    ThreadBackend,
    OhMyThreadsBackend,
    SerialBackend,
    PolyesterBackend,
    LocalHaloArray,
    ThreadedHaloArray,
    ThreadedMultiHaloArray,
    MultiHaloArray,
    LocalMultiHaloArray,
    ArrayOfHaloArray,
    MaybeHaloArray,
    AbstractCartesianTopology,
    CartesianTopology,
    ThreadedCartesianTopology,
    AbstractBoundaryCondition,
    Reflecting,
    Antireflecting,
    Repeating,
    Periodic,
    NoBoundaryCondition,
    Side,
    Dim

# Basic API
export interior_view,
    interior_range,
    storage_size,
    interior_size,
    interior_axes,
    is_root,
    halo_backend,
    thread_backend,
    tile_foreach,
    tile_mapreduce,
    field_shape,
    halo_width,
    get_comm,
    global_size,
    tile_size,
    tile_count,
    tile_parent,
    field_storages,
    tile_coordinates,
    neighbor_tile_id,
    FaceRanges,
    accumulate_flux_divergence!,
    CellRanges,
    FaceKernelRegion,
    ColoredFaceKernelRegion,
    CellKernelRegion,
    ColoredCellKernelRegion,
    get_left_face,
    get_internal_face,
    get_right_face,
    get_colored_left_face,
    get_colored_internal_face,
    get_colored_right_face,
    get_left_face_region,
    get_internal_face_region,
    get_right_face_region,
    get_colored_left_face_region,
    get_colored_internal_face_region,
    get_colored_right_face_region,
    get_interior_cells,
    get_colored_interior_cell_ranges,
    get_interior_cell_region,
    get_colored_interior_cell_region,
    cell_index,
    is_cell_index_inbounds,
    get_unit_vector,
    interior_to_global_index,
    fill_from_global_indices!,
    fill_from_local_indices!

# Boundary conditions
export boundary_condition!,
    boundary_condition_threads!,
    AbstractCoupledBoundaryCondition,
    apply_coupled_bc!,
    is_physical_boundary,
    eachfield

# Halo exchange
export halo_exchange!,
    halo_exchange_threads!,
    start_halo_exchange!,
    finish_halo_exchange!,
    synchronize_halo!,
    synchronize_halo_threads!,
    get_send_view,
    get_recv_view

# Reductions and gather
export mapreduce_haloarray_dims,
    gather_haloarray

# Maybe helpers
export isactive

# HDF5 I/O helpers
export append_haloarray_to_file!,
    write_haloarray_timestep!,
    create_haloarray_output_file,
    gather_and_save_haloarray,
    gather_and_append_haloarray!

# Internal-but-stable helpers remain accessible qualified, e.g.
# HaloArrays.to_bc, HaloArrays.unwrap, HaloArrays.internal_face_range,
# HaloArrays.save_array_hdf5, HaloArrays.append_haloarray!.

end
