using MPI
using HaloArrays

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

# Setup full 3D Cartesian communicator
tot_dimension=2
dims = MPI.Dims_create(nprocs, ntuple(_ -> 0, tot_dimension))  # 3D communicator
periods = ntuple(_ -> true, tot_dimension)  # periodic in all dims
cart_comm = MPI.Cart_create(comm, dims, periodic=periods)
coords = MPI.Cart_coords(cart_comm, rank)

# Dimension to reduce (1-based indexing)
dim_to_reduce = 2  # z-dimension

# Coordinate along reduced dimension for this rank
reduce_coord = coords[dim_to_reduce]

MPI.Barrier(comm)
for r in 0:nprocs-1
    MPI.Barrier(comm)
    if r == rank
        println("Rank $rank")
        println("  Original dims     = $dims")
        println("  Original coords   = $coords")
        println("  Reduced coords    = $reduce_coord")
        println("--------------------------------------------------")
    end
end

MPI.Barrier(comm)

# Create communicator grouping all ranks with the same coordinate along dim_to_reduce
#color = reduce_coord
#sub_comm = MPI.Comm_split(cart_comm, color, rank)

# Identify roots for each slice (coord == 0 along reduced dim)
is_root = reduce_coord == 0

# For roots: build a new 2D Cartesian communicator after removing dim_to_reduce
root_color = is_root ? 0 : nothing # MPI.UNDEFINED excludes others
root_comm = MPI.Comm_split(cart_comm, root_color, rank)

new_coords = nothing
if is_root
    # Remove the reduced dimension from dims and periods to get new dims
    new_dims = Tuple(deleteat!(collect(dims), dim_to_reduce))
    new_periods = Tuple(deleteat!(collect(periods), dim_to_reduce))

    # Create 2D Cartesian communicator for roots only
    new_cart = MPI.Cart_create(root_comm, new_dims, periodic=new_periods)
    new_coords = MPI.Cart_coords(new_cart, MPI.Comm_rank(new_cart))
end

MPI.Barrier(comm)

# Print summary for all ranks (collect info to rank 0 for ordered print)
summary = MPI.gather((rank, coords, is_root, new_coords), comm)

if rank == 0
    println("Summary of Cartesian Split:")
    for (r, c, root, ncoords) in summary
        println("Rank $r: coords = $c, root = $root, new_coords = $(ncoords === nothing ? "-" : ncoords)")
    end
end

MPI.Finalize()