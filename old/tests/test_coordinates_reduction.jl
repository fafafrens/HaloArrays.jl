using MPI

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

# ------------------------------------------------------------
# 1. Create sub-Cartesian communicator excluding dim_to_reduce
# ------------------------------------------------------------
remain_dims = ntuple(i -> i != dim_to_reduce ? true : false, length(dims))
sub_cart = MPI.Cart_sub(cart_comm, remain_dims)

# ------------------------------------------------------------
# 2. Split root ranks from each sub_cart
# ------------------------------------------------------------
root_comm = MPI.COMM_NULL

reduced_coords = ()
key = -1
expected_coords = ()

local_rank = MPI.Comm_rank(sub_cart)
# Only rank 0 in each sub_cart goes into root_comm
color = (local_rank == 0) ? 0 : nothing
root_comm = MPI.Comm_split(cart_comm, color, rank)


if root_comm != MPI.COMM_NULL
    reduced_coords = Tuple(deleteat!(collect(coords), dim_to_reduce))
    remaining_dims = Tuple(deleteat!(collect(dims), dim_to_reduce))
    new_periods = Tuple(deleteat!(collect(periods), dim_to_reduce))
    
    # Get the actual size of root_comm to ensure compatibility
    root_comm_size = MPI.Comm_size(root_comm)

    
    # Verify that the remaining_dims are compatible with root_comm_size
    expected_size = prod(remaining_dims)
    if expected_size != root_comm_size
        error("Dimension mismatch: remaining_dims=$remaining_dims requires $expected_size processes, but root_comm has $root_comm_size processes")
    end
    
    root_cart = MPI.Cart_create(
        root_comm,
        remaining_dims,
        periodic=new_periods,
        reorder=false
    )

    root_coords = MPI.Cart_coords(root_cart, MPI.Comm_rank(root_comm))
    expected_coords = reduced_coords
else
    root_cart = MPI.COMM_NULL
    root_coords = nothing
end


# Note: root_cart is already created above in the first section

# ------------------------------------------------------------
# 4. Pretty printing
# ------------------------------------------------------------
MPI.Barrier(comm)
for r in 0:nprocs-1
    MPI.Barrier(comm)
    if r == rank
        println("Rank $rank")
        println("  Original coords   = $coords")
        println("  Reduced coords    = $reduced_coords, key = $key")
        println("  Root_comm coords  = $(root_coords === nothing ? "N/A" : root_coords)")
        println("  Expected coords   = $expected_coords")
        println("--------------------------------------------------")
    end
end

MPI.Barrier(comm)
MPI.Barrier(comm)
