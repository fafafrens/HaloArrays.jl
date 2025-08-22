using MPI

# include topology definition
include("/Users/eduardogrossi/mpistuff/cartesian_topology.jl")

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

# choose total dimension and the dimension to reduce (1-based)
N = 3
dim_to_reduce = 2

# build full Cartesian topology using provided constructor
topo = CartesianTopology(comm, ntuple(_->0, Val(N)); periodic=ntuple(_->true, Val(N)))

# basic checks
if !isactive(topo)
    error("Base CartesianTopology not active on rank $rank")
end

dims = topo.dims
coords = topo.cart_coords

# prepare local value and perform a reduction over each slice (ranks that
# share all coordinates except dim_to_reduce)
local_value = rank + 1

# compute color for Comm_split: compress coords except dim_to_reduce
function coords_to_color(coords, dims, dim_remove)
    coords_list = [coords[i] for i in 1:length(coords) if i != dim_remove]
    dims_list   = [dims[i]   for i in 1:length(dims)   if i != dim_remove]
    color = 0
    mul = 1
    for (c,d) in zip(coords_list, dims_list)
        color += c * mul
        mul *= d
    end
    return color
end

color = coords_to_color(coords, dims, dim_to_reduce)
key = coords[dim_to_reduce]  # ordering inside subcomm by coordinate along reduced dim
sub_comm = MPI.Comm_split(topo.cart_comm, color, key)

# perform reduction: root inside sub_comm is the process with reduced-dim coord == 0,
# which has subrank == 0 because we used key = coords[dim_to_reduce] (coords start at 0).
root_in_sub = 0
sum_on_root = MPI.Reduce(local_value, MPI.SUM, root_in_sub, sub_comm)
subrank = MPI.Comm_rank(sub_comm)

# compute expected sum by enumerating coords along reduced dimension
expected = 0
for k in 0:(dims[dim_to_reduce]-1)
    coords_k = Tuple(
        [ (i==dim_to_reduce ? k : coords[i]) for i in 1:length(coords) ]
    )
    r = MPI.Cart_rank(topo.cart_comm, coords_k)
    global expected += r + 1
end

# check reduction result on the root of the slice
if subrank == root_in_sub
    if sum_on_root != expected
        error("Reduction mismatch on global rank $rank (slice root). got=$sum_on_root expected=$expected")
    end
end

MPI.Barrier(comm)
MPI.free(sub_comm)  # free the temporary sub communicator

# Build root communicator (ranks with coords[dim_to_reduce] == 0)
is_root = coords[dim_to_reduce] == 0
color_root = is_root ? 0 : nothing
root_comm = MPI.Comm_split(topo.cart_comm, color_root, rank)

# new dims and periods with the removed dimension
new_dims = Tuple(deleteat!(collect(dims), dim_to_reduce))
new_periods = Tuple(deleteat!(collect(topo.periodic_boundary_condition), dim_to_reduce))

# create a CartesianTopology for the root group; for non-roots create a non-active topology
if is_root && root_comm != MPI.COMM_NULL
    root_topo = CartesianTopology(root_comm, new_dims; periodic=new_periods)
else
    # constructor handles comm == MPI.COMM_NULL and active=false semantics;
    # supply active=false explicitly for clarity
    root_topo = CartesianTopology(MPI.COMM_NULL, new_dims; periodic=new_periods, active=false)
end

# Validate root_topo properties
if is_root
    if !isactive(root_topo)
        error("Expected active root_topo on rank $rank but got inactive")
    end
    # check dims match
    if root_topo.dims != new_dims
        error("Root topology dims mismatch on rank $rank: got=$(root_topo.dims) expected=$new_dims")
    end
else
    if isactive(root_topo)
        error("Expected inactive root_topo on non-root rank $rank")
    end
end

# Collect a brief summary to rank 0 and print
summary = MPI.gather((rank, coords, is_root, isactive(root_topo) ? root_topo.cart_coords : nothing), comm)

if rank == 0
    println("Test summary for root_topology (dim_to_reduce = $dim_to_reduce):")
    for (r, c, rootflag, rcoords) in summary
        println("Rank $r: coords=$c is_root=$rootflag root_coords=$(rcoords === nothing ? "-" : rcoords)")
    end
    println("\nAll checks passed if no errors were thrown.")
end

MPI.Barrier(comm)
MPI.Finalize()