# ...existing code...
using MPI

function create_cartesian(comm, tot_dim)
    nprocs = MPI.Comm_size(comm)
    dims = MPI.Dims_create(nprocs, ntuple(_ -> 0, tot_dim))
    periods = ntuple(_ -> true, tot_dim)
    cart_comm = MPI.Cart_create(comm, dims, periodic=periods)
    return (cart_comm, dims, periods)
end

# Convert coords tuple (with one dim removed) to integer color for Comm_split
function coords_to_color(coords, dims, dim_to_remove::Int)
    coords_list = [coords[i] for i in 1:length(coords) if i != dim_to_remove]
    dims_list   = [dims[i]   for i in 1:length(dims)   if i != dim_to_remove]
    color = 0
    multiplier = 1
    for (c, d) in zip(coords_list, dims_list)
        color += c * multiplier
        multiplier *= d
    end
    return color
end

# Split communicator so that each subcomm groups ranks that differ only along dim_to_reduce
function subcomms_by_slice(cart_comm, dims, dim_to_reduce)
    rank = MPI.Comm_rank(cart_comm)
    coords = MPI.Cart_coords(cart_comm, rank)
    color = coords_to_color(coords, dims, dim_to_reduce)
    # key: use coordinate along reduced dim so ranks inside subcomm ordered by that coord
    key = coords[dim_to_reduce]
    sub_comm = MPI.Comm_split(cart_comm, color, key)
    return (sub_comm, coords)
end

# On each subcomm perform reduction (sum) along the reduced dimension.
# The root will be the rank inside subcomm whose reduced-dim coordinate == 0
function reduce_along_dim(cart_comm, dims, dim_to_reduce, local_value)
    (sub_comm, coords) = subcomms_by_slice(cart_comm, dims, dim_to_reduce)
    # rank inside sub_comm
    subrank = MPI.Comm_rank(sub_comm)
    # find if this subrank corresponds to coordinate 0 along reduced dim:
    # because we used key=coords[dim_to_reduce], the rank ordering inside subcomm
    # follows increasing reduced-dim coord; root should be subrank==0 if coord 0 exists.
    root = 0
    # perform reduction: result only valid on root
    sum_on_root = MPI.Reduce(local_value, MPI.SUM, root, sub_comm)
    # Free subcomm
    MPI.free(sub_comm)
    return (sum_on_root, subrank == root ? sum_on_root : nothing)
end

# Build 2D Cartesian communicator for root ranks (those with coord==0 along dim_to_reduce)
function root_cartesian_for_dim(cart_comm, dims, periods, dim_to_reduce)
    rank = MPI.Comm_rank(cart_comm)
    coords = MPI.Cart_coords(cart_comm, rank)
    is_root = coords[dim_to_reduce] == 0
    color = is_root ? 0 : nothing
    root_comm = MPI.Comm_split(cart_comm, color, rank)
    if is_root
        new_dims = Tuple(deleteat!(collect(dims), dim_to_reduce))

        new_periods = Tuple(deleteat!(collect(periods), dim_to_reduce))
        new_cart = MPI.Cart_create(root_comm, new_dims, periodic=new_periods)
        return (true, new_cart, MPI.Cart_coords(new_cart, MPI.Comm_rank(new_cart)))
    else
        return (false, MPI.COMM_NULL, nothing)
    end
end

# ---------- main script ----------
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

tot_dimension = 4
cart_comm, dims, periods = create_cartesian(comm, tot_dimension)
coords = MPI.Cart_coords(cart_comm, rank)

dim_to_reduce = 2  # 1-based index: riduci la seconda dimensione

MPI.Barrier(comm)
for r in 0:nprocs-1
    MPI.Barrier(comm)
    if r == rank
        println("Rank $rank")
        println("  dims    = $dims")
        println("  coords  = $coords")
        println("--------------------------------------------------")
    end
end
MPI.Barrier(comm)

# Esempio di valore locale da ridurre (usiamo il rank+1 per semplicità)
local_value = rank + 1

# Esegui la riduzione lungo la dimensione scelta
(sum_on_root, my_root_result) = reduce_along_dim(cart_comm, dims, dim_to_reduce, local_value)

# Stampiamo risultato: only roots will have the sum (my_root_result non-nothing)
MPI.Barrier(comm)
if my_root_result !== nothing
    println("Rank $rank is root for its slice. Sum along dim $dim_to_reduce = $my_root_result")
end

# Costruisci communicator per root ranks e stampa le nuove coords dei root
(is_root, new_cart, new_coords) = root_cartesian_for_dim(cart_comm, dims, periods, dim_to_reduce)

MPI.Barrier(comm)
summary = MPI.gather((rank, coords, is_root, new_coords), comm)

if rank == 0
    println("\nSummary after split + reduction:")
    for (r, c, root, ncoords) in summary
        println("Rank $r: coords = $c, is_root = $root, root_cart_coords = $(ncoords === nothing ? "-" : ncoords)")
    end
end

# pulizia