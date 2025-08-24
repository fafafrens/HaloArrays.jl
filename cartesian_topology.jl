# -- CartesianTopology struct and constructor --
"""
    CartesianTopology{N}

Represents an MPI Cartesian topology of dimension `N`.

Fields:
- `nprocs::Int` : number of processes in the reference communicator.
- `dims::NTuple{N,Int}` : grid partition (result of MPI.Dims_create).
- `global_rank::Int` : rank in the original communicator.
- `cart_coords::NTuple{N,Int}` : Cartesian coordinates of this process in `cart_comm`.
- `neighbors::NTuple{N,NTuple{2,Int}}` : for each dimension, `(src, dest)` from MPI.Cart_shift.
- `comm::MPI.Comm` : reference communicator (e.g. MPI.COMM_WORLD).
- `cart_comm::MPI.Comm` : Cartesian communicator created by MPI.Cart_create.
- `periodic_boundary_condition::NTuple{N,Bool}` : periodicity flags per dimension.
- `active::Bool` : true if the topology is active on this rank (false => `cart_comm == MPI.COMM_NULL`).
"""
struct CartesianTopology{N}
    nprocs::Int
    dims::NTuple{N,Int}
    global_rank::Int
    #shared_rank::Int
    cart_coords::NTuple{N,Int}
    neighbors::NTuple{N,NTuple{2,Int}}
    comm::MPI.Comm
    cart_comm::MPI.Comm
    #shared_comm::MPI.Comm
    #node_name::String
    periodic_boundary_condition::NTuple{N, Bool}  # True if periodic in that dimension
    active::Bool
end

function inactive_cartesian_topology(dims::NTuple{N,Int}) where {N}
   # usa i costanti MPI per valori "null"
   nprocs = 0
   global_rank = MPI.PROC_NULL
   cart_coords = ntuple(i -> MPI.PROC_NULL, Val(N))
   neighbors = ntuple(i -> (MPI.PROC_NULL, MPI.PROC_NULL), Val(N))
   periodic = ntuple(i -> false, Val(N))
    
   return CartesianTopology{N}(nprocs, dims, global_rank, cart_coords, neighbors, MPI.COMM_NULL, MPI.COMM_NULL, periodic, false)
end

inactive_cartesian_topology(n_dimension::Int) = inactive_cartesian_topology(ntuple(i->0, n_dimension))

inactive_cartesian_topology(::Val{N}) where N = inactive_cartesian_topology(ntuple(i->0, Val(N)))

"""
    CartesianTopology(comm::MPI.Comm, dims::NTuple{N,Int}; periodic=ntuple(i->true, Val(N)), active::Bool=true)

Construct a `CartesianTopology` for `comm`.

- If `comm == MPI.COMM_NULL` or `active == false` returns an inactive object
  (`cart_comm == MPI.COMM_NULL`, `cart_coords` set to -1, etc.).
- `dims` may contain zeros; `MPI.Dims_create` will fill them.
- `periodic` indicates per-dimension periodicity.
"""
function CartesianTopology(comm::MPI.Comm, dims::NTuple{N,Int};periodic=ntuple(i->true, Val(N)),active::Bool=true) where {N}
     
     if comm == MPI.COMM_NULL || !active
        return inactive_cartesian_topology(dims)
     end

    nprocs = MPI.Comm_size(comm)
    dims = MPI.Dims_create(nprocs, dims) |> Tuple
    cart_comm = MPI.Cart_create(comm, dims; periodic=periodic, reorder=false) # periodic for demo
    global_rank = MPI.Comm_rank(cart_comm)
    #shared_comm = MPI.Comm_split_type(cart_comm, MPI.COMM_TYPE_SHARED, global_rank)
    #shared_rank = MPI.Comm_rank(shared_comm)
    #node_name = MPI.Get_processor_name()
    cart_coords = MPI.Cart_coords(cart_comm, global_rank) |> Tuple

    neighbors = ntuple(Val(N)) do dim
        MPI.Cart_shift(cart_comm, dim - 1, 1)
    end

    #CartesianTopology{N}(nprocs, dims, global_rank, shared_rank, cart_coords, neighbors, comm, cart_comm, shared_comm, node_name,periodic)
    CartesianTopology{N}(nprocs, dims, global_rank, cart_coords, neighbors, comm, cart_comm,periodic,true)
end

function CartesianTopology(comm::MPI.Comm,n_dimension;periodic=ntuple(i->true, n_dimension),active::Bool=true) 
CartesianTopology(comm, ntuple(i->0, n_dimension); periodic=periodic, active=active)
end


"""
    isactive(cart::CartesianTopology)

Return `true` if the topology is active on this process.
"""
function isactive(cart::CartesianTopology)
    return cart.active
end

"""
    coords_to_color_multi(coords, dims, dims_to_remove) -> Int

Compute an integer `color` for `MPI.Comm_split` by compressing the coordinates
that are NOT removed. `dims_to_remove` is an iterable of 1-based dimension indices.

All ranks that share the same coordinates on the kept dimensions get the same color.
"""
function coords_to_color_multi(coords::NTuple{N,Int}, dims::NTuple{N,Int}, dims_to_remove) where {N}

    tuple_dims_to_remove = Tuple(dims_to_remove)

    rem = (i for i in 1:N if !(i in tuple_dims_to_remove))
    coords_list = (coords[i] for i in rem)
    dims_list   = (dims[i]   for i in rem)
    color = 0
    mul = 1
    for (c,d) in zip(coords_list, dims_list)
        color += c * mul
        mul *= d
    end
    return color
end

"""
    subcomm_for_slices(cart::CartesianTopology{N}, dims_to_reduce) -> (sub_comm, coords, subrank)

Create a sub-communicator grouping ranks equal on all dimensions except those in `dims_to_reduce`.

Returns `(sub_comm, coords, subrank)` where `sub_comm` may be `MPI.COMM_NULL`
and `subrank` is `-1` if so.
"""
function subcomm_for_slices(cart::CartesianTopology{N}, dims_to_reduce) where {N}
     rank = cart.global_rank
     coords = cart.cart_coords
     tuple_dims_to_reduce = Tuple(dims_to_reduce)
     color = coords_to_color_multi(coords, cart.dims, tuple_dims_to_reduce)
     # key: ordina i ranks dentro la slice combinando le coords sulle dimensioni rimosse
     key = 0
     mul = 1
     for i in tuple_dims_to_reduce
        key += coords[i] * mul
        mul *= cart.dims[i]
    end
    sub_comm = MPI.Comm_split(cart.cart_comm, color, key)
    subrank = (sub_comm == MPI.COMM_NULL) ? MPI.PROC_NULL : MPI.Comm_rank(sub_comm)
     return (sub_comm, coords, subrank)
 end

"""
    root_topology_multi(cart::CartesianTopology{N}, dims_to_reduce; root_coord::Int = 0)

Construct a reduced-dimension `CartesianTopology` removing dimensions in `dims_to_reduce`.
Only processes with `coords[dim] == root_coord` for every removed dim build an active topology;
others receive an inactive `CartesianTopology` (`cart_comm == MPI.COMM_NULL`).

Returns a `CartesianTopology{M}` with `M = N - length(dims_to_reduce)`.
"""
function root_topology_multi(cart::CartesianTopology{N}, dims_to_reduce; root_coord::Int = 0) where {N}
    coords = cart.cart_coords
    tuple_dims_to_reduce = Tuple(dims_to_reduce)
    is_root = all(i -> coords[i] == root_coord, tuple_dims_to_reduce)
    color = is_root ? 0 : nothing
    root_comm = MPI.Comm_split(cart.cart_comm, color, cart.global_rank)

    rem = (i for i in 1:N if !(i in tuple_dims_to_reduce))
    new_dims = Tuple(cart.dims[i] for i in rem)
    new_periods = Tuple(cart.periodic_boundary_condition[i] for i in rem)

    if !is_root || root_comm == MPI.COMM_NULL
        return CartesianTopology(root_comm, new_dims; periodic=new_periods, active=false)
    else
        return CartesianTopology(root_comm, new_dims; periodic=new_periods, active=true)
    end
end
