# -- CartesianTopology struct and constructor --
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

function CartesianTopology(comm::MPI.Comm, dims::NTuple{N,Int}=ntuple(i->0, Val(N));periodic=ntuple(i->true, Val(N)),active::Bool=true) where {N,Int}
    
    if comm == MPI.COMM_NULL || !active
        nprocs = 0
        global_rank = -1
        cart_comm = MPI.COMM_NULL
        cart_coords = ntuple(i->-1, Val(N))
        neighbors = ntuple(i->(-1,-1), Val(N))
        return CartesianTopology{N}(nprocs, dims, global_rank, cart_coords, neighbors, MPI.COMM_NULL, MPI.COMM_NULL, periodic, false)
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


function isactive(cart::CartesianTopology)
    return cart.active
end

function coords_to_color_multi(coords::NTuple{N,Int}, dims::NTuple{N,Int}, dims_to_remove::AbstractVector{Int}) where {N}
    rem = (i for i in 1:N if !(i in dims_to_remove))
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


function subcomm_for_slices(cart::CartesianTopology{N}, dims_to_reduce::AbstractVector{Int}) where {N}
    rank = cart.global_rank
    coords = cart.cart_coords
    color = coords_to_color_multi(coords, cart.dims, dims_to_reduce)
    # key: ordina i ranks dentro la slice combinando le coords sulle dimensioni rimosse
    key = 0
    mul = 1
    for i in dims_to_reduce
        key += coords[i] * mul
        mul *= cart.dims[i]
    end
    sub_comm = MPI.Comm_split(cart.cart_comm, color, key)
    subrank = (sub_comm == MPI.COMM_NULL) ? -1 : MPI.Comm_rank(sub_comm)
    return (sub_comm, coords, subrank)
end

function root_topology_multi(cart::CartesianTopology{N}, dims_to_reduce::AbstractVector{Int}; root_coord::Int = 0) where {N}
    coords = cart.cart_coords
    is_root = all(i -> coords[i] == root_coord, dims_to_reduce)
    color = is_root ? 0 : nothing
    root_comm = MPI.Comm_split(cart.cart_comm, color, cart.global_rank)

    rem = [i for i in 1:N if !(i in dims_to_reduce)]
    new_dims = Tuple(cart.dims[i] for i in rem)
    new_periods = Tuple(cart.periodic_boundary_condition[i] for i in rem)

    if !is_root || root_comm == MPI.COMM_NULL
        return CartesianTopology(root_comm, new_dims; periodic=new_periods, active=false)
    else
        return CartesianTopology(root_comm, new_dims; periodic=new_periods)
    end
end
