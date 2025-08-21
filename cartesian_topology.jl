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


