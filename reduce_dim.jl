
function mapreduce_dim_split(halo::HaloArray, f, op, dim::Int)
    comm = get_comm(halo)
    cart_comm = halo.topology.cart_comm
    N = length(halo.topology.dims)
    @assert 1 ≤ dim ≤ N "Invalid dimension $dim for HaloArray with $N dimensions."

    # 1. Local reduction
    local_data = interior_view(halo)
    reduced_local = mapreduce(f, op, local_data; dims=dim)
    reduced_local = dropdims(reduced_local; dims=dim)

    # 2. Create sub-communicator (excluding 'dim')
    remain_dims = ntuple(i -> i != dim, N)
    sub_comm = MPI.Cart_sub(cart_comm, remain_dims)
    sub_rank = MPI.Comm_rank(sub_comm)

    # 3. MPI reduction within sub_comm
    reduced_global = similar(reduced_local)
    op_mpi = MPI.Op(op, typeof(reduced_local))
    MPI.Reduce!(reduced_local, reduced_global, op_mpi, 0, sub_comm)

    # 4. Extract only the root rank of each sub_comm
    color = (sub_rank == 0) ? 0 : MPI.UNDEFINED
    key = MPI.Comm_rank(cart_comm)
    root_comm = MPI.Comm_split(cart_comm, color, key)

    if sub_rank == 0
        new_dims = Tuple(deleteat!(collect(top.dims), dim))
        new_periods = Tuple(deleteat!(collect(top.periods), dim))
        new_topology = CartesianTopology(root_comm, new_dims, periodic=new_periods)
        return active(HaloArray(reduced_global, new_topology, halo.halo, halo.boundary))
    else
        dummy = HaloArray(similar(reduced_local, (0,)), top, halo.halo, halo.boundary)
        return inactive(dummy)
    end
end


