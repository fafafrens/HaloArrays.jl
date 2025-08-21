function gather_haloarray(halo::HaloArray; root::Int=0)
    comm = halo.topology.cart_comm
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)
    N = ndims(halo)
    T = eltype(halo)

    h = halo_width(halo)
    coords = halo.topology.cart_coords
    dims = halo.topology.dims

    local_data = interior_view(halo)

    local_size = size(local_data)
    local_len = prod(local_size)

    # Gather all buffers as flat arrays
    sendbuf = collect(vec(local_data))# Array(local_data)
    recvbuf = if rank == root
        Array{T}(undef, local_len * nproc)
    else
        nothing
    end

    # Do the actual gather
    MPI.Gather!(sendbuf, recvbuf, comm; root=root)

    # Reconstruct the full array at the root
    if rank == root
        global_size = ntuple(i -> dims[i] * local_size[i], Val(N))
        global_array = Array{T}(undef, global_size)

        for r in 0:nproc-1
            coords_r = MPI.Cart_coords(comm, r) |> Tuple
            offset = ntuple(i -> coords_r[i] * local_size[i], Val(N))
            inds = ntuple(i -> (offset[i]+1):(offset[i]+local_size[i]), Val(N))

            flat_offset = r * local_len + 1
            subarray = reshape(view( recvbuf,flat_offset : flat_offset + local_len - 1), local_size...)
            @views global_array[inds...] .= subarray
        end

        return global_array
    else
        return nothing
    end
end