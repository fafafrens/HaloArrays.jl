"""
    gather_haloarray(halo::HaloArray; root=0) -> Array | nothing

Collect a distributed [`HaloArray`](@ref) onto the `root` rank: each rank sends
its interior (ghost-free) subdomain and `root` assembles the full global array.
Returns the assembled `Array` on `root` and `nothing` on every other rank.
"""
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

    owned_shape = size(local_data)
    local_len = prod(owned_shape)

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
        global_size = ntuple(i -> dims[i] * owned_shape[i], Val(N))
        global_array = Array{T}(undef, global_size)

        for r in 0:nproc-1
            coords_r = MPI.Cart_coords(comm, r) |> Tuple
            offset = ntuple(i -> coords_r[i] * owned_shape[i], Val(N))
            inds = ntuple(i -> (offset[i]+1):(offset[i]+owned_shape[i]), Val(N))

            flat_offset = r * local_len + 1
            subarray = reshape(view( recvbuf,flat_offset : flat_offset + local_len - 1), owned_shape...)
            @views global_array[inds...] .= subarray
        end

        return global_array
    else
        return nothing
    end
end
