const FLUX_CONTRIBUTION_TAG_OFFSET = 1000

@inline tag_flux_contribution_send(dim::Int, side::Int) =
    FLUX_CONTRIBUTION_TAG_OFFSET + tag_send(dim, side)

@inline tag_flux_contribution_recv(dim::Int, side::Int) =
    tag_flux_contribution_send(dim, 3 - side)

"""
    synchronize_flux_contributions!(du)

Add contributions stored in MPI ghost cells back into the neighboring owned
boundary cells.

This is the reverse/additive companion to `synchronize_halo!`: normal halo
exchange copies owned cells into ghost cells for reading, while this routine
sends ghost-cell contributions to the rank that owns those cells and adds the
received values into the local owned boundary slabs.
"""
function synchronize_flux_contributions!(halo::HaloArray{T, N, A, H, B, BCondition}) where {T, N, A, H, B, BCondition}
    comm = halo.topology.cart_comm
    topo = halo.topology
    recv_reqs = halo.comm_state.recv_reqs_flat
    send_reqs = halo.comm_state.send_reqs_flat
    recv_bufs = halo.receive_bufs
    send_bufs = halo.send_bufs

    @inbounds for dim in 1:N, side in 1:2
        nbrank = topo.neighbors[dim][side]
        if nbrank != MPI.PROC_NULL
            idx = tag_send(dim, side)
            ghost_contributions = get_recv_view(Side(side), dim, halo)
            copyto!(send_bufs[dim][side], ghost_contributions)
            recv_reqs[idx] = MPI.Irecv!(
                recv_bufs[dim][side],
                comm,
                recv_reqs[idx];
                source = nbrank,
                tag = tag_flux_contribution_recv(dim, side),
            )
            send_reqs[idx] = MPI.Isend(
                send_bufs[dim][side],
                comm,
                send_reqs[idx];
                dest = nbrank,
                tag = tag_flux_contribution_send(dim, side),
            )
        end
    end

    MPI.Waitall(recv_reqs)

    @inbounds for dim in 1:N, side in 1:2
        nbrank = topo.neighbors[dim][side]
        if nbrank != MPI.PROC_NULL
            owned_boundary = get_send_view(Side(side), dim, halo)
            owned_boundary .+= recv_bufs[dim][side]
        end
    end

    MPI.Waitall(send_reqs)
    return halo
end

function synchronize_flux_contributions!(halo::LocalHaloArray)
    N = ndims(halo)

    @inbounds for dim in 1:N
        if isperiodic(halo.boundary_condition[dim][1]) && isperiodic(halo.boundary_condition[dim][2])
            left_ghost = get_recv_view(Side(1), Dim(dim), halo)
            right_ghost = get_recv_view(Side(2), Dim(dim), halo)
            left_owned = get_send_view(Side(1), Dim(dim), halo)
            right_owned = get_send_view(Side(2), Dim(dim), halo)

            right_owned .+= left_ghost
            left_owned .+= right_ghost
        end
    end

    return halo
end

function synchronize_flux_contributions!(halo::MultiHaloArray)
    foreach_field!(synchronize_flux_contributions!, halo)
    return halo
end

function synchronize_flux_contributions!(halo::LocalMultiHaloArray)
    foreach_field!(synchronize_flux_contributions!, halo)
    return halo
end
