# MPI tags: side 1 receives from side 2 of the neighbor, and conversely.
@inline tag_send(dim::Int, side::Int) = 2 * (dim - 1) + side
@inline tag_recv(dim::Int, side::Int) = tag_send(dim, 3 - side)

@inline tag_send(::Val{D}, ::Val{S}) where {D,S} = 2 * (D - 1) + S
@inline tag_recv(::Val{D}, ::Val{S}) where {D,S} = tag_send(Val{D}(), Val{3 - S}())

function halo_exchange_waitall!(halo::HaloArray{T,N,A,H,B,BCondition}) where {T,N,A,H,B,BCondition}
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
            send_view = get_send_view(Side(side), dim, halo)
            copyto!(send_bufs[dim][side], send_view)
            recv_reqs[idx] = MPI.Irecv!(recv_bufs[dim][side], comm, recv_reqs[idx]; source=nbrank, tag=tag_recv(dim, side))
            send_reqs[idx] = MPI.Isend(send_bufs[dim][side], comm, send_reqs[idx]; dest=nbrank, tag=tag_send(dim, side))
        end
    end

    MPI.Waitall(recv_reqs)

    @inbounds for dim in 1:N, side in 1:2
        nbrank = topo.neighbors[dim][side]
        if nbrank != MPI.PROC_NULL
            recv_view = get_recv_view(Side(side), dim, halo)
            copyto!(recv_view, recv_bufs[dim][side])
        end
    end

    MPI.Waitall(send_reqs)
    return nothing
end

function halo_exchange_waitall_unsafe!(halo::HaloArray{T,N,A,H,B,BCondition}) where {T,N,A,H,B,BCondition}
    comm = halo.topology.cart_comm
    topo = halo.topology
    recv_reqs = halo.comm_state.unsafe_recv_reqs
    send_reqs = halo.comm_state.unsafe_send_reqs
    recv_bufs = halo.receive_bufs
    send_bufs = halo.send_bufs

    recv_state = (recv_reqs, recv_bufs)
    send_state = (send_reqs, send_bufs)

    @inbounds for dim in 1:N, side in 1:2
        nbrank = topo.neighbors[dim][side]
        if nbrank != MPI.PROC_NULL
            idx = tag_send(dim, side)
            send_view = get_send_view(Side(side), dim, halo)
            copyto!(send_bufs[dim][side], send_view)
            GC.@preserve recv_state MPI.Irecv!(recv_bufs[dim][side], comm, recv_reqs[idx]; source=nbrank, tag=tag_recv(dim, side))
            GC.@preserve send_state MPI.Isend(send_bufs[dim][side], comm, send_reqs[idx]; dest=nbrank, tag=tag_send(dim, side))
        end
    end

    GC.@preserve recv_state MPI.Waitall(recv_reqs)

    @inbounds for dim in 1:N, side in 1:2
        nbrank = topo.neighbors[dim][side]
        if nbrank != MPI.PROC_NULL
            recv_view = get_recv_view(Side(side), dim, halo)
            copyto!(recv_view, recv_bufs[dim][side])
        end
    end

    GC.@preserve send_state MPI.Waitall(send_reqs)
    return nothing
end

function _start_halo_exchange_safe!(halo::HaloArray{T,N,A,H,B,BCondition}) where {T,N,A,H,B,BCondition}
    comm = halo.topology.cart_comm
    topo = halo.topology
    recv_reqs = halo.comm_state.recv_reqs
    send_reqs = halo.comm_state.send_reqs
    recv_bufs = halo.receive_bufs
    send_bufs = halo.send_bufs

    @inbounds for dim in 1:N, side in 1:2
        nbrank = topo.neighbors[dim][side]
        if nbrank != MPI.PROC_NULL
            send_view = get_send_view(Side(side), dim, halo)
            copyto!(send_bufs[dim][side], send_view)
            recv_reqs[dim][side] = MPI.Irecv!(recv_bufs[dim][side], comm, recv_reqs[dim][side]; source=nbrank, tag=tag_recv(dim, side))
            send_reqs[dim][side] = MPI.Isend(send_bufs[dim][side], comm, send_reqs[dim][side]; dest=nbrank, tag=tag_send(dim, side))
        end
    end
    return nothing
end

function _finish_halo_exchange_safe!(halo::HaloArray{T,N,A,H,B,BCondition}) where {T,N,A,H,B,BCondition}
    topo = halo.topology
    recv_reqs = halo.comm_state.recv_reqs
    send_reqs = halo.comm_state.send_reqs
    recv_bufs = halo.receive_bufs

    @inbounds for dim in 1:N, side in 1:2
        nbrank = topo.neighbors[dim][side]
        if nbrank != MPI.PROC_NULL
            MPI.Wait(recv_reqs[dim][side])
            recv_view = get_recv_view(Side(side), dim, halo)
            copyto!(recv_view, recv_bufs[dim][side])
            MPI.Wait(send_reqs[dim][side])
        end
    end
    return nothing
end

function start_halo_exchange_async_unsafe!(halo::HaloArray{T,N,A,H,B,BCondition}) where {T,N,A,H,B,BCondition}
    comm = halo.topology.cart_comm
    topo = halo.topology
    recv_reqs = halo.comm_state.unsafe_recv_reqs_vv
    send_reqs = halo.comm_state.unsafe_send_reqs_vv
    recv_bufs = halo.receive_bufs
    send_bufs = halo.send_bufs

    recv_state = (recv_reqs, recv_bufs)
    send_state = (send_reqs, send_bufs)

    @inbounds for dim in 1:N, side in 1:2
        nbrank = topo.neighbors[dim][side]
        if nbrank != MPI.PROC_NULL
            send_view = get_send_view(Side(side), dim, halo)
            copyto!(send_bufs[dim][side], send_view)
            GC.@preserve recv_state MPI.Irecv!(recv_bufs[dim][side], comm, recv_reqs[dim][side]; source=nbrank, tag=tag_recv(dim, side))
            GC.@preserve send_state MPI.Isend(send_bufs[dim][side], comm, send_reqs[dim][side]; dest=nbrank, tag=tag_send(dim, side))
        end
    end
    return nothing
end

function end_halo_exchange_async_wait_unsafe!(halo::HaloArray{T,N,A,H,B,BCondition}) where {T,N,A,H,B,BCondition}
    topo = halo.topology
    recv_reqs = halo.comm_state.unsafe_recv_reqs_vv
    send_reqs = halo.comm_state.unsafe_send_reqs_vv
    recv_bufs = halo.receive_bufs

    recv_state = (recv_reqs, recv_bufs)
    send_state = send_reqs

    @inbounds for dim in 1:N, side in 1:2
        nbrank = topo.neighbors[dim][side]
        if nbrank != MPI.PROC_NULL
            GC.@preserve recv_state MPI.Wait(recv_reqs[dim][side])
            recv_view = get_recv_view(Side(side), dim, halo)
            copyto!(recv_view, recv_bufs[dim][side])
            GC.@preserve send_state MPI.Wait(send_reqs[dim][side])
        end
    end
    return nothing
end

# Public exchange API.
halo_exchange!(halo::HaloArray) = halo_exchange_waitall_unsafe!(halo)
halo_exchange!(halo::LocalHaloArray) = halo

function halo_exchange!(halo::MultiHaloArray)
    foreach_field!(halo_exchange!, halo)
    return halo
end

function halo_exchange!(halo::LocalMultiHaloArray)
    return halo
end

start_halo_exchange!(halo::HaloArray) = start_halo_exchange_async_unsafe!(halo)
finish_halo_exchange!(halo::HaloArray) = end_halo_exchange_async_wait_unsafe!(halo)
start_halo_exchange!(halo::LocalHaloArray) = halo
finish_halo_exchange!(halo::LocalHaloArray) = halo

function start_halo_exchange!(halo::MultiHaloArray)
    foreach_field!(start_halo_exchange!, halo)
    return halo
end

function finish_halo_exchange!(halo::MultiHaloArray)
    foreach_field!(finish_halo_exchange!, halo)
    return halo
end

start_halo_exchange!(halo::LocalMultiHaloArray) = halo
finish_halo_exchange!(halo::LocalMultiHaloArray) = halo

function synchronize_halo!(halo::HaloArray)
    halo_exchange!(halo)
    boundary_condition!(halo)
    return halo
end

function synchronize_halo!(halo::LocalHaloArray)
    boundary_condition!(halo)
    return halo
end

function synchronize_halo!(halo::MultiHaloArray)
    halo_exchange!(halo)
    boundary_condition!(halo)
    return halo
end

function synchronize_halo!(halo::LocalMultiHaloArray)
    boundary_condition!(halo)
    return halo
end

# Compatibility wrappers for the older API names.
halo_exchange_wait!(halo::HaloArray) = halo_exchange_waitall!(halo)

function start_halo_exchange_async!(halo::HaloArray)
    _start_halo_exchange_safe!(halo)
    return nothing
end

function end_halo_exchange_wait!(halo::HaloArray)
    _finish_halo_exchange_safe!(halo)
    return nothing
end

function halo_exchange_async!(halo::HaloArray)
    start_halo_exchange_async!(halo)
    end_halo_exchange_wait!(halo)
    return nothing
end

function halo_exchange_async_unsafe!(halo::HaloArray)
    start_halo_exchange_async_unsafe!(halo)
    end_halo_exchange_async_wait_unsafe!(halo)
    return nothing
end

halo_exchange_async_wait!(halo::HaloArray) = end_halo_exchange_wait!(halo)
halo_exchange_async_wait_unsafe!(halo::HaloArray) = end_halo_exchange_async_wait_unsafe!(halo)

function start_halo_exchange_async_unsafe!(halo::MultiHaloArray)
    foreach_field!(start_halo_exchange_async_unsafe!, halo)
    return nothing
end

function end_halo_exchange_async_wait_unsafe!(halo::MultiHaloArray)
    foreach_field!(end_halo_exchange_async_wait_unsafe!, halo)
    return nothing
end
