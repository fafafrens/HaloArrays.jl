function halo_exchange!(halo::HaloArray,side::Side{S}, dim::Dim{D}) where {S,D}
    h= halo_width(halo)
    comm = halo.topology.cart_comm
    topo = halo.topology

    recv_reqs = halo.comm_state.recv_reqs
    send_reqs = halo.comm_state.send_reqs
    recv_bufs = halo.receive_bufs
    send_bufs = halo.send_bufs

    nbrank = topo.neighbors[D][S]
    if nbrank != MPI.PROC_NULL
        recv_reqs[D][S] = MPI.Irecv!(recv_bufs[D][S], comm, recv_reqs[D][S]; source=nbrank, tag=tag_recv(Val{D}(), Val{S}()))
        send_view = get_send_view(Side(S), Dim(D), halo.data, h)
        copyto!(send_bufs[D][S], send_view)

        send_reqs[D][S] = MPI.Isend(send_bufs[D][S], comm, send_reqs[D][S]; dest=nbrank, tag=tag_send(Val{D}(), Val{S}()))

        recv_ready = false
        send_ready = false

        while !(recv_ready && send_ready)
            if MPI.Test(recv_reqs[D][S]) && !recv_ready
                recv_view = get_recv_view(Side(S), Dim(D), halo.data, h)
                copyto!(recv_view, recv_bufs[D][S])
                recv_ready = true
            end

            send_ready = MPI.Test(send_reqs[D][S])
        end
    end
    return 1
end


function halo_exchange!(halo::HaloArray{T, N, A,H,S, B,C}) where {T,N,A,H,S,B,C}
    ntuple(Val(N)) do D
        ntuple(Val(2)) do S
            halo_exchange!(halo,Side(S), Dim(D))
        end
    end
    return nothing
end


# helper tags: runtime (Int) and compile-time (Val) interfaces
@inline tag_send(dim::Int, side::Int) = 2*(dim - 1) + side
@inline tag_recv(dim::Int, side::Int) = tag_send(dim, 3 - side)

@inline tag_send(::Val{D}, ::Val{S}) where {D,S} = 2*(D - 1) + S
@inline tag_recv(::Val{D}, ::Val{S}) where {D,S} = tag_send(Val{D}(), Val{3 - S}())


function halo_exchange_wait!(halo::HaloArray,side::Side{S}, dim::Dim{D}) where {S,D}
    h= halo_width(halo)
    comm = halo.topology.cart_comm
    topo = halo.topology

    recv_reqs = halo.comm_state.recv_reqs
    send_reqs = halo.comm_state.send_reqs
    recv_bufs = halo.receive_bufs
    send_bufs = halo.send_bufs

    nbrank = topo.neighbors[D][S]
    if nbrank != MPI.PROC_NULL
        send_view = get_send_view(side, dim, halo.data, h)
        copyto!(send_bufs[D][S], send_view)
        # Post non-blocking recv and send (use compile-time tags)
        recv_reqs[D][S] = MPI.Irecv!(recv_bufs[D][S], comm, recv_reqs[D][S]; source=nbrank, tag=tag_recv(Val{D}(), Val{S}()))
        send_reqs[D][S] = MPI.Isend(send_bufs[D][S], comm, send_reqs[D][S]; dest=nbrank, tag=tag_send(Val{D}(), Val{S}()))
        # Wait for the receive to complete, then copy buffer into halo region
        MPI.Wait(recv_reqs[D][S])
        recv_view = get_recv_view(side, dim, halo.data, h)
        copyto!(recv_view, recv_bufs[D][S])
        # Wait for send completion
        MPI.Wait(send_reqs[D][S])
    end
    return 1
end

function halo_exchange_wait!(halo::HaloArray{T,N,A,Halo,B,C,BCondition}) where {T,N,A,Halo,B,C,BCondition}
    ntuple(Val(N)) do D
        ntuple(Val(2)) do s
            # Call the halo exchange wait for each side and dimension
            halo_exchange_wait!(halo,Side(s), Dim(D))
        end
    end
    return nothing
end


    function halo_exchange_waitall!(halo::HaloArray{T, N, A,H,S, B,C,BCondition}) where {T,N,A,H,S,B,C,BCondition}

    h= halo_width(halo)
    comm = halo.topology.cart_comm
    topo = halo.topology

    recv_reqs_flat = halo.comm_state.recv_reqs_flat
    send_reqs_flat = halo.comm_state.send_reqs_flat
    recv_bufs = halo.receive_bufs
    send_bufs = halo.send_bufs

    for D in 1:N
        nbrank = topo.neighbors[D][1]
        if nbrank != MPI.PROC_NULL
            send_view = get_send_view(Side(1), D, halo)
            copyto!(send_bufs[D][1], send_view)
            idx = 2*(D - 1) + 1
            recv_reqs_flat[idx] = MPI.Irecv!(recv_bufs[D][1], comm, recv_reqs_flat[idx]; source=nbrank, tag=tag_recv(D,1))
            send_reqs_flat[idx] = MPI.Isend(send_bufs[D][1], comm, send_reqs_flat[idx]; dest=nbrank, tag=tag_send(D,1))
        end

        nbrank = topo.neighbors[D][2]
        if nbrank != MPI.PROC_NULL
            send_view = get_send_view(Side(2), D, halo)
            copyto!(send_bufs[D][2], send_view)
            idx = 2*(D - 1) + 2
            recv_reqs_flat[idx] = MPI.Irecv!(recv_bufs[D][2], comm, recv_reqs_flat[idx]; source=nbrank, tag=tag_recv(D,2))
            send_reqs_flat[idx] = MPI.Isend(send_bufs[D][2], comm, send_reqs_flat[idx]; dest=nbrank, tag=tag_send(D,2))
        end
    end

    MPI.Waitall(recv_reqs_flat)

    @inbounds for D in 1:N
        nbrank = topo.neighbors[D][1]
        if nbrank != MPI.PROC_NULL
            recv_view = get_recv_view(Side(1), D, halo)
            copyto!(recv_view, recv_bufs[D][1])
        end
        nbrank = topo.neighbors[D][2]
        if nbrank != MPI.PROC_NULL
            recv_view = get_recv_view(Side(2), D, halo)
            copyto!(recv_view, recv_bufs[D][2])
        end
    end

    MPI.Waitall(send_reqs_flat)

    return nothing
end


function halo_exchange_waitall_unsafe!(halo::HaloArray{T,N,A,Halo,B,C,BCondition}) where {T,N,A,Halo,B,C,BCondition}

    h= halo_width(halo)
    comm = halo.topology.cart_comm
    topo = halo.topology

    unsafe_rec_req = halo.comm_state.unsafe_recv_reqs
    unsafe_send_req = halo.comm_state.unsafe_send_reqs
    rec_bufs = halo.receive_bufs
    send_bufs = halo.send_bufs

    rec_state = (unsafe_rec_req, rec_bufs)
    send_state = (unsafe_send_req, send_bufs)

    for D in 1:N
        @inbounds nbrank = topo.neighbors[D][1]
        if nbrank != MPI.PROC_NULL
            send_view = get_send_view(Side{1}(), D, halo)
            copyto!(send_bufs[D][1], send_view)
            idx = 2*(D - 1) + 1
            GC.@preserve rec_state MPI.Irecv!(rec_bufs[D][1], comm, unsafe_rec_req[idx]; source=nbrank, tag=tag_recv(D,1))
            GC.@preserve send_state MPI.Isend(send_bufs[D][1], comm, unsafe_send_req[idx]; dest=nbrank, tag=tag_send(D,1))
        end
    end

    @inbounds for D in 1:N
        nbrank = topo.neighbors[D][2]
        if nbrank != MPI.PROC_NULL
            send_view = get_send_view(Side{2}(), D, halo)
            copyto!(send_bufs[D][2], send_view)
            idx = 2*(D - 1) + 2
            GC.@preserve rec_state MPI.Irecv!(rec_bufs[D][2], comm, unsafe_rec_req[idx]; source=nbrank, tag=tag_recv(D,2))
            GC.@preserve send_state MPI.Isend(send_bufs[D][2], comm, unsafe_send_req[idx]; dest=nbrank, tag=tag_send(D,2))
        end
    end

    GC.@preserve rec_state MPI.Waitall(unsafe_rec_req)

    @inbounds for D in 1:N
        nbrank = topo.neighbors[D][1]
        if nbrank != MPI.PROC_NULL
            recv_view = get_recv_view(Side{1}(), D, halo)
            copyto!(recv_view, rec_bufs[D][1])
        end
    end

    @inbounds for D in 1:N
        nbrank = topo.neighbors[D][2]
        if nbrank != MPI.PROC_NULL
            recv_view = get_recv_view(Side{2}(),D, halo)
            copyto!(recv_view, rec_bufs[D][2])
        end
    end

    GC.@preserve send_state MPI.Waitall(unsafe_send_req)

    return nothing
end



function halo_exchange_async!(halo::HaloArray, side::Side{S}, dim::Dim{D}) where {S,D}
    h = halo_width(halo)
    comm = halo.topology.cart_comm
    topo = halo.topology

    recv_reqs = halo.comm_state.recv_reqs
    send_reqs = halo.comm_state.send_reqs
    recv_bufs = halo.receive_bufs
    send_bufs = halo.send_bufs

    nbrank = topo.neighbors[D][S]
    if nbrank != MPI.PROC_NULL
        recv_reqs[D][S] = MPI.Irecv!(recv_bufs[D][S], comm, recv_reqs[D][S]; source=nbrank, tag=tag_recv(Val{D}(), Val{S}()))
        send_view = get_send_view(side, dim, halo)
        copyto!(send_bufs[D][S], send_view)
        send_reqs[D][S] = MPI.Isend(send_bufs[D][S], comm, send_reqs[D][S]; dest=nbrank, tag=tag_send(Val{D}(), Val{S}()))
    end

end

function halo_exchange_async!(halo::HaloArray, side::Side{S}, D::Int) where {S}
    h = halo_width(halo)
    comm = halo.topology.cart_comm
    topo = halo.topology

    recv_reqs = halo.comm_state.recv_reqs
    send_reqs = halo.comm_state.send_reqs
    recv_bufs = halo.receive_bufs
    send_bufs = halo.send_bufs

    nbrank = topo.neighbors[D][S]
    if nbrank != MPI.PROC_NULL
        recv_reqs[D][S] = MPI.Irecv!(recv_bufs[D][S], comm, recv_reqs[D][S]; source=nbrank, tag=tag_recv(D,S))
        send_view = get_send_view(side, D, halo)
        copyto!(send_bufs[D][S], send_view)
        send_reqs[D][S] = MPI.Isend(send_bufs[D][S], comm, send_reqs[D][S]; dest=nbrank, tag=tag_send(D,S))
    end

end

function start_halo_exchange_async!(halo::HaloArray{T,N,A,Halo,B,C,BCondition}) where {T,N,A,Halo,B,C,BCondition}
    for D in 1:N
        halo_exchange_async!(halo, Side{1}(),D)
        halo_exchange_async!(halo, Side{2}(), D)
    end
    return nothing
end

function halo_exchange_async_wait!(halo::HaloArray, side::Side{S}, dim::Dim{D}) where {S,D}
    h = halo_width(halo)
    comm = halo.topology.cart_comm
    topo = halo.topology

    recv_reqs = halo.comm_state.recv_reqs
    send_reqs = halo.comm_state.send_reqs
    recv_bufs = halo.receive_bufs
    send_bufs = halo.send_bufs

    nbrank = topo.neighbors[D][S]
    if nbrank != MPI.PROC_NULL
        MPI.Wait(recv_reqs[D][S])
        recv_view = get_recv_view(side, dim, halo)
        copyto!(recv_view, recv_bufs[D][S])
        MPI.Wait(send_reqs[D][S])
    end

end

function halo_exchange_async_wait!(halo::HaloArray, side::Side{S}, D::Int) where {S}
    h = halo_width(halo)
    comm = halo.topology.cart_comm
    topo = halo.topology

    recv_reqs = halo.comm_state.recv_reqs
    send_reqs = halo.comm_state.send_reqs
    recv_bufs = halo.receive_bufs
    send_bufs = halo.send_bufs

    nbrank = topo.neighbors[D][S]
    if nbrank != MPI.PROC_NULL
        MPI.Wait(recv_reqs[D][S])
        recv_view = get_recv_view(side, D, halo)
        copyto!(recv_view, recv_bufs[D][S])
        MPI.Wait(send_reqs[D][S])
    end

end

function end_halo_exchange_wait!(halo::HaloArray{T, N, A,Halo,B,C,BCondition}) where {T,N,A,Halo,B,C,BCondition}
    for D in 1:N
        halo_exchange_async_wait!(halo, Side{1}(),D)
        halo_exchange_async_wait!(halo, Side{2}(),D)
    end
    return nothing
end

function halo_exchange_async!(halo::HaloArray{T, N, A,Halo,B,C,BCondition}) where {T,N,A,Halo,B,C,BCondition}
    start_halo_exchange_async!(halo)
    ###Here you can put additional
    end_halo_exchange_wait!(halo)
    return nothing
end



function halo_exchange_async_unsafe!(halo::HaloArray, side::Side{S}, D::Int) where {S}
    h = halo_width(halo)
    comm = halo.topology.cart_comm
    topo = halo.topology

    recv_reqs = halo.comm_state.unsafe_recv_reqs_vv
    send_reqs = halo.comm_state.unsafe_send_reqs_vv
    recv_bufs = halo.receive_bufs
    send_bufs = halo.send_bufs

    rec_state = (recv_reqs , recv_bufs)
    send_state = (send_reqs, send_bufs)

    nbrank = topo.neighbors[D][S]
    if nbrank != MPI.PROC_NULL
        GC.@preserve rec_state MPI.Irecv!(recv_bufs[D][S], comm, recv_reqs[D][S]; source=nbrank, tag=tag_recv(D,S))
        send_view = get_send_view(side, D, halo)
        copyto!(send_bufs[D][S], send_view)
        GC.@preserve send_state MPI.Isend(send_bufs[D][S], comm, send_reqs[D][S]; dest=nbrank, tag=tag_send(D,S))
    end

end

function start_halo_exchange_async_unsafe!(halo::HaloArray{T, N, A,Halo,B,C,BCondition}) where {T,N,A,Halo,B,C,BCondition}
    for D in 1:N
        halo_exchange_async_unsafe!(halo, Side{1}(),D)
        halo_exchange_async_unsafe!(halo, Side{2}(), D)
    end
    return nothing
end

function halo_exchange_async_wait_unsafe!(halo::HaloArray, side::Side{S}, D::Int) where {S}
    h = halo_width(halo)
    comm = halo.topology.cart_comm
    topo = halo.topology

    recv_reqs = halo.comm_state.unsafe_recv_reqs_vv
    send_reqs = halo.comm_state.unsafe_send_reqs_vv
    recv_bufs = halo.receive_bufs
    send_bufs = halo.send_bufs

    rec_state = (recv_reqs , recv_bufs)
    send_state = (send_reqs, send_bufs)

    nbrank = topo.neighbors[D][S]
    if nbrank != MPI.PROC_NULL
        GC.@preserve rec_state MPI.Wait(recv_reqs[D][S])
        recv_view = get_recv_view(side, D, halo)
        copyto!(recv_view, recv_bufs[D][S])
        GC.@preserve send_state MPI.Wait(send_reqs[D][S])
    end

end

function end_halo_exchange_async_wait_unsafe!(halo::HaloArray{T, N, A,Halo,B,C,BCondition}) where {T,N,A,Halo,B,C,BCondition}
    for D in 1:N
        halo_exchange_async_wait_unsafe!(halo, Side{1}(),D)
        halo_exchange_async_wait_unsafe!(halo, Side{2}(),D)
    end
    return nothing
end

function halo_exchange_async_unsafe!(halo::HaloArray{T,N,A,Halo,B,C,BCondition}) where {T,N,A,Halo,B,C,BCondition}
    start_halo_exchange_async_unsafe!(halo)
    ###Here you can put additional
    end_halo_exchange_async_wait_unsafe!(halo)
    return nothing
end

function start_halo_exchange_async_unsafe!(halo::MultiHaloArray)
    foreach_field!( start_halo_exchange_async_unsafe!,halo)
    return nothing
end

function end_halo_exchange_async_wait_unsafe!(halo::MultiHaloArray)
    foreach_field!( end_halo_exchange_async_wait_unsafe!,halo)
    return nothing
end