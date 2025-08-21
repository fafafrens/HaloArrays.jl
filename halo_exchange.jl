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
                #recv_reqs[D][S] = MPI.Irecv!(recv_bufs[D][S], comm; source=nbrank)
                recv_reqs[D][S] = MPI.Irecv!(recv_bufs[D][S], comm,recv_reqs[D][S]; source=nbrank,tag=tag_rec(Val(S)))

                send_view = get_send_view(Side(S), Dim(D), halo.data, h)
                copyto!(send_bufs[D][S], send_view)
            
                            # initiate non-blocking MPI send
                #send_reqs[D][S] = MPI.Isend(send_bufs[D][S], comm; dest=nbrank)
                send_reqs[D][S] = MPI.Isend(send_bufs[D][S], comm,send_reqs[D][S]; dest=nbrank,tag=tag_send(Val(S)))

            recv_ready = false
            send_ready = false

    # test send and receive requests, initiating device-to-device copy
    # to the receive buffer if the receive is complete
            while !(recv_ready && send_ready)
                    if MPI.Test(recv_reqs[D][S]) && !recv_ready
                        recv_view = get_recv_view(Side(S), Dim(D), halo.data, h)
                        copyto!(recv_view, recv_bufs[D][S])
                        recv_ready = true
                    end
                    
                send_ready= MPI.Test(send_reqs[D][S])
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


@inline tag_rec(::Val{1})=2
@inline tag_rec(::Val{2})=1
@inline tag_send(::Val{1})=1
@inline tag_send(::Val{2})=2

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
        # Post non-blocking recv and send
        recv_reqs[D][S] = MPI.Irecv!(recv_bufs[D][S], comm; source=nbrank)
        send_reqs[D][S] = MPI.Isend(send_bufs[D][S], comm; dest=nbrank)
        # Wait for the receive to complete, then copy buffer into halo region
        MPI.Wait(recv_reqs[D][S])
        recv_view = get_recv_view(side, dim, halo.data, h)
        copyto!(recv_view, recv_bufs[D][S])
        # Wait for send completion
        MPI.Wait(send_reqs[D][S])
            
    end
    return 1 
end

function halo_exchange_wait!(halo::HaloArray{T, N, A,H,S, B,C,BCondition}) where {T,N,A,H,S,B,C,BCondition}
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

    #ntuple(Val(N)) do Dprieme
        #D= N - Dprieme + 1
    for D in 1:N
        nbrank = topo.neighbors[D][1]
        if nbrank != MPI.PROC_NULL
            send_view = get_send_view(Side(1), D, halo)
            copyto!(send_bufs[D][1], send_view)
            # Post non-blocking recv and send
            idx = 2*(D - 1) + 1
            #recv_reqs_flat[idx] = MPI.Irecv!(recv_bufs[D][1], comm; source=nbrank)
            #send_reqs_flat[idx] = MPI.Isend(send_bufs[D][1], comm; dest=nbrank)
            recv_reqs_flat[idx]=MPI.Irecv!(recv_bufs[D][1], comm,recv_reqs_flat[idx]; source=nbrank)
            send_reqs_flat[idx]=MPI.Isend(send_bufs[D][1], comm,send_reqs_flat[idx]; dest=nbrank)

        end

        nbrank = topo.neighbors[D][2]
        if nbrank != MPI.PROC_NULL
            send_view = get_send_view(Side(2), D, halo)
            copyto!(send_bufs[D][2], send_view)
            # Post non-blocking recv and send
            idx = 2*(D - 1) + 2
            #recv_reqs_flat[idx] = MPI.Irecv!(recv_bufs[D][2], comm; source=nbrank)
            #send_reqs_flat[idx] = MPI.Isend(send_bufs[D][2], comm; dest=nbrank)
            recv_reqs_flat[idx]=MPI.Irecv!(recv_bufs[D][2], comm,recv_reqs_flat[idx]; source=nbrank)
            send_reqs_flat[idx]=MPI.Isend(send_bufs[D][2], comm,send_reqs_flat[idx]; dest=nbrank)

        end 
    end

    MPI.Waitall(recv_reqs_flat)

    #ntuple(Val(N)) do Dprieme
    #    D= N - Dprieme + 1
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

    # Step 3: Wait for all sends

    MPI.Waitall(send_reqs_flat)
    
    return nothing
end






function halo_exchange_waitall_unsafe!(halo::HaloArray{T, N, A,H,S, B,C,BCondition}) where {T,N,A,H,S,B,C,BCondition}

    h= halo_width(halo)
    comm = halo.topology.cart_comm
    topo = halo.topology

    unsafe_rec_req = halo.comm_state.unsafe_recv_reqs
    unsafe_send_req = halo.comm_state.unsafe_send_reqs
    rec_bufs = halo.receive_bufs
    send_bufs = halo.send_bufs
    
    rec_state = (unsafe_rec_req, rec_bufs)
    send_state = (unsafe_send_req, send_bufs)
    #ntuple(Val(N)) do Dprieme
        #D= N - Dprieme + 1
   
#this is the left side
        for D in 1:N
            
            @inbounds nbrank = topo.neighbors[D][1]
            if nbrank != MPI.PROC_NULL
                #send_view = get_send_view(Side(s), Dim(D), halo.data, h)
                send_view = get_send_view(Side{1}(), D, halo)
                copyto!(send_bufs[D][1], send_view)
                # Post non-blocking recv and send
                idx = 2*(D - 1) + 1
                GC.@preserve rec_state MPI.Irecv!(rec_bufs[D][1], comm, unsafe_rec_req[idx]; source=nbrank,tag=2)
                GC.@preserve send_state MPI.Isend(send_bufs[D][1], comm, unsafe_send_req[idx]; dest=nbrank,tag=1)
                
            end
            
        end

        @inbounds for D in 1:N
            nbrank = topo.neighbors[D][2]
            if nbrank != MPI.PROC_NULL
                #send_view = get_send_view(Side(s), Dim(D), halo.data, h)
                send_view = get_send_view(Side{2}(), D, halo)
                copyto!(send_bufs[D][2], send_view)
                # Post non-blocking recv and send
                idx = 2*(D - 1) + 2
                GC.@preserve rec_state MPI.Irecv!(rec_bufs[D][2], comm, unsafe_rec_req[idx]; source=nbrank,tag=1)
                GC.@preserve send_state MPI.Isend(send_bufs[D][2], comm, unsafe_send_req[idx]; dest=nbrank,tag=2)
            end
        end


    GC.@preserve rec_state MPI.Waitall(unsafe_rec_req)

    #ntuple(Val(N)) do Dprieme
    #    D= N - Dprieme + 1
#ntuple(Val(N)) do D
        #ntuple(Val(2)) do s
        @inbounds for D=1:N
            nbrank = topo.neighbors[D][1]
            if nbrank != MPI.PROC_NULL
                #recv_view = get_recv_view(Side(s), Dim(D), halo.data, h)
                recv_view = get_recv_view(Side{1}(), D, halo)
                copyto!(recv_view, rec_bufs[D][1])
            end
       end

        @inbounds for D=1:N
            nbrank = topo.neighbors[D][2]
            if nbrank != MPI.PROC_NULL
                #recv_view = get_recv_view(Side(s), Dim(D), halo.data, h)
                recv_view = get_recv_view(Side{2}(),D, halo)
                copyto!(recv_view, rec_bufs[D][2])
            end
       end


    # Step 3: Wait for all sends

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
        # Start non-blocking receive from neighbor
        #recv_reqs[D][S] = MPI.Irecv!(recv_bufs[D][S], comm; source=nbrank)
        recv_reqs[D][S] = MPI.Irecv!(recv_bufs[D][S], comm,recv_reqs[D][S]; source=nbrank,tag=tag_rec(Val(S)))

        # Prepare send buffer with data from halo.data
        send_view = get_send_view(side, dim, halo)
        copyto!(send_bufs[D][S], send_view)

        # Start non-blocking send to neighbor
        #send_reqs[D][S] = MPI.Isend(send_bufs[D][S], comm; dest=nbrank)
        send_reqs[D][S] = MPI.Isend(send_bufs[D][S], comm,send_reqs[D][S]; dest=nbrank,tag=tag_send(Val(S)))
        # Note: we do NOT wait here — user should call wait or test later
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
        # Start non-blocking receive from neighbor
        #recv_reqs[D][S] = MPI.Irecv!(recv_bufs[D][S], comm; source=nbrank)
        recv_reqs[D][S] = MPI.Irecv!(recv_bufs[D][S], comm,recv_reqs[D][S]; source=nbrank, tag=tag_rec(Val(S)))

        # Prepare send buffer with data from halo.data
        send_view = get_send_view(side, D, halo)
        copyto!(send_bufs[D][S], send_view)

        # Start non-blocking send to neighbor
        #send_reqs[D][S] = MPI.Isend(send_bufs[D][S], comm; dest=nbrank)
        send_reqs[D][S] = MPI.Isend(send_bufs[D][S], comm,send_reqs[D][S]; dest=nbrank,tag=tag_send(Val(S)))
        # Note: we do NOT wait here — user should call wait or test later
    end
   
end

function start_halo_exchange_async!(halo::HaloArray{T, N, A,H,S, B,C,BCondition}) where {T,N,A,H,S,B,C,BCondition}
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
        # Wait for receive to complete, then copy into halo region
        MPI.Wait(recv_reqs[D][S])
        recv_view = get_recv_view(side, dim, halo)
        copyto!(recv_view, recv_bufs[D][S])
        # Wait for send completion
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
        # Wait for receive to complete, then copy into halo region
        MPI.Wait(recv_reqs[D][S])
        recv_view = get_recv_view(side, D, halo)
        copyto!(recv_view, recv_bufs[D][S])
        # Wait for send completion
        MPI.Wait(send_reqs[D][S])
    end
    
end

function end_halo_exchange_wait!(halo::HaloArray{T, N, A,H,S, B,C,BCondition}) where {T,N,A,H,S,B,C,BCondition}
    for D in 1:N
    
            halo_exchange_async_wait!(halo, Side{1}(),D)
            halo_exchange_async_wait!(halo, Side{2}(),D)

    end
    return nothing
end

function halo_exchange_async!(halo::HaloArray{T, N, A,H,S, B,C,BCondition}) where {T,N,A,H,S,B,C,BCondition}
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
        # Start non-blocking receive from neighbor
        #recv_reqs[D][S] = MPI.Irecv!(recv_bufs[D][S], comm; source=nbrank)
        GC.@preserve rec_state  MPI.Irecv!(recv_bufs[D][S], comm,recv_reqs[D][S]; source=nbrank,tag=tag_rec(Val(S)) )

        # Prepare send buffer with data from halo.data
        send_view = get_send_view(side, D, halo)
        copyto!(send_bufs[D][S], send_view)

        # Start non-blocking send to neighbor
        #send_reqs[D][S] = MPI.Isend(send_bufs[D][S], comm; dest=nbrank)
        GC.@preserve send_state  MPI.Isend(send_bufs[D][S], comm,send_reqs[D][S]; dest=nbrank,tag= tag_send(Val(S)) )
        # Note: we do NOT wait here — user should call wait or test later
    end
   
end

function start_halo_exchange_async_unsafe!(halo::HaloArray{T, N, A,H,S, B,C,BCondition}) where {T,N,A,H,S,B,C,BCondition}
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
        # Wait for receive to complete, then copy into halo region
        GC.@preserve rec_state MPI.Wait(recv_reqs[D][S])
        recv_view = get_recv_view(side, D, halo)
        copyto!(recv_view, recv_bufs[D][S])
        # Wait for send completion
        GC.@preserve send_state MPI.Wait(send_reqs[D][S])
    end
    
end

function end_halo_exchange_async_wait_unsafe!(halo::HaloArray{T, N, A,H,S, B,C,BCondition}) where {T,N,A,H,S,B,C,BCondition}
    for D in 1:N
    
            halo_exchange_async_wait_unsafe!(halo, Side{1}(),D)
            halo_exchange_async_wait_unsafe!(halo, Side{2}(),D)
        
    end
    return nothing
end

function halo_exchange_async_unsafe!(halo::HaloArray{T, N, A,H,S, B,C,BCondition}) where {T,N,A,H,S,B,C,BCondition}
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