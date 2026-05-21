"""
Verifica che le dimensioni dell'array globale siano divisibili per il numero di processi
"""
function check_scatter_dimensions(global_size::NTuple{N,Int}, dims::NTuple{N,Int}) where N
    for i in 1:N
        if global_size[i] % dims[i] != 0
            error("La dimensione globale $(global_size[i]) non è divisibile per $(dims[i]) processi nella dimensione $i")
        end
    end
    return true
end




function scatter_haloarray(global_array::Array{T,N}, comm::MPI.Comm, topo::CartesianTopology, h::Int; root::Int=0) where {T,N}
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)
    dims = topo.dims
    
    # Verifica dimensioni prima di procedere
    if rank == root
        try
            check_scatter_dimensions(size(global_array), dims)
        catch e
            # Propaga l'errore a tutti i processi
            MPI.Abort(comm, 1)
            error("Scatter fallito: $e")
        end
    end
    
    # Sincronizza tutti i processi dopo il controllo
    MPI.Barrier(comm)
    
    # Calcola le dimensioni locali
    local_size = ntuple(i -> div(size(global_array, i), dims[i]), Val(N))
    local_len = prod(local_size)
    
    
    # Buffer per ricevere i dati locali
    sendbuf = Vector{T}(undef, local_len * nproc)
    recvbuf = Vector{T}(undef, local_len)
    
    if rank == root
        # Prepara il buffer di invio
        for r in 0:nproc-1
            coords_r = MPI.Cart_coords(comm, r) |> Tuple
            offset = ntuple(i -> coords_r[i] * local_size[i], Val(N))
            inds = ntuple(i -> (offset[i]+1):(offset[i]+local_size[i]), Val(N))
            
            # Estrai il blocco locale e linearizzalo
            flat_offset = r * local_len + 1
            sendbuf[flat_offset:flat_offset+local_len-1] = vec(view(global_array, inds...))
        end
    end
    
    # Scatter i dati a tutti i processi
    MPI.Scatter!(sendbuf, local_len, recvbuf, local_len, root, comm)
    
    # Crea l'HaloArray locale
    bc = ntuple(i -> (Periodic(), Periodic()), Val(N))
    halo = HaloArray(T, local_size, h, topo; boundary_condition=bc)
    
    # Riempi la regione interna
    interior = reshape(recvbuf, local_size)
    interior_view(halo) .= interior
    
    return halo
end