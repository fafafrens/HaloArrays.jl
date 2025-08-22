using MPI
include("/Users/eduardogrossi/mpistuff/cartesian_topology.jl")

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

# configura topologia e dimensione di riduzione (1-based)
N = 3
dim_to_reduce = 2
topo = CartesianTopology(comm, ntuple(_->0, Val(N)); periodic=ntuple(_->true, Val(N)))

# esempio: ogni processo ha un array "halo" lungo la dimensione ridotta
# qui usiamo vettori 1D per semplicità; nella pratica può essere una porzione di array ND
local_halo = fill(rank+1, 8)  # buffer di esempio (tutti gli elementi uguali per test)

# costruiamo subcomm che raggruppa ranks con stesse coords su tutte le altre dimensioni
(sub_comm, coords, subrank) = subcomm_for_slices(topo, [dim_to_reduce])

if sub_comm != MPI.COMM_NULL
    # scelta del root nella slice (ad es. chi ha coordinate ridotta == 0)
    root_subrank = 0

    # Reduzione: somma dei buffer lungo la slice, risultato solo sul root
    if subrank == root_subrank
        recvbuf = zeros(Int, length(local_halo))
    else
        recvbuf = nothing
    end
    recvbuf = MPI.Reduce(local_halo, MPI.SUM, sub_comm; root=root_subrank)

    # Se vuoi che TUTTI nella slice ricevano il risultato, usa Allreduce:
    # allbuf = similar(local_halo)
    # MPI.Allreduce(local_halo, allbuf, MPI.SUM, sub_comm)

    # esempio: il root stampa il risultato aggregato
    msg = nothing
    if subrank == root_subrank
        msg = "Global rank $(rank): reduced halo on slice (coords=$(coords)) = $(recvbuf)"
    end
    # stampa sincronizzata su tutti i rank globali: ogni rank aspetta il suo turno
    MPI.Barrier(comm)
    for r in 0:nprocs-1
        MPI.Barrier(comm)
        if rank == r && sub_comm != MPI.COMM_NULL && subrank == root_subrank && msg !== nothing
            println(msg)
            flush(stdout)
        end
    end
    MPI.Barrier(comm)

    MPI.free(sub_comm)
end

MPI.Barrier(comm)
MPI.Finalize()