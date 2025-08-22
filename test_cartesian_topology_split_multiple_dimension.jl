using MPI
using Base.Iterators
include("/Users/eduardogrossi/mpistuff/cartesian_topology.jl")

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

# --- funzioni multi-dimensione definite direttamente nel test ---
# coords::NTuple{N,Int}, dims::NTuple{N,Int}
# dims_to_remove: array di indici 1-based da rimuovere
function coords_to_color_multi(coords::NTuple{N,Int}, dims::NTuple{N,Int}, dims_to_remove::AbstractVector{Int}) where {N}
    rem = [i for i in 1:N if !(i in dims_to_remove)]
    coords_list = [coords[i] for i in rem]
    dims_list   = [dims[i]   for i in rem]
    color = 0
    mul = 1
    for (c,d) in zip(coords_list, dims_list)
        color += c * mul
        mul *= d
    end
    return color
end

function subcomm_for_slices(cart::CartesianTopology{N}, dims_to_reduce::AbstractVector{Int}) where {N}
    rank = cart.global_rank
    coords = cart.cart_coords
    color = coords_to_color_multi(coords, cart.dims, dims_to_reduce)
    # key: ordina i ranks dentro la slice combinando le coords sulle dimensioni rimosse
    key = 0
    mul = 1
    for i in dims_to_reduce
        key += coords[i] * mul
        mul *= cart.dims[i]
    end
    sub_comm = MPI.Comm_split(cart.cart_comm, color, key)
    subrank = (sub_comm == MPI.COMM_NULL) ? -1 : MPI.Comm_rank(sub_comm)
    return (sub_comm, coords, subrank)
end

function root_topology_multi(cart::CartesianTopology{N}, dims_to_reduce::AbstractVector{Int}; root_coord::Int = 0) where {N}
    coords = cart.cart_coords
    is_root = all(i -> coords[i] == root_coord, dims_to_reduce)
    color = is_root ? 0 : nothing
    root_comm = MPI.Comm_split(cart.cart_comm, color, cart.global_rank)

    rem = [i for i in 1:N if !(i in dims_to_reduce)]
    new_dims = Tuple(cart.dims[i] for i in rem)
    new_periods = Tuple(cart.periodic_boundary_condition[i] for i in rem)

    if !is_root || root_comm == MPI.COMM_NULL
        return CartesianTopology(root_comm, new_dims; periodic=new_periods, active=false)
    else
        return CartesianTopology(root_comm, new_dims; periodic=new_periods)
    end
end

# ---------- test ----------
# configurazione
N_total = 3                            # dimensione totale del grid
dims_to_remove = [1, 3]                 # rimuovere le dim 2 e 4 (1-based)
topo = CartesianTopology(comm, ntuple(_->0, Val(N_total)); periodic=ntuple(_->true, Val(N_total)))

if !isactive(topo)
    error("CartesianTopology inattiva su rank $rank")
end

coords = topo.cart_coords
dims = topo.dims

# creiamo il sub-communicator che raggruppa i ranks uguali su tutte le dim
# eccetto quelle in dims_to_remove
(sub_comm, s_coords, subrank) = subcomm_for_slices(topo, dims_to_remove)

# valore locale da ridurre: usiamo rank+1 per semplicità
local_value = rank + 1

# riduzione (somma) dentro la slice; root = 0 (ordering key costruito nel split)
sum_on_root = MPI.Reduce(local_value, MPI.SUM, 0, sub_comm)

# calcola valore atteso enumerando tutte le combinazioni sulle dimensioni rimosse

ranges = (0:(topo.dims[i]-1) for i in dims_to_remove)
expected = 0
for ks in Iterators.product(ranges...)
    coords_k = collect(coords)
    for (j, i) in enumerate(dims_to_remove)
        coords_k[i] = ks[j]
    end
    global expected += MPI.Cart_rank(topo.cart_comm, Tuple(coords_k)) + 1
end

# controllo: solo il root della slice (subrank==0) ha sum_on_root definito e deve matchare expected
if sub_comm != MPI.COMM_NULL && subrank == 0
    if sum_on_root != expected
        error("Mismatch riduzione sulla slice (global rank $rank): got=$sum_on_root expected=$expected")
    end
end

# libera sub_comm se allocato
if sub_comm != MPI.COMM_NULL
    MPI.free(sub_comm)
end

# costruzione della root-topology che rimuove le dims specificate
root_topo = root_topology_multi(topo, dims_to_remove; root_coord=0)
is_root = all(i->coords[i]==0, dims_to_remove)

# verifica stato della root_topo
if is_root
    if !isactive(root_topo)
        error("Root topo attesa attiva su rank $rank ma è inattiva")
    end
    if root_topo.dims != Tuple(topo.dims[i] for i in 1:N_total if !(i in dims_to_remove))
        error("Root topo dims non corrispondono su rank $rank")
    end
else
    if isactive(root_topo)
        error("Root topo non attesa attiva su non-root rank $rank")
    end
end

# raccogli e stampa un breve sommario su rank 0
summary = MPI.gather((rank, coords, is_root, isactive(root_topo) ? root_topo.cart_coords : nothing), comm)
if rank == 0
    println("Sommario test tst_cartesiantopology (dims_to_remove = $dims_to_remove):")
    for (r, c, rootflag, rcoords) in summary
        println("Rank $r: coords=$c is_root=$rootflag root_coords=$(rcoords === nothing ? "-" : rcoords)")
    end
    println("\nTest completato (nessun errore = OK).")
end

MPI