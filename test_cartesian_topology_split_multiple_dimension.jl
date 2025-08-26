using MPI
using Base.Iterators
include("cartesian_topology.jl") 
include("haloarray.jl")
include("haloarrays.jl")
include("boundary.jl")        # <<-- boundary prima
include("interior_broadcast.jl")
include("halo_exchange.jl")
include("meybehaloarray.jl")

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)


"""
    reduce_haloarray_dims(ha::HaloArray, dims_to_remove::AbstractVector{Int}; root_coord=0)

Reduce `ha` over the dimensions in `dims_to_remove` (1-based). Assumes the local interior
size along each dimension is 1 (i.e. each process owns exactly one interior cell),
so the per-slice reduction produces a single scalar placed on the corresponding root
process of the reduced topology.

Returns a MaybeHaloArray wrapping the new lower-dimensional HaloArray (active only on
the chosen root coordinates).
"""
function reduce_haloarray_dims(op,ha::HaloArray{T,N,A,Halo,B,BCondition}, dims_to_remove::AbstractVector{Int}; root_coord::Int=0) where {T,N,A,Halo,B,BCondition}
    topo = ha.topology

    # split communicator grouping processes that share coords on the kept dims
    (sub_comm, coords, subrank) = subcomm_for_slices(topo, dims_to_remove)

    mpi_op=MPI.Op(op, T; iscommutative=true)

    dimension_to_reduce = Tuple(dims_to_remove)
    local_value = dropdims(reduce(op,ha.data,dims=dimension_to_reduce),dims=dimension_to_reduce)
    # perform reduction (sum) inside the sub-communicator; result only valid on subrank==0

    
    sum_on_root = MPI.Reduce(local_value, mpi_op, sub_comm, root=0)

    # build the root (reduced) topology
    root_topo = root_topology_multi(topo, dims_to_remove; root_coord=root_coord)
    rem = Tuple(i for i in 1:N if !(i in dims_to_remove))
    new_boundary=Tuple(ha.boundary_condition[i] for i in 1:N if !(i in dims_to_remove))
    
    M = length(rem)

    # construct new HaloArray for reduced topology (halo width preserved)
    new_ha = HaloArray{T, M, typeof(local_value) , Halo}(undef, new_boundary)

    # if this process is the root of its sub-comm, place reduced scalar into the interior cell
    if isactive(root_topo)
        sizes = size(sum_on_root)

        new_ranges = ntuple(i -> halo_width(new_ha) + 1:sizes[i], Val(max(1,M)))

        new_ha.data = sum_on_root
        new_ha.topology = root_topo
        new_ha.receive_bufs = make_recv_buffers(new_ha.data, Halo)
        new_ha.send_bufs = make_send_buffers(new_ha.data, Halo)
    end

    # free sub_comm if allocated
    if sub_comm != MPI.COMM_NULL
        MPI.free(sub_comm)
    end

    return MaybeHaloArray(new_ha)
end

# ---------- tests ----------
# small test scenario: N_total dims, local interior size = 1 per dim (each process holds one cell)
N_total = 5
topo = CartesianTopology(comm, ntuple(_->0, Val(N_total)); periodic=ntuple(_->true, Val(N_total)))
if !isactive(topo)
    error("CartesianTopology inactive on rank $rank")
end

# build a HaloArray with one interior cell per process, halo width 0 for simplicity
ha = HaloArray(Float64, ntuple(_->1, Val(N_total)), 0, topo, ntuple(_->(Periodic(), Periodic()), Val(N_total)))

# fill interior with a value depending on global rank for predictability
center = ntuple(i->halo_width(ha)+1, Val(N_total))
ha.data[center...] = rank + 1

# try multiple reduction patterns
tests = [
    [1],           # reduce first dim
    [2],           # reduce second dim
    [1, 3],        # reduce dims 1 and 3 (like original test)
]

results = Dict{Tuple{Vararg{Int}}, Any}()

for dims_to_remove in tests
    maybe_reduced = reduce_haloarray_dims(+,ha, dims_to_remove; root_coord=0)

    # compute expected sum for this process's kept coords (only meaningful on root_topo active procs)
    coords = topo.cart_coords
    rem = [i for i in 1:N_total if !(i in dims_to_remove)]
    # enumerate all combinations of removed-dim coordinates and sum (rank+1) of the corresponding global ranks
    ranges = (0:(topo.dims[i]-1) for i in dims_to_remove)
    expected = 0
    for ks in Iterators.product(ranges...)
        coords_k = collect(coords)
        for (j, i) in enumerate(dims_to_remove)
            coords_k[i] = ks[j]
        end
        expected += MPI.Cart_rank(topo.cart_comm, Tuple(coords_k)) + 1
    end

    # check result on the processes that host the reduced HaloArray (active roots)
    if isactive(maybe_reduced)
        reduced = unwrap(maybe_reduced)
        # reduced has interior size ones per remaining dim; check its center value
        new_center = ntuple(i->halo_width(reduced)+1, Val(ndims(reduced)))
        got = reduced.data[new_center...]
        ok = got == expected
        show("Rank $rank: got=$got expected=$expected ok=$ok")
    else
        ok = true  # non-root ranks are expected to be inactive; nothing to check here
        got = nothing
    end

    results[Tuple(dims_to_remove)] = (ok=ok, got=got, expected=expected, isactive=isactive(maybe_reduced))
end

# gather brief summary to rank 0 and print
summary = MPI.gather((rank, topo.cart_coords, results), comm)
if rank == 0
    println("Reduction tests summary:")
    for (r, c, resmap) in summary
        println("Rank $r coords=$c")
        for (k, v) in resmap
            println("  remove=$(collect(k)) active=$(v.isactive) got=$(v.got) expected=$(v.expected) ok=$(v.ok)")
        end
    end
    println("Test completed.")
end

# =========================
# Additional test: multicell interior >1 and halo >0
# =========================
if rank == 0
    println("Running multicell + halo reduction test...")
end

N_total = 3
topo = CartesianTopology(comm, ntuple(_->0, Val(N_total)); periodic=ntuple(_->true, Val(N_total)))

# setup a new halo array with local interior >1 and halo >0
local_inner = (2, 2, 1)      # interior per-process
halo = 1
# boundary condition: Periodic for all dims
bc2 = ntuple(_ -> (Periodic(), Periodic()), Val(N_total))

ha2 = HaloArray(Float64, local_inner, halo, topo; boundary_condition = bc2)

# fill interior deterministically so expected sums are computable
function linear_index_tuple(li, sizes)
    idx = 0
    mul = 1
    for i in 1:length(sizes)
        idx += (li[i]-1) * mul
        mul *= sizes[i]
    end
    return idx + 1
end

inds_interior = ntuple(d -> (halo+1):(halo+local_inner[d]), Val(3))
for I in Iterators.product((collect(r) for r in inds_interior)...)
    local_idx = ntuple(d -> I[d] - halo, Val(3))
    ha2.data[I...] = (rank + 1) * 1000 + linear_index_tuple(local_idx, local_inner)
end

# local reduction helper
function local_reduce_block(op,block, dims_to_remove)
    reduced = dropdims(reduce(op,block, dims=dims_to_remove), dims=dims_to_remove)
    return reduced
end

tests2 = [
    [1],        # reduce dim 1
    [2],        # reduce dim 2
    [1,3],      # reduce dims 1 and 3
]

results2 = Dict{Tuple{Vararg{Int}}, Any}()

for dims_to_remove in tests2


    maybe_new = reduce_haloarray_dims(+,ha2, dims_to_remove; root_coord=0)


    # compute expected by summing contributions from all ranks in the slice
    expected = nothing
    ranges = (0:(topo.dims[i]-1) for i in dims_to_remove)
    for ks in Iterators.product(ranges...)
        coords_k = collect(topo.cart_coords)
        for (j, i) in enumerate(dims_to_remove)
            coords_k[i] = ks[j]
        end
        gr = MPI.Cart_rank(topo.cart_comm, Tuple(coords_k))
        local_block = Array{Float64}(undef, local_inner...)
        for li in Iterators.product((1:local_inner[d] for d in 1:length(local_inner))...)
            val = (gr + 1) * 1000 + linear_index_tuple(li, local_inner)
            local_block[Tuple(li)...] = val
        end
        lr = local_reduce_block(+,local_block, Tuple(dims_to_remove))
        if expected === nothing
            expected = copy(lr)
        else
            expected .+= lr
        end
    end

    if isactive(maybe_new)
        newh = unwrap(maybe_new)
        interior = interior_view(newh)
        got = Array(interior)
        ok = all(got .== expected)
        println("Rank $rank coords=$(topo.cart_coords) remove=$(dims_to_remove) ok=$ok")
    else
        
        got = nothing
        ok = true
    end

    results2[Tuple(dims_to_remove)] = (ok=ok, got=got, expected=expected, isactive=isactive(maybe_new))

end

summary2 = MPI.gather((rank, topo.cart_coords, results2), comm)
if rank == 0
    println("Multicell + halo reduction tests summary:")
    for (r, c, resmap) in summary2
        println("Rank $r coords=$c")
        for (k, v) in resmap
            println("  remove=$(collect(k)) active=$(v.isactive) ok=$(v.ok)")
        end
    end
end


MPI.Barrier(comm)

