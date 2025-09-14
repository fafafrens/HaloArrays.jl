using MPI
using Base.Broadcast: broadcasted
using Test
using HaloArrays

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

println("Rank $rank/$nprocs: starting MaybeHaloArray broadcast test")

# --- topology: 1D process grid (nprocs)
N_total = 1
topo = CartesianTopology(comm, ntuple(_->0, Val(N_total)); periodic=ntuple(_->true, Val(N_total)))
@assert isactive(topo)

# common params
local_inner = ntuple(_->1, Val(N_total))
halo = 0
bc_tuple = ntuple(_->(Periodic(), Periodic()), Val(N_total))
center = ntuple(i->halo+1, Val(N_total))

# -------------------------
# Test A: single-field HaloArray wrapped in MaybeHaloArray
# -------------------------
ha = HaloArray(Float64, local_inner, halo, topo; boundary_condition=bc_tuple)
ha.data[center...] = rank + 1
mha = MaybeHaloArray(ha)

# dot-syntax (idiomatic): materializza in res
res = mha .+ 10
dest = similar(mha)
dest .= mha .+ 10

if isactive(res)
    newha = unwrap(res)
    val = newha.data[center...]
    expected = (rank + 1) + 10
    println("Rank $rank: single-field res active, val=$val expected=$expected ok=$(val==expected)")
else
    println("Rank $rank: single-field res inactive (unexpected)")
end

# -------------------------
# Test B: MultiHaloArray inner (two fields u,v) wrapped in MaybeHaloArray
# -------------------------
ha_u = HaloArray(Float64, local_inner, halo, topo; boundary_condition=bc_tuple)
ha_v = HaloArray(Float64, local_inner, halo, topo; boundary_condition=bc_tuple)
ha_u.data[center...] = rank + 1
ha_v.data[center...] = (rank + 1) * 10

mha_fields = (; u = ha_u, v = ha_v)
multi = MultiHaloArray(mha_fields)
maybe_multi = MaybeHaloArray(multi)

# broadcast scalar add to all fields via dot-syntax
res_multi = maybe_multi .+ 3
dest_multi = similar(maybe_multi)
dest_multi .= maybe_multi .+ 3

if isactive(res_multi)
    outm = unwrap(res_multi)
    val_u = outm.arrays.u.data[center...]
    val_v = outm.arrays.v.data[center...]
    println("Rank $rank: multi u=$(val_u) expected=$( (rank+1)+3 ) ok=$(val_u == (rank+1)+3 )")
    println("Rank $rank: multi v=$(val_v) expected=$( (rank+1)*10+3 ) ok=$(val_v == (rank+1)*10+3 )")
else
    println("Rank $rank: multi result inactive (unexpected)")
end

# -------------------------
# Test C: reduce HaloArray / MultiHaloArray along dim=1 on a 2D interior, then broadcast the Maybe result
# -------------------------
println("Rank $rank: starting reduction+broadcast tests (2D example)")

# prepare a 2D HaloArray with interior >1 on dim1 so reduction over dim1 is meaningful
topo2 = CartesianTopology(comm, ntuple(_->0, Val(2)); periodic=ntuple(_->true, Val(2)))
halo2 = 0
local_inner_red = (4, 3)   # interior shape (n1,n2)
bc_tuple2 = ntuple(_->(Periodic(),Periodic()), 2)

ha_red = HaloArray(Float64, local_inner_red, halo2, topo2; boundary_condition=bc_tuple2)
# fill interior with predictable values: use the global rank so max across processes is trivial
for j in 1:local_inner_red[2], i in 1:local_inner_red[1]
    idx = (halo2 + i, halo2 + j)
    ha_red.data[idx...] = rank   # <- semplice e prevedibile
end

# usa max come operatore di riduzione (prevedibile)
op = max   # reduction operator (max)

# riduzione lungo la dimensione 1 -> risultato ha shape (n2,)
maybe_red = mapreduce_haloarray_dims(identity, op, ha_red, (1,))

# broadcast on the MaybeHaloArray result (dot-syntax)
res_red = maybe_red .+ 5
if isactive(res_red)
    out = unwrap(res_red)
    iv = interior_view(out)   # dovrebbe essere un Array 1D (o NamedTuple a seconda del caso)
    println("Rank $rank: HaloArray 2D->1D reduction active, interior_view = $iv")
else
    println("Rank $rank: HaloArray reduction inactive on this rank")
end

# MultiHaloArray: due campi 2D
ha_u = HaloArray(Float64, local_inner_red, halo2, topo2; boundary_condition=bc_tuple2)
ha_v = HaloArray(Float64, local_inner_red, halo2, topo2; boundary_condition=bc_tuple2)
for j in 1:local_inner_red[2], i in 1:local_inner_red[1]
    idx = (halo2 + i, halo2 + j)
    ha_u.data[idx...] = rank*10 + i + j
    ha_v.data[idx...] = rank*100 + i + 2*j
end
mfields = (; u = ha_u, v = ha_v)
multi = MultiHaloArray(mfields)

# reduce MultiHaloArray field-by-field along dim 1
maybe_multi_red = mapreduce_mhaloarray_dims(identity, op, multi, (1,))

res_mm = maybe_multi_red .+ 2
if isactive(res_mm)
    outm = res_mm.arrays
    iu = interior_view(getdata(outm.u))   # dovrebbe essere 1D array lungo dim2
    ivv = interior_view(getdata(outm.v))
    println("Rank $rank: MultiHaloArray reduction active, u interior = $iu, v interior = $ivv")
else
    println("Rank $rank: MultiHaloArray reduction inactive on this rank")
end

