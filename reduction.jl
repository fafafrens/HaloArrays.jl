# We force specialisation on each function to avoid (tiny) allocations.
#reduce
# Note that, for mapreduce, we can assume that the operation is commutative,
# which allows MPI to freely reorder operations.
#
# We also define mapfoldl (and mapfoldr) for completeness, even though the global
# operations are not strictly performed from left to right (or from right to
# left), since each process locally reduces first.
for (func, commutative) in [:mapreduce => true, :mapfoldl => false, :mapfoldr => false]
    @eval function Base.$func(
            f::F, op::OP, halo::HaloArray, etc::Vararg{HaloArray}; kws...,
        ) where {F, OP}
        #foreach(v -> _check_compatible_arrays(u, v), etc)
        comm = get_comm(halo)
        #we create a tuple of view 
        ups = map(interior_view, (halo, etc...))
        rlocal = $func(f, op, ups...; kws...)
        op_mpi = MPI.Op(op, typeof(rlocal); iscommutative = $commutative)
        MPI.Allreduce(rlocal, op_mpi, comm)
    end

    # Make things work with zip(u::PencilArray, v::PencilArray, ...)
    @eval function Base.$func(
            f::F, op::OP, z::Iterators.Zip{<:Tuple{Vararg{HaloArray}}}; kws...,
        ) where {F, OP}
        g(args...) = f(args)
        $func(g, op, z.is...; kws...)
    end
end

function Base.any(f::F, u::HaloArray) where {F }
    xlocal = any(f, interior_view(u)) :: Bool
    MPI.Allreduce(xlocal, |, get_comm(u))
end

function Base.all(f::F, u::HaloArray) where {F }
    xlocal = all(f, interior_view(u)) :: Bool
    MPI.Allreduce(xlocal, &, get_comm(u))
end



for (func, commutative) in [:mapreduce => true, :mapfoldl => false, :mapfoldr => false]
    @eval function Base.$func(
            f::F, op::OP, halo::MultiHaloArray, etc::Vararg{MultiHaloArray}; kws...,
        ) where {F, OP}
        
        # Get names (field keys) and bundle all inputs together
        
        all_arrays = (to_tuple(halo), to_tuple.(etc)...)
        N=n_field(halo)

        # Compute per-field reduction to scalars
        per_field_results = map(1:N) do idx
            # Extract the field arrays for this index
            field_arrays = map(all_arrays) do arrs
                arrs[idx]
            end
            $func(f, op, field_arrays...; kws...)
        end

        # Final reduction over all fields
        return reduce(op, per_field_results; kws...)
    end

    # Make things work with zip(u::PencilArray, v::PencilArray, ...)
    @eval function Base.$func(
            f::F, op::OP, z::Iterators.Zip{<:Tuple{Vararg{MultiHaloArray}}}; kws...,
        ) where {F, OP}
        g(args...) = f(args)
        $func(g, op, z.is...; kws...)
    end
end


function Base.all(f::F, mha::MultiHaloArray) where {F}
    field_results = map(values(mha.arrays)) do field
        all(f, field)
    end
    return all(field_results)
end

function Base.any(f::F, mha::MultiHaloArray) where {F}
    field_results = map(values(mha.arrays)) do field
        any(f, field)
    end
    return any(field_results)
end


"""
    mapreduce_haloarray_dims(f,op,ha::HaloArray, dims; root_coord=0)

Reduce `ha` over the dimensions in `dims_to_remove` (1-based). Assumes the local interior
size along each dimension is 1 (i.e. each process owns exactly one interior cell),
so the per-slice reduction produces a single scalar placed on the corresponding root
process of the reduced topology.

Returns a MaybeHaloArray wrapping the new lower-dimensional HaloArray (active only on
the chosen root coordinates).
"""
function mapreduce_haloarray_dims(f,op,ha::HaloArray{T,N,A,Halo,B,BCondition}, dims) where {T,N,A,Halo,B,BCondition}
    topo = ha.topology
    root_coord=0
    dims_to_remove = Tuple(dims)
    # split communicator grouping processes that share coords on the kept dims
    (sub_comm, coords, subrank) = subcomm_for_slices(topo, dims_to_remove)

    mpi_op=MPI.Op(op, T; iscommutative=true)

    dimension_to_reduce = Tuple(dims_to_remove)
    local_value = dropdims(mapreduce(f,op,ha.data,dims=dimension_to_reduce),dims=dimension_to_reduce)
    # perform reduction (sum) inside the sub-communicator; result only valid on subrank==0


    sum_on_root = MPI.Reduce(local_value, mpi_op, sub_comm, root=root_coord)

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

"""
    reduce_mhaloarray_dims(op, mha::MultiHaloArray, dims; root_coord=0)

Riduce `mha` campo-per-campo lungo `dims` usando `mapreduce_haloarray_dims(identity, op, ...)`.
Restituisce un `MaybeHaloArray(MultiHaloArray(...))` attivo solo sui root delle slice ridotte.
Se nessun campo è root sulla slice (tutti inactive) ritorna MaybeHaloArray(mha, false).
Se alcuni campi risultano active e altri no -> errore (incoerenza).
"""
function mapreduce_mhaloarray_dims(f, op, mha::MultiHaloArray, dims)
    
    names = keys(mha.arrays)

    list_of_maybe = map_over_field( mha) do field 
        mapreduce_haloarray_dims(f, op, field, dims)
    end
    # unwrap e costruisci NamedTuple dei campi ridotti
    
    nt = NamedTuple{names}(list_of_maybe)

    return MultiHaloArray(nt)   # constructor decide active in base ai campi
end


