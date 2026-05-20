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
        ) where {F<:Function, OP}
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
        ) where {F<:Function, OP}
        g(args...) = f(args)
        $func(g, op, z.is...; kws...)
    end
end

function Base.any(f::F, u::HaloArray) where {F<:Function}
    xlocal = any(f, interior_view(u)) :: Bool
    MPI.Allreduce(xlocal, |, get_comm(u))
end

function Base.all(f::F, u::HaloArray) where {F<:Function}
    xlocal = all(f, interior_view(u)) :: Bool
    MPI.Allreduce(xlocal, &, get_comm(u))
end

function _combine_threaded_reduction(op, values)
    result = first(values)
    for i in 2:length(values)
        result = op(result, values[i])
    end
    return result
end

for func in (:mapreduce, :mapfoldl, :mapfoldr)
    @eval function Base.$func(
            f::F, op::OP, halo::LocalHaloArray, etc::Vararg{LocalHaloArray}; kws...,
        ) where {F<:Function, OP}
        interiors = map(interior_view, (halo, etc...))
        return $func(f, op, interiors...; kws...)
    end

    @eval function Base.$func(
            f::F, op::OP, z::Iterators.Zip{<:Tuple{Vararg{LocalHaloArray}}}; kws...,
        ) where {F<:Function, OP}
        g(args...) = f(args)
        return $func(g, op, z.is...; kws...)
    end

    @eval function Base.$func(
            f::F, op::OP, halo::ThreadedHaloArray, etc::Vararg{ThreadedHaloArray}; kws...,
        ) where {F<:Function, OP}
        tile_results = tmap(1:tile_count(halo)) do tile_id
            interiors = map(h -> interior_view(h, tile_id), (halo, etc...))
            $func(f, op, interiors...; kws...)
        end
        return _combine_threaded_reduction(op, tile_results)
    end

    @eval function Base.$func(
            f::F, op::OP, z::Iterators.Zip{<:Tuple{Vararg{ThreadedHaloArray}}}; kws...,
        ) where {F<:Function, OP}
        g(args...) = f(args)
        return $func(g, op, z.is...; kws...)
    end
end

function Base.any(f::F, u::LocalHaloArray) where {F<:Function}
    return any(f, interior_view(u))
end

function Base.all(f::F, u::LocalHaloArray) where {F<:Function}
    return all(f, interior_view(u))
end

function Base.any(f::F, u::ThreadedHaloArray) where {F<:Function}
    return tmapreduce(tile_id -> any(f, interior_view(u, tile_id)), |, 1:tile_count(u))
end

function Base.all(f::F, u::ThreadedHaloArray) where {F<:Function}
    return tmapreduce(tile_id -> all(f, interior_view(u, tile_id)), &, 1:tile_count(u))
end

for (func, commutative) in [:mapreduce => true, :mapfoldl => false, :mapfoldr => false]
    @eval function Base.$func(
            f::F, op::OP, halo::MultiHaloArray, etc::Vararg{MultiHaloArray}; kws...,
        ) where {F<:Function, OP}
        
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
        ) where {F<:Function, OP}
        g(args...) = f(args)
        $func(g, op, z.is...; kws...)
    end
end


function Base.all(f::F, mha::MultiHaloArray) where {F<:Function}
    field_results = map(values(mha.arrays)) do field
        all(f, field)
    end
    return all(field_results)
end

function Base.any(f::F, mha::MultiHaloArray) where {F<:Function}
    field_results = map(values(mha.arrays)) do field
        any(f, field)
    end
    return any(field_results)
end


"""
    mapreduce_haloarray_dims(f,op,ha::HaloArray, dims; root_coord=0)

Reduce `ha` over the dimensions in `dims_to_remove` (1-based). Assumes the owned interior
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
    dims_to_keep = Tuple(i for i in 1:N if !(i in dims_to_remove))
    M = length(dims_to_keep)
    M == 0 && throw(ArgumentError("Reducing all dimensions to a scalar is not supported by mapreduce_haloarray_dims"))

    # split communicator grouping processes that share coords on the kept dims
    (sub_comm, coords, subrank) = subcomm_for_slices(topo, dims_to_remove)

    mpi_op=MPI.Op(op, T; iscommutative=true)

    dimension_to_reduce = Tuple(dims_to_remove)
    local_value = dropdims(mapreduce(f, op, interior_view(ha), dims=dimension_to_reduce), dims=dimension_to_reduce)

    # perform reduction inside the sub-communicator; result only valid on subrank==0

    sum_on_root = MPI.Reduce(local_value, mpi_op, sub_comm, root=root_coord)

    # build the root (reduced) topology
    root_topo = root_topology_multi(topo, dims_to_remove; root_coord=root_coord)
    new_boundary = ntuple(i -> ha.boundary_condition[dims_to_keep[i]], Val(M))
    reduced_owned_size = size(local_value)
    new_ha = HaloArray(T, reduced_owned_size, Halo, root_topo; boundary_condition=new_boundary)

    # if this process is the root of its sub-comm, place reduced scalar into the interior cell
    if isactive(root_topo)
        interior_view(new_ha) .= sum_on_root
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
    active_states = map(isactive, values(list_of_maybe))
    if any(active_states) && !all(active_states)
        error("Inconsistent active state across reduced MultiHaloArray fields")
    end

    nt = NamedTuple{names}(map(getdata, values(list_of_maybe)))

    return MaybeHaloArray(MultiHaloArray(nt))
end
