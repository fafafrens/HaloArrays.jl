

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