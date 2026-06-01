# mapreduce / any / all for HaloArray (MPI) live in mpi_support.jl

function _combine_threaded_reduction(op, result, values)
    for value in values
        result = op(result, value)
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
        ntile = tile_count(halo)
        first_interiors = map(h -> interior_view(h, 1), (halo, etc...))
        first_result = $func(f, op, first_interiors...; kws...)
        ntile == 1 && return first_result

        tile_results = tmap(typeof(first_result), 2:ntile; scheduler=:static) do tile_id
            interiors = map(h -> interior_view(h, tile_id), (halo, etc...))
            $func(f, op, interiors...; kws...)
        end
        return _combine_threaded_reduction(op, first_result, tile_results)
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
    return tmapreduce(tile_id -> any(f, interior_view(u, tile_id)), |, 1:tile_count(u); scheduler=:static)
end

function Base.all(f::F, u::ThreadedHaloArray) where {F<:Function}
    return tmapreduce(tile_id -> all(f, interior_view(u, tile_id)), &, 1:tile_count(u); scheduler=:static)
end

for func in (:mapreduce, :mapfoldl, :mapfoldr)
    @eval function Base.$func(
            f::F, op::OP, halo::MultiHaloArray, etc::Vararg{MultiHaloArray}; kws...,
        ) where {F<:Function, OP}
        
        all_arrays = (to_tuple(halo), to_tuple.(etc)...)
        N=n_field(halo)

        per_field_results = map(1:N) do idx
            field_arrays = map(all_arrays) do arrs
                arrs[idx]
            end
            $func(f, op, field_arrays...; kws...)
        end

        return reduce(op, per_field_results; kws...)
    end

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

for func in (:mapreduce, :mapfoldl, :mapfoldr)
    @eval function Base.$func(
            f::F, op::OP, halo::ArrayOfHaloArray, etc::Vararg{ArrayOfHaloArray}; kws...,
        ) where {F<:Function, OP}
        all_arrays = (parent(halo), parent.(etc)...)
        per_field_results = map(eachindex(parent(halo))) do idx
            field_arrays = map(arrs -> arrs[idx], all_arrays)
            $func(f, op, field_arrays...; kws...)
        end
        return reduce(op, per_field_results; kws...)
    end

    @eval function Base.$func(
            f::F, op::OP, z::Iterators.Zip{<:Tuple{Vararg{ArrayOfHaloArray}}}; kws...,
        ) where {F<:Function, OP}
        g(args...) = f(args)
        $func(g, op, z.is...; kws...)
    end
end

for func in (:mapreduce, :mapfoldl, :mapfoldr)
    @eval function Base.$func(
            f::F, op::OP, halo::MaybeHaloArray, etc::Vararg{MaybeHaloArray}; kws...,
        ) where {F<:Function, OP}
        all(isactive, (halo, etc...)) ||
            throw(ErrorException("MaybeHaloArray: attempt to reduce inactive value"))
        return $func(f, op, getdata(halo), getdata.(etc)...; kws...)
    end

    @eval function Base.$func(
            f::F, op::OP, z::Iterators.Zip{<:Tuple{Vararg{MaybeHaloArray}}}; kws...,
        ) where {F<:Function, OP}
        g(args...) = f(args)
        $func(g, op, z.is...; kws...)
    end
end

function Base.all(f::F, halo::MaybeHaloArray) where {F<:Function}
    isactive(halo) || throw(ErrorException("MaybeHaloArray: attempt to reduce inactive value"))
    return all(f, getdata(halo))
end

function Base.any(f::F, halo::MaybeHaloArray) where {F<:Function}
    isactive(halo) || throw(ErrorException("MaybeHaloArray: attempt to reduce inactive value"))
    return any(f, getdata(halo))
end

Base.sum(halo::AbstractHaloArray) = mapreduce(identity, +, halo)
Base.sum(f::F, halo::AbstractHaloArray) where {F<:Function} = mapreduce(f, +, halo)
Base.maximum(halo::AbstractHaloArray) = mapreduce(identity, max, halo)
Base.minimum(halo::AbstractHaloArray) = mapreduce(identity, min, halo)

# The inner product is a reduction: ⟨x,y⟩ = Σ conj(xᵢ)·yᵢ over owned cells.
# Reusing the two-argument mapreduce makes it inherit the correct global
# semantics on every backend — MPI Allreduce, threaded tile reduction, and
# per-field for collections — exactly like sum/maximum above. This overrides
# the generic AbstractArray dot, which would only reduce locally (silently
# wrong across MPI ranks).
LinearAlgebra.dot(x::AbstractHaloArray, y::AbstractHaloArray) = mapreduce(LinearAlgebra.dot, +, x, y)


# mapreduce_haloarray_dims and mapreduce_mhaloarray_dims live in mpi_support.jl
