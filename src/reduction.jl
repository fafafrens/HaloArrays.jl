# mapreduce / any / all for HaloArray (MPI) live in mpi_support.jl

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
        # Reduce each tile (serially, with the user's kwargs), then combine the
        # per-tile results with `op` across tiles via the array's thread backend.
        per_tile(tile_id) = $func(f, op, map(h -> interior_view(h, tile_id), (halo, etc...))...; kws...)
        return tile_mapreduce(thread_backend(halo), per_tile, op, 1:tile_count(halo); scheduler=:static)
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
    return tile_mapreduce(thread_backend(u), tile_id -> any(f, interior_view(u, tile_id)), |,
        1:tile_count(u); scheduler=:static)
end

function Base.all(f::F, u::ThreadedHaloArray) where {F<:Function}
    return tile_mapreduce(thread_backend(u), tile_id -> all(f, interior_view(u, tile_id)), &,
        1:tile_count(u); scheduler=:static)
end

# mapreduce/mapfoldl/mapfoldr over a multi-field container reduce each field
# across the inputs, then reduce the per-field results. One definition covers any
# AbstractHaloCollection (MultiHaloArray + ArrayOfHaloArray) via `eachfield`.
for func in (:mapreduce, :mapfoldl, :mapfoldr)
    @eval function Base.$func(
            f::F, op::OP, halo::AbstractHaloCollection, etc::Vararg{AbstractHaloCollection}; kws...,
        ) where {F<:Function, OP}
        all_fields = map(eachfield, (halo, etc...))
        per_field_results = map(eachindex(eachfield(halo))) do idx
            $func(f, op, map(fields -> fields[idx], all_fields)...; kws...)
        end
        return reduce(op, per_field_results; kws...)
    end

    @eval function Base.$func(
            f::F, op::OP, z::Iterators.Zip{<:Tuple{Vararg{AbstractHaloCollection}}}; kws...,
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

# ArrayOfHaloArray reductions are covered by the AbstractHaloCollection methods above.

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
