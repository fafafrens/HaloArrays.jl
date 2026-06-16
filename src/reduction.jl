# mapreduce / any / all for HaloArray (MPI) live in mpi_support.jl

# Reduce over one or more already-resolved interior views. With a single view we
# forward straight to Base's (efficient) reduction; with two or more we reduce
# over a lazy `zip` rather than `reducer(f, op, A, B)` — Base's multi-iterator
# `mapreduce` materializes `map(f, A, B)` into a full interior-sized array
# (O(N) allocation per call). `reducer` is `mapreduce`/`mapfoldl`/`mapfoldr`.
@inline function _reduce_views(reducer::R, f::F, op::OP, views::Tuple; kws...) where {R,F,OP}
    return length(views) == 1 ? reducer(f, op, views[1]; kws...) :
                                reducer(t -> f(t...), op, zip(views...); kws...)
end

for func in (:mapreduce, :mapfoldl, :mapfoldr)
    @eval function Base.$func(
            f::F, op::OP, halo::LocalHaloArray, etc::Vararg{LocalHaloArray}; kws...,
        ) where {F<:Function, OP}
        return _reduce_views($func, f, op, map(interior_view, (halo, etc...)); kws...)
    end

    @eval function Base.$func(
            f::F, op::OP, z::Iterators.Zip{<:Tuple{LocalHaloArray,Vararg{LocalHaloArray}}}; kws...,
        ) where {F<:Function, OP}
        g(args...) = f(args)
        return $func(g, op, z.is...; kws...)
    end

    @eval function Base.$func(
            f::F, op::OP, halo::ThreadedHaloArray, etc::Vararg{ThreadedHaloArray}; kws...,
        ) where {F<:Function, OP}
        # Reduce each tile (serially, with the user's kwargs), then combine the
        # per-tile results with `op` across tiles via the array's thread backend.
        per_tile(tile_id) = _reduce_views($func, f, op, map(h -> interior_view(h, tile_id), (halo, etc...)); kws...)
        return tile_mapreduce(thread_backend(halo), per_tile, op, 1:tile_count(halo); scheduler=:static)
    end

    @eval function Base.$func(
            f::F, op::OP, z::Iterators.Zip{<:Tuple{ThreadedHaloArray,Vararg{ThreadedHaloArray}}}; kws...,
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
            f::F, op::OP, z::Iterators.Zip{<:Tuple{AbstractHaloCollection,Vararg{AbstractHaloCollection}}}; kws...,
        ) where {F<:Function, OP}
        g(args...) = f(args)
        $func(g, op, z.is...; kws...)
    end
end


# Field-wise short-circuit all/any for any collection kind (named or indexed):
# reduce over fields, each field reducing over its own interior.
Base.all(f::F, c::FieldCollection) where {F<:Function} =
    all(field -> all(f, field), _fields(c))
Base.any(f::F, c::FieldCollection) where {F<:Function} =
    any(field -> any(f, field), _fields(c))

for func in (:mapreduce, :mapfoldl, :mapfoldr)
    @eval function Base.$func(
            f::F, op::OP, halo::MaybeHaloArray, etc::Vararg{MaybeHaloArray}; kws...,
        ) where {F<:Function, OP}
        all(isactive, (halo, etc...)) ||
            throw(ErrorException("MaybeHaloArray: attempt to reduce inactive value"))
        return $func(f, op, getdata(halo), getdata.(etc)...; kws...)
    end

    @eval function Base.$func(
            f::F, op::OP, z::Iterators.Zip{<:Tuple{MaybeHaloArray,Vararg{MaybeHaloArray}}}; kws...,
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

# dot, norm, and the in-place BLAS-1 ops (rmul!/lmul!/axpy!/axpby!) — all built
# on the mapreduce/broadcast above — live in vector_space.jl.


# mapreduce_haloarray_dims and mapreduce_mhaloarray_dims live in mpi_support.jl
