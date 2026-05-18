function ThreadedMultiHaloArray(arrs::NamedTuple; check=nothing)
    field_values = values(arrs)
    isempty(field_values) && throw(ArgumentError("ThreadedMultiHaloArray requires at least one field"))

    if !all(a -> a isa ThreadedHaloArray, field_values)
        throw(ArgumentError("All fields must be ThreadedHaloArray"))
    end

    ref = first(field_values)
    for (name, a) in zip(keys(arrs), field_values)
        tile_size(a) == tile_size(ref) ||
            throw(DimensionMismatch("Field `$(name)` has tile_size $(tile_size(a)) != $(tile_size(ref))"))
        halo_width(a) == halo_width(ref) ||
            throw(DimensionMismatch("Field `$(name)` has halo width $(halo_width(a)) != $(halo_width(ref))"))
        a.topology.dims == ref.topology.dims ||
            throw(DimensionMismatch("Field `$(name)` has topology dims $(a.topology.dims) != $(ref.topology.dims)"))
        tile_count(a) == tile_count(ref) ||
            throw(DimensionMismatch("Field `$(name)` has tile_count $(tile_count(a)) != $(tile_count(ref))"))
    end

    return MultiHaloArray(arrs; check=check)
end

function ThreadedMultiHaloArray(::Type{T}, tile_size::NTuple{N,<:Integer}, halo::Integer;
        dims::NTuple{N,<:Integer},
        boundary_conditions::NamedTuple{names,<:Tuple}) where {T,N,names}
    return MultiHaloArray(ThreadedHaloArray, T, tile_size, halo;
        dims=dims, boundary_conditions=boundary_conditions)
end

ThreadedMultiHaloArray(tile_size::NTuple{N,<:Integer}, halo::Integer; kwargs...) where {N} =
    ThreadedMultiHaloArray(Float64, tile_size, halo; kwargs...)
