function LocalMultiHaloArray(arrs::NamedTuple; check=nothing)
    field_values = values(arrs)
    isempty(field_values) && throw(ArgumentError("LocalMultiHaloArray requires at least one field"))

    if !all(a -> a isa LocalHaloArray, field_values)
        throw(ArgumentError("All fields must be LocalHaloArray"))
    end

    return MultiHaloArray(arrs; check=check)
end

function LocalMultiHaloArray(::Type{T}, owned_dims::NTuple{N,<:Integer}, halo::Integer;
        boundary_conditions::NamedTuple{names,<:Tuple}) where {T,N,names}
    normalized_owned_dims = ntuple(d -> Int(owned_dims[d]), Val(N))
    return MultiHaloArray(LocalHaloArray, T, normalized_owned_dims, Int(halo);
        boundary_conditions=boundary_conditions)
end

function LocalMultiHaloArray(owned_dims::NTuple{N,<:Integer}, halo::Integer,
        bcs::NamedTuple{names,<:Tuple}) where {N,names}
    return LocalMultiHaloArray(Float64, owned_dims, halo; boundary_conditions=bcs)
end

function LocalMultiHaloArray(owned_dims::NTuple{N,<:Integer}, halo::Integer;
        boundary_conditions::NamedTuple{names,<:Tuple}) where {N,names}
    return LocalMultiHaloArray(Float64, owned_dims, halo;
        boundary_conditions=boundary_conditions)
end
