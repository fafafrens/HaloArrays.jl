function LocalMultiHaloArray(arrs::NamedTuple; check=nothing)
    field_values = values(arrs)
    isempty(field_values) && throw(ArgumentError("LocalMultiHaloArray requires at least one field"))

    if !all(a -> a isa LocalHaloArray, field_values)
        throw(ArgumentError("All fields must be LocalHaloArray"))
    end

    return MultiHaloArray(arrs; check=check)
end

function LocalMultiHaloArray(::Type{T}, local_size::NTuple{N,<:Integer}, halo::Integer;
        boundary_conditions::NamedTuple{names,<:Tuple}) where {T,N,names}
    owned_size = ntuple(d -> Int(local_size[d]), Val(N))
    return MultiHaloArray(LocalHaloArray, T, owned_size, Int(halo);
        boundary_conditions=boundary_conditions)
end

function LocalMultiHaloArray(local_size::NTuple{N,<:Integer}, halo::Integer,
        bcs::NamedTuple{names,<:Tuple}) where {N,names}
    return LocalMultiHaloArray(Float64, local_size, halo; boundary_conditions=bcs)
end

function LocalMultiHaloArray(local_size::NTuple{N,<:Integer}, halo::Integer;
        boundary_conditions::NamedTuple{names,<:Tuple}) where {N,names}
    return LocalMultiHaloArray(Float64, local_size, halo;
        boundary_conditions=boundary_conditions)
end
