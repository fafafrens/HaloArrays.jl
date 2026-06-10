"""
    MultiHaloArray(HaloArray, T, owned_dims, halo[, topology]; boundary_conditions)
    MultiHaloArray(named_tuple_of_fields)

A collection of several **named** halo-array fields sharing the same geometry
(dimensionality, interior size, halo width, and backend). Access a field by
name (`state.rho`), refresh them all with one [`synchronize_halo!`](@ref)`(state)`,
and broadcast/reduce over all fields at once (`state .*= 2`).

`boundary_conditions` is a `NamedTuple` mapping each field name to its boundary
condition; the field names are taken from its keys. The backing fields are
[`HaloArray`](@ref)s (MPI) here; use [`LocalMultiHaloArray`](@ref) or
[`ThreadedMultiHaloArray`](@ref) for local/threaded fields, or pass a
`NamedTuple` of pre-built arrays.

Use this when a solver evolves several fields on one grid (e.g. `rho`, `u`, `v`,
`p`). For an integer/matrix-indexed collection instead of names, see
[`ArrayOfHaloArray`](@ref).

# Examples
```julia
state = LocalMultiHaloArray(Float64, (64, 64), 1; boundary_conditions=(
    rho = ((Periodic(), Periodic()), (Periodic(), Periodic())),
    p   = ((Reflecting(), Reflecting()), (Periodic(), Periodic())),
))
state.rho .= 1.0
synchronize_halo!(state)   # refreshes every field
```
"""
mutable struct MultiHaloArray{T,N,A,D} <: AbstractHaloCollection{T,D,N}
    arrays::A
end

function _check_multihaloarray_compatible(field_names, field_values)
    isempty(field_values) && throw(ArgumentError("MultiHaloArray requires at least one field"))
    _check_fields_compatible("MultiHaloArray", first(field_values),
        zip(field_names, field_values))
    return nothing
end

function MultiHaloArray(arrs::NamedTuple; check=nothing)
    field_names = keys(arrs)
    field_values = values(arrs)
    _check_multihaloarray_compatible(field_names, field_values)

    field_eltypes = map(eltype, field_values)
   
    TTypes = promote_type(field_eltypes...)
    N_ref = _spatial_ndims(first(field_values))

    D = N_ref + 1
    return MultiHaloArray{TTypes,N_ref,typeof(arrs),D}(arrs)
end


# The MPI collection constructors are field-type-first: the first argument is
# always the field type (HaloArray), never the element type. The element-type-
# first forms were removed — they made the first positional argument mean two
# different things and caused method ambiguities against the specialized
# Local/Threaded constructors.

function MultiHaloArray(::Type{<:HaloArray}, ::Type{T}, owned_dims::NTuple{N,Int},
        halo::Int, topology::CartesianTopology{N};
        boundary_conditions::Union{NamedTuple,Nothing} = nothing,
        fields::Union{NTuple{<:Any,Symbol},Nothing} = nothing,
        boundary_condition = :repeating) where {T,N}
    bcs = _resolve_bcs(fields, boundary_condition, boundary_conditions)
    arrays = NamedTuple{keys(bcs)}(map(bcs) do bc
        HaloArray(T, owned_dims, halo, topology; boundary_condition=bc)
    end)
    return MultiHaloArray(arrays)
end

function MultiHaloArray(::Type{<:HaloArray}, ::Type{T}, owned_dims::NTuple{N,Int},
        halo::Int;
        boundary_conditions::Union{NamedTuple,Nothing} = nothing,
        fields::Union{NTuple{<:Any,Symbol},Nothing} = nothing,
        boundary_condition = :repeating) where {T,N}
    bcs = _resolve_bcs(fields, boundary_condition, boundary_conditions)
    arrays = NamedTuple{keys(bcs)}(map(bcs) do bc
        HaloArray(T, owned_dims, halo; boundary_condition=bc)
    end)
    return MultiHaloArray(arrays)
end

# Float64 defaults
MultiHaloArray(::Type{<:HaloArray}, owned_dims::NTuple{N,Int}, halo::Int,
        topology::CartesianTopology{N}; kwargs...) where {N} =
    MultiHaloArray(HaloArray, Float64, owned_dims, halo, topology; kwargs...)

MultiHaloArray(::Type{<:HaloArray}, owned_dims::NTuple{N,Int}, halo::Int;
        kwargs...) where {N} =
    MultiHaloArray(HaloArray, Float64, owned_dims, halo; kwargs...)

function MultiHaloArray(::Type{<:LocalHaloArray}, ::Type{T}, owned_dims::NTuple{N,Int},
        halo::Int; boundary_conditions::NamedTuple{names,<:Tuple}) where {T,N,names}
    arrays = NamedTuple{names}(map(boundary_conditions) do bc
        LocalHaloArray(T, owned_dims, halo; boundary_condition=bc)
    end)
    return MultiHaloArray(arrays)
end

function MultiHaloArray(::Type{<:LocalHaloArray}, owned_dims::NTuple{N,Int},
        halo::Int; boundary_conditions::NamedTuple{names,<:Tuple}) where {N,names}
    return MultiHaloArray(LocalHaloArray, Float64, owned_dims, halo;
        boundary_conditions=boundary_conditions)
end

function MultiHaloArray(::Type{<:ThreadedHaloArray}, ::Type{T},
        tile_size::NTuple{N,<:Integer}, halo::Integer;
        dims::NTuple{N,<:Integer} = ntuple(d -> d == N ? Threads.nthreads() : 1, Val(N)),
        boundary_conditions::NamedTuple{names,<:Tuple}) where {T,N,names}
    arrays = NamedTuple{names}(map(boundary_conditions) do bc
        ThreadedHaloArray(T, tile_size, halo; dims=dims, boundary_condition=bc)
    end)
    return MultiHaloArray(arrays)
end

function MultiHaloArray(::Type{<:ThreadedHaloArray}, tile_size::NTuple{N,<:Integer},
        halo::Integer;
        dims::NTuple{N,<:Integer} = ntuple(d -> d == N ? Threads.nthreads() : 1, Val(N)),
        boundary_conditions::NamedTuple{names,<:Tuple}) where {N,names}
    return MultiHaloArray(ThreadedHaloArray, Float64, tile_size, halo;
        dims=dims, boundary_conditions=boundary_conditions)
end


Base.getindex(mha::MultiHaloArray, name::Symbol) = mha.arrays[name]

# Named-field access: `state.rho` forwards to the backing field, while
# `state.arrays` still returns the underlying NamedTuple (`arrays` is the only
# real struct field). `getfield` is used internally to avoid recursion.
@inline Base.getproperty(mha::MultiHaloArray, name::Symbol) =
    name === :arrays ? getfield(mha, :arrays) : getfield(mha, :arrays)[name]
Base.propertynames(mha::MultiHaloArray) = keys(getfield(mha, :arrays))

Base.eltype(mha::MultiHaloArray{T,N,A,D}) where {T,N,A,D} = T
Base.eltype(::Type{<:MultiHaloArray{T,N,A,D}}) where {T,N,A,D} = T
Base.ndims(mha::MultiHaloArray{T,N,A,D}) where {T,N,A,D} = D
Base.ndims(::Type{<:MultiHaloArray{T,N,A,D}}) where {T,N,A,D} = D

@inline Base.size(mha::MultiHaloArray) = global_size(mha)

@inline Base.size(mha::MultiHaloArray, i::Int) = size(mha)[i]
@inline Base.length(mha::MultiHaloArray) = prod(size(mha))
@inline Base.eachindex(mha::MultiHaloArray) = CartesianIndices(axes(mha))

n_field(halos::MultiHaloArray{T,N,A,D}) where {T,N,A,D} = length(halos.arrays)

@inline interior_size(halos::MultiHaloArray) = (n_field(halos), _spatial_interior_size(first(values(halos.arrays)))...)
@inline owned_size(halos::MultiHaloArray) = (n_field(halos), _spatial_owned_size(first(values(halos.arrays)))...)
@inline global_size(halos::MultiHaloArray) = (n_field(halos), _spatial_global_size(first(values(halos.arrays)))...)
@inline storage_size(halos::MultiHaloArray) = (n_field(halos), _spatial_storage_size(first(values(halos.arrays)))...)
@inline storage_size(halos::MultiHaloArray,i) = storage_size(halos)[i]
@inline halo_width(halo::MultiHaloArray, i) = map(halo_width, halo.arrays)
@inline Base.parent(halo::MultiHaloArray)  = map(parent, halo.arrays)

# AbstractHaloCollection helpers (concrete methods; stubs in abstract_haloarray.jl)
@inline _first_field(mha::MultiHaloArray) = first(values(mha.arrays))
@inline _fields(mha::MultiHaloArray)      = values(mha.arrays)
@inline Base.axes(x::MultiHaloArray) = (Base.OneTo(n_field(x)), _spatial_axes(first(values(x.arrays)))...)
@inline Base.axes(x::MultiHaloArray,i) = axes(x)[i]
@inline owned_axes(x::MultiHaloArray) = (Base.OneTo(n_field(x)), _spatial_owned_axes(first(values(x.arrays)))...)
@inline owned_axes(x::MultiHaloArray, i::Int) = owned_axes(x)[i]
@inline tile_parent(halos::MultiHaloArray, tile_id::Integer) =
    NamedTuple{keys(halos.arrays)}(map(a -> tile_parent(a, tile_id), values(halos.arrays)))
# halo_backend, halo_width, tile_size, tile_count, tile_coordinates, neighbor_tile_id
# inherited from AbstractHaloCollection (abstract_haloarray.jl)

to_tuple(mha::MultiHaloArray) = (mha.arrays...,)

function Base.getindex(mha::MultiHaloArray, field_index::Integer)
    1 <= field_index <= n_field(mha) || throw(BoundsError(mha, (field_index,)))
    return values(mha.arrays)[field_index]
end

function Base.getindex(mha::MultiHaloArray, field_index::Integer, I::Vararg{Integer})
    return getindex(getindex(mha, field_index), I...)
end

function Base.setindex!(mha::MultiHaloArray, value, field_index::Integer, I::Vararg{Integer})
    setindex!(getindex(mha, field_index), value, I...)
    return mha
end

function Base.similar(mha::MultiHaloArray{AA,N,A,D}, ::Type{T}, dims::Dims{M}) where {AA,N,A,D,T,M}
    M == N + 1 || throw(DimensionMismatch("MultiHaloArray similar dims must have $(N + 1) dimensions"))
    Int(dims[1]) == n_field(mha) ||
        throw(DimensionMismatch("MultiHaloArray similar cannot change the named field count from $(n_field(mha)) to $(dims[1])"))
    spatial_dims = ntuple(d -> Int(dims[d + 1]), Val(N))

    arrs = map(a -> similar(a, T, spatial_dims), values(mha.arrays))
    names = keys(mha.arrays)
    nt = NamedTuple{names}(arrs)
    return MultiHaloArray(nt)
end

# Non-Int dims are normalized to Dims by Base's generic similar fallbacks.

Base.similar(mha::MultiHaloArray, dims::Dims{M}) where {M} =
    similar(mha, eltype(mha), dims)
Base.similar(mha::MultiHaloArray, dims::NTuple{M,<:Integer}) where {M} =
    similar(mha, eltype(mha), dims)

function Base.similar(mha::MultiHaloArray{AA,N,A,D}, ::Type{T}) where {AA,N,A,D,T}
    arrs = map(a -> similar(a, T), values(mha.arrays))
    names = keys(mha.arrays)
    nt = NamedTuple{names}(arrs)
    return MultiHaloArray(nt)
end


function Base.similar(mha::MultiHaloArray{AA,N,A,D}) where {AA,N,A,D}
    arrs = map(a -> similar(a), values(mha.arrays))
    names = keys(mha.arrays)
    nt = NamedTuple{names}(arrs)
    return MultiHaloArray(nt)
end


function Base.copy(mha::MultiHaloArray)
    newfields = map(x -> copy(x), values(mha.arrays))
    new_ntuple = NamedTuple{keys(mha.arrays)}(newfields)
    return MultiHaloArray(new_ntuple)
end

function Base.copyto!(dest::MultiHaloArray, src::MultiHaloArray)
    keys(dest.arrays) == keys(src.arrays) ||
        throw(DimensionMismatch("MultiHaloArray copyto! requires matching field names"))
    for name in keys(dest.arrays)
        copyto!(dest.arrays[name], src.arrays[name])
    end
    return dest
end

function Base.fill!(mha::MultiHaloArray, value)
    foreach(field -> fill!(field, value), values(mha.arrays))
    return mha
end

function Base.zero(mha::MultiHaloArray)
    z = similar(mha)
    fill!(z, zero(eltype(mha)))
    return z
end

function Base.map(f, mha::MultiHaloArray)
    newfields = map(x -> map(f, x), values(mha.arrays))
    new_ntuple = NamedTuple{keys(mha.arrays)}(newfields)
    return MultiHaloArray(new_ntuple)
end

# foreach_field!(f!, ::AbstractHaloCollection) inherited from abstract_haloarray.jl


function foreach_field!(f!, mha::MultiHaloArray, etc::Vararg{MultiHaloArray})
    for (name, arr) in pairs(mha.arrays)
        f!(arr, map(x -> x.arrays[name], etc)...)
    end
    return nothing
end

function map_over_field(f, mha::MultiHaloArray)

    return map(f, mha.arrays)

end

function map_over_field(f, mha::MultiHaloArray,etc::Vararg{MultiHaloArray})
    n_fields = n_field(mha)
    keyset = keys(mha.arrays)

    result = ntuple(n_fields) do n
        f(mha.arrays[n], map(x -> x.arrays[n], etc)...)
    end

    return NamedTuple{keyset}(result)
end

"""
    interior_view(mha::MultiHaloArray)

Return a `NamedTuple` with the interior view of each field.
"""
function interior_view(mha::MultiHaloArray)
    return NamedTuple{keys(mha.arrays)}(map(interior_view, values(mha.arrays)))
end

function interior_view(mha::MultiHaloArray, tile_id::Integer)
    return NamedTuple{keys(mha.arrays)}(map(a -> interior_view(a, tile_id), values(mha.arrays)))
end

# isactive, is_root inherited from AbstractHaloCollection (abstract_haloarray.jl)

function active_fields(mha::MultiHaloArray)
    return (; (name => isactive(ha) for (name, ha) in mha.arrays)...)
end

# ---- LocalMultiHaloArray constructors -----------------------------------

function LocalMultiHaloArray(arrs::NamedTuple; check=nothing)
    field_values = values(arrs)
    isempty(field_values) && throw(ArgumentError("LocalMultiHaloArray requires at least one field"))
    all(a -> a isa LocalHaloArray, field_values) ||
        throw(ArgumentError("All fields must be LocalHaloArray"))
    return MultiHaloArray(arrs; check=check)
end

"""
    LocalMultiHaloArray(T, owned_dims, halo; boundary_conditions)

A [`MultiHaloArray`](@ref) whose fields are [`LocalHaloArray`](@ref)s
(single-process). `boundary_conditions` is a `NamedTuple` of per-field boundary
conditions, which also fixes the field names. See [`MultiHaloArray`](@ref).
"""
function LocalMultiHaloArray(::Type{T}, owned_dims::NTuple{N,<:Integer}, halo::Integer;
        boundary_conditions::Union{NamedTuple,Nothing} = nothing,
        fields::Union{NTuple{<:Any,Symbol},Nothing} = nothing,
        boundary_condition = :repeating) where {T,N}
    bcs = _resolve_bcs(fields, boundary_condition, boundary_conditions)
    normalized_owned_dims = ntuple(d -> Int(owned_dims[d]), Val(N))
    return MultiHaloArray(LocalHaloArray, T, normalized_owned_dims, Int(halo);
        boundary_conditions = bcs)
end

# positional form kept for backward compatibility
LocalMultiHaloArray(owned_dims::NTuple{N,<:Integer}, halo::Integer,
        bcs::NamedTuple; kwargs...) where {N} =
    LocalMultiHaloArray(Float64, owned_dims, halo; boundary_conditions=bcs, kwargs...)

# Float64 default — kwargs forwarded to T-explicit above
LocalMultiHaloArray(owned_dims::NTuple{N,<:Integer}, halo::Integer; kwargs...) where {N} =
    LocalMultiHaloArray(Float64, owned_dims, halo; kwargs...)

# ---- ThreadedMultiHaloArray constructors --------------------------------

function ThreadedMultiHaloArray(arrs::NamedTuple; check=nothing)
    field_values = values(arrs)
    isempty(field_values) && throw(ArgumentError("ThreadedMultiHaloArray requires at least one field"))
    all(a -> a isa ThreadedHaloArray, field_values) ||
        throw(ArgumentError("All fields must be ThreadedHaloArray"))
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

"""
    ThreadedMultiHaloArray(T, tile_size, halo; dims, boundary_conditions)

A [`MultiHaloArray`](@ref) whose fields are [`ThreadedHaloArray`](@ref)s sharing
one tile layout (`dims`). `synchronize_halo!` exchanges every field's tiles in a
single call. `boundary_conditions` is a `NamedTuple` of per-field boundary
conditions. See [`MultiHaloArray`](@ref) and [`ThreadedHaloArray`](@ref).
"""
function ThreadedMultiHaloArray(::Type{T}, tile_size::NTuple{N,<:Integer}, halo::Integer;
        dims::NTuple{N,<:Integer} = ntuple(d -> d == N ? Threads.nthreads() : 1, Val(N)),
        boundary_conditions::Union{NamedTuple,Nothing} = nothing,
        fields::Union{NTuple{<:Any,Symbol},Nothing} = nothing,
        boundary_condition = :repeating) where {T,N}
    bcs = _resolve_bcs(fields, boundary_condition, boundary_conditions)
    return MultiHaloArray(ThreadedHaloArray, T, tile_size, halo;
        dims=dims, boundary_conditions=bcs)
end

ThreadedMultiHaloArray(tile_size::NTuple{N,<:Integer}, halo::Integer; kwargs...) where {N} =
    ThreadedMultiHaloArray(Float64, tile_size, halo; kwargs...)

# ============================================================
# Helpers for the uniform-BC shorthand (fields + boundary_condition)
# ============================================================

function _make_boundary_conditions(fields::NTuple{M,Symbol}, bc) where {M}
    return NamedTuple{fields}(ntuple(_ -> bc, Val(M)))
end

function _resolve_bcs(
        fields::Union{NTuple{<:Any,Symbol},Nothing},
        bc,
        boundary_conditions::Union{NamedTuple,Nothing})
    fields !== nothing && return _make_boundary_conditions(fields, bc)
    boundary_conditions !== nothing && return boundary_conditions
    throw(ArgumentError(
        "provide either `fields` (with `boundary_condition`) or `boundary_conditions`"))
end

