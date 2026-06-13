# The MultiHaloArray alias, its docstring, and the ground-truth
# MultiHaloArray(::NamedTuple) constructor live in field_collection.jl.


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

# eltype/ndims come from AbstractArray{T,D} via FieldCollection{T,D,S,C}.

# size/axes/eachindex/length, n_field, interior/global/storage size, and
# interior_axes come from AbstractHaloCollection (field_shape prefix + _spatial_*).
@inline field_shape(mha::MultiHaloArray) = (length(mha.arrays),)
@inline Base.parent(halo::MultiHaloArray)  = map(parent, halo.arrays)

# Everything else MultiHaloArray needs is container-generic and defined once on
# FieldCollection (field_collection.jl) / AbstractHaloCollection
# (abstract_haloarray.jl): the _fields/_first_field/_map_fields/_check_same_fields
# hooks, tile_parent, to_tuple, active_fields, integer+Cartesian getindex/
# setindex!, similar(c[, T][, dims]), copy/copyto!/fill!/zero, map, interior_view,
# map_over_field, all/any, and halo_backend/halo_width/tile_*/isactive/is_root.

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
