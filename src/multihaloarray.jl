# The MultiHaloArray alias, its docstring, and the ground-truth
# MultiHaloArray(::NamedTuple) constructor live in field_collection.jl.


# The MPI collection constructors are field-type-first: the first argument is
# always the field type (HaloArray), never the element type. The element-type-
# first forms were removed — they made the first positional argument mean two
# different things and caused method ambiguities against the specialized
# Local/Threaded constructors.

function MultiHaloArray(FT::Type{<:HaloArray}, ::Type{T}, owned_dims::NTuple{N,Int},
        halo::Int, topology::CartesianTopology{N};
        boundary_conditions::Union{NamedTuple,Nothing} = nothing,
        fields::Union{NTuple{<:Any,Symbol},Nothing} = nothing,
        boundary_condition = :repeating) where {T,N}
    bcs = _resolve_bcs(fields, boundary_condition, boundary_conditions)
    return MultiHaloArray(map(bc -> _make_field(FT, T, owned_dims, halo, topology, bc), bcs))
end

function MultiHaloArray(FT::Type{<:HaloArray}, ::Type{T}, owned_dims::NTuple{N,Int},
        halo::Int;
        boundary_conditions::Union{NamedTuple,Nothing} = nothing,
        fields::Union{NTuple{<:Any,Symbol},Nothing} = nothing,
        boundary_condition = :repeating) where {T,N}
    bcs = _resolve_bcs(fields, boundary_condition, boundary_conditions)
    return MultiHaloArray(map(bc -> _make_field(FT, T, owned_dims, halo, bc), bcs))
end

# Float64 defaults
MultiHaloArray(::Type{<:HaloArray}, owned_dims::NTuple{N,Int}, halo::Int,
        topology::CartesianTopology{N}; kwargs...) where {N} =
    MultiHaloArray(HaloArray, Float64, owned_dims, halo, topology; kwargs...)

MultiHaloArray(::Type{<:HaloArray}, owned_dims::NTuple{N,Int}, halo::Int;
        kwargs...) where {N} =
    MultiHaloArray(HaloArray, Float64, owned_dims, halo; kwargs...)

function MultiHaloArray(FT::Type{<:LocalHaloArray}, ::Type{T}, owned_dims::NTuple{N,<:Integer},
        halo::Integer;
        boundary_conditions::Union{NamedTuple,Nothing} = nothing,
        fields::Union{NTuple{<:Any,Symbol},Nothing} = nothing,
        boundary_condition = :repeating) where {T,N}
    bcs = _resolve_bcs(fields, boundary_condition, boundary_conditions)
    dims_n = ntuple(d -> Int(owned_dims[d]), Val(N))
    return MultiHaloArray(map(bc -> _make_field(FT, T, dims_n, Int(halo), bc), bcs))
end

MultiHaloArray(::Type{<:LocalHaloArray}, owned_dims::NTuple{N,<:Integer}, halo::Integer;
        kwargs...) where {N} =
    MultiHaloArray(LocalHaloArray, Float64, owned_dims, halo; kwargs...)

function MultiHaloArray(FT::Type{<:ThreadedHaloArray}, ::Type{T},
        tile_size::NTuple{N,<:Integer}, halo::Integer;
        dims::NTuple{N,<:Integer} = ntuple(d -> d == N ? Threads.nthreads() : 1, Val(N)),
        boundary_conditions::Union{NamedTuple,Nothing} = nothing,
        fields::Union{NTuple{<:Any,Symbol},Nothing} = nothing,
        boundary_condition = :repeating) where {T,N}
    bcs = _resolve_bcs(fields, boundary_condition, boundary_conditions)
    return MultiHaloArray(map(bc -> _make_field(FT, T, tile_size, halo, bc; dims=dims), bcs))
end

MultiHaloArray(::Type{<:ThreadedHaloArray}, tile_size::NTuple{N,<:Integer}, halo::Integer;
        kwargs...) where {N} =
    MultiHaloArray(ThreadedHaloArray, Float64, tile_size, halo; kwargs...)


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
"""
    field_shape(c) -> Dims

The shape of the field index of a halo collection `c`: the sizes of the index
dimensions that select a field, leaving the spatial dimensions out. For a
`MultiHaloArray` of `n` named fields this is `(n,)`; for an `ArrayOfHaloArray` it
is the `size` of the backing field array (e.g. `(4, 2)` for a `4×2` grid of
fields). For a nested collection the outer container's shape is followed by
the fields' own field shape; a `MultiHaloArray` whose fields differ in field
shape has the single axis `(number of leaves,)`. `prod(field_shape(c))` is the
number of leaf fields; a single halo array has field shape `()`.
"""
@inline field_shape(c::AbstractHaloCollection) = _has_leaf_axis(c) ?
    (sum(_leaf_count, _fields(c)),) :
    (_container_shape(c)..., field_shape(_first_field(c))...)
@inline field_shape(::AbstractSingleHaloArray) = ()
@inline _container_shape(mha::MultiHaloArray) = (length(getfield(mha, :arrays)),)
# parent (the field NamedTuple) and field_storages (the storage NamedTuple) are
# container-generic in field_collection.jl.

# Everything else MultiHaloArray needs is container-generic and defined once on
# FieldCollection (field_collection.jl) / AbstractHaloCollection
# (abstract_haloarray.jl): the _fields/_first_field/_map_fields/_check_same_fields
# hooks, tile_parent, to_tuple, active_fields, integer+Cartesian getindex/
# setindex!, similar(c[, T][, dims]), copy/copyto!/fill!/zero, map, interior_view,
# map_over_field, all/any, and halo_backend/halo_width/tile_*/is_active/is_root.

# ---- deprecated backend-named constructors ---------------------------------
# `MultiHaloArray(LocalHaloArray, …)` / `MultiHaloArray(ThreadedHaloArray, …)`
# accept the same keywords; a NamedTuple of fields is just `MultiHaloArray(nt)`.
Base.@deprecate LocalMultiHaloArray(arrs::NamedTuple) MultiHaloArray(arrs) false
Base.@deprecate LocalMultiHaloArray(T::Type, owned_dims::Tuple, halo::Integer; kwargs...) MultiHaloArray(LocalHaloArray, T, owned_dims, halo; kwargs...) false
Base.@deprecate LocalMultiHaloArray(owned_dims::Tuple, halo::Integer, bcs::NamedTuple; kwargs...) MultiHaloArray(LocalHaloArray, Float64, owned_dims, halo; boundary_conditions=bcs, kwargs...) false
Base.@deprecate LocalMultiHaloArray(owned_dims::Tuple, halo::Integer; kwargs...) MultiHaloArray(LocalHaloArray, Float64, owned_dims, halo; kwargs...) false
Base.@deprecate ThreadedMultiHaloArray(arrs::NamedTuple) MultiHaloArray(arrs) false
Base.@deprecate ThreadedMultiHaloArray(T::Type, tile_size::Tuple, halo::Integer; kwargs...) MultiHaloArray(ThreadedHaloArray, T, tile_size, halo; kwargs...) false
Base.@deprecate ThreadedMultiHaloArray(tile_size::Tuple, halo::Integer; kwargs...) MultiHaloArray(ThreadedHaloArray, Float64, tile_size, halo; kwargs...) false

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
