# Local storage access: no halo exchange or intermediate field-container map.
@inline _cell_storage(field::_SingleBlock, ::Nothing) = parent(field)
@inline _cell_storage(field::ThreadedHaloArray, ::Nothing) =
    throw(ArgumentError("threaded field access requires an explicit tile id"))
Base.@propagate_inbounds function _cell_storage(field::AbstractSingleHaloArray, tile::Integer)
    @boundscheck 1 <= tile <= tile_count(field) || throw(BoundsError(field, tile))
    return tile_parent(field, tile)
end

"""
    siteview(state, I::CartesianIndex[, tile])
    siteview(tile_parent(state, tile), I::CartesianIndex)

Return a lazy, writable array of all fields at local padded-storage index `I`,
shaped like the collection's fields, without copying component values. An
`ArrayOfHaloArray` of size `(2, 2, nx, ny, nz)` gives a 2×2 matrix whose entry
`q[a, b]` is field `(a, b)` at the site. A `MultiHaloArray` gives a vector in
declaration order, and a single `LocalHaloArray`, `HaloArray`, or
`ThreadedHaloArray` a one-element vector. `size(q) == field_shape(state)` for
collections; the number of dimensions is part of the view's type, the extents of
an `ArrayOfHaloArray` are not. Linear indexing `q[k]` follows the column-major
field order, and `vec(q)` gives a flat view. Array-valued cell elements are
preserved as one component, not flattened.

`I` uses the storage coordinates of `interior_cells(CellRanges(state))`, including
allocated halo cells. Threaded arrays and collections require an explicit tile id;
Local/MPI collections access local storage and optionally accept tile id `1`.
No communication occurs. Synchronize halos before reading ghost cells and
refresh them after modifying interior values when subsequent stencils need them.
These scalar accessors are intended for CPU storage, not GPU kernel launches.

Writes immediately modify the underlying fields. Use `copy(q)` for an independent
snapshot; `similar(q)` creates an ordinary uninitialized array of the same
shape. Copying and
broadcasting use Julia's standard implementations and checks: `copyto!` copies
in linear (column-major) order and checks destination capacity, and broadcast
checks compatible shapes with singleton expansion, so broadcasting into a 2×2
site needs a 2×2 (or broadcastable) operand; use `vec(q)` with flat buffers. No additional component-length or shape checks are defined here.
Operations requiring an automatic alias-protection copy of a site view throw
`ArgumentError`; copy the source explicitly, e.g. `q .= copy(view(q, 4:-1:1))`.
Alias detection is conservative: separate views sharing field storage may require
an explicit copy even at distinct sites. Direct self-broadcast such as `q .*= 2`
and operations with independent buffers are supported.
Scalar reads convert to the view's element type: the state's element type,
promoted for collections, so `q[k] isa eltype(q)` holds for mixed field types.
Read a field directly for its native type. Allocated outputs such as `copy(q)` and
`similar(q)` use the same element type. Writes convert to the individual
destination field's element type. The view is not contiguous storage.
Do not resize or replace the collection's fields while using a view. Concurrent
writes must target disjoint storage or be externally synchronized.

Construction performs no validation. The caller must supply a single array or flat collection,
a storage index of the correct dimension and within bounds, and a valid tile
(explicit for threaded storage). Ordinary scalar indexing remains bounds-checked;
`@inbounds` may elide those checks. Alias detection remains enabled.

In a tile loop that also writes, resolve the tile once and index its storage:
`siteview(tile_parent(state, t), I)` accepts what [`tile_parent`](@ref) returns
(a padded array, or a tuple or `NamedTuple` of them, nested like the
collection) and gives a vector of the leaf fields in the same order. On
threaded storage `siteview(state, I, t)` looks the tile up at every access,
which the compiler cannot hoist out of a loop that stores into the fields.

```julia
q = siteview(state, I)          # pass a final tile id for threaded storage
copyto!(buffer, q)             # gather (any buffer of length(q), column-major)
copyto!(q, buffer)             # scatter
q .+= 0.5 .* buffer            # accumulate; buffer shaped like q
U = ArrayOfHaloArray(LocalHaloArray, Float64, (2, 2), (nx, ny, nz), 1)
m = siteview(U, I)             # lazy 2×2 matrix: m[a, b] is field (a, b)
```
"""
@inline function siteview(state::AbstractSingleHaloArray{T}, I::CartesianIndex,
        tile::Union{Nothing,Integer}=nothing) where {T}
    return SiteView{T,1,typeof(state),typeof(I),typeof(tile)}(state, I, tile)
end

# A collection's first D - S dimensions select the field.
@inline function siteview(state::AbstractHaloCollection{T,D,S}, I::CartesianIndex,
        tile::Union{Nothing,Integer}=nothing) where {T,D,S}
    return SiteView{T,D - S,typeof(state),typeof(I),typeof(tile)}(state, I, tile)
end

struct SiteView{T,N,C,I,K} <: AbstractArray{T,N}
    state::C
    index::I
    tile::K
end

# A single field uses a one-element tuple, without constructing a collection.
@inline _site_fields(state::AbstractSingleHaloArray) = (state,)
@inline _site_fields(state::AbstractHaloCollection) = eachfield(state)

Base.parent(q::SiteView) = q.state
Base.size(q::SiteView) = _site_size(q.state)
@inline _site_size(::AbstractSingleHaloArray) = (1,)
@inline _site_size(state::AbstractHaloCollection) = field_shape(state)
Base.IndexStyle(::Type{<:SiteView}) = IndexLinear()

Base.@propagate_inbounds function Base.getindex(q::SiteView{T}, k::Int) where {T}
    @boundscheck checkbounds(q, k)
    return convert(T, _site_get(q.state, k, q.tile, q.index))
end

Base.@propagate_inbounds function Base.setindex!(q::SiteView, value, k::Int)
    @boundscheck checkbounds(q, k)
    _site_set!(q.state, value, k, q.tile, q.index)
    return q
end

# Read/write leaf `k` at the site through `_leaf_field`. A collection with a
# leaf axis has fields of different types, so it walks its field tuple at the
# value level instead (each step returns an element, not a leaf array of a
# different type, which would put a type union in the hot loop).
# These carry the caller's @inbounds to the storage access: without it every
# site access is bounds-checked and the loop stops vectorising (2-3x slower).
Base.@propagate_inbounds _site_get(state, k::Int, tile, I) = _cell_storage(_leaf_field(state, k), tile)[I]
Base.@propagate_inbounds _site_get(state::FieldCollection{T,D,S,C,LeafAxis}, k::Int, tile, I) where {T,D,S,C} =
    _concat_get(_site_fields(state), k, tile, I)
Base.@propagate_inbounds _site_set!(state, v, k::Int, tile, I) = (_cell_storage(_leaf_field(state, k), tile)[I] = v; nothing)
Base.@propagate_inbounds _site_set!(state::FieldCollection{T,D,S,C,LeafAxis}, v, k::Int, tile, I) where {T,D,S,C} =
    _concat_set!(_site_fields(state), v, k, tile, I)
_concat_get(::Tuple{}, k::Int, tile, I) = throw(BoundsError((), k))
Base.@propagate_inbounds function _concat_get(fields::Tuple, k::Int, tile, I)
    f = first(fields); n = _leaf_count(f)
    return k <= n ? _site_get(f, k, tile, I) : _concat_get(Base.tail(fields), k - n, tile, I)
end
_concat_set!(::Tuple{}, v, k::Int, tile, I) = throw(BoundsError((), k))
Base.@propagate_inbounds function _concat_set!(fields::Tuple, v, k::Int, tile, I)
    f = first(fields); n = _leaf_count(f)
    return k <= n ? _site_set!(f, v, k, tile, I) : _concat_set!(Base.tail(fields), v, k - n, tile, I)
end

Base.similar(::SiteView, ::Type{T}, dims::Dims) where {T} = Array{T}(undef, dims)
Base.copy(q::SiteView) = copyto!(similar(q), q)
Base.unaliascopy(::SiteView) = throw(ArgumentError(
    "overlapping site-view operations require an explicit copy of the source; use copy(source)"))

# Expose backing storage identity to Julia's alias handling, including SubArrays
# of site views and collections that share fields in different orders.
Base.dataids(q::SiteView) = _site_dataids(_site_fields(q.state), q.tile)

# Tuple fields (single arrays, MultiHaloArray) give a statically sized id tuple;
# a nested field contributes the ids of its own fields.
@inline _storage_dataids(a::AbstractSingleHaloArray, tile) = Base.dataids(_cell_storage(a, tile))
@inline _storage_dataids(c::AbstractHaloCollection, tile) = _site_dataids(_site_fields(c), tile)
@inline _site_dataids(::Tuple{}, _) = ()
@inline _site_dataids(fields::Tuple, tile) =
    (_storage_dataids(first(fields), tile)..., _site_dataids(Base.tail(fields), tile)...)
# Array field containers have a runtime field count, so their id tuple allocates.
_site_dataids(fields, tile) = Tuple(id for field in fields
    for id in _storage_dataids(field, tile))

# Avoid allocating a runtime-sized tuple of storage ids on ordinary copies and
# broadcasts. dataids remains the fallback for wrappers such as SubArray.
# The checks descend to the leaves, like `_storage_dataids`.
_storage_mightalias(f::AbstractSingleHaloArray, tile, a) = Base.mightalias(_cell_storage(f, tile), a)
_storage_mightalias(c::AbstractHaloCollection, tile, a) =
    any(field -> _storage_mightalias(field, tile, a), _site_fields(c))
Base.mightalias(q::SiteView, a::AbstractArray) = _storage_mightalias(q.state, q.tile, a)
Base.mightalias(a::AbstractArray, q::SiteView) = Base.mightalias(q, a)
Base.mightalias(q::SiteView, r::SiteView) = _storage_mightalias(q.state, q.tile, r)

# ─── Static site vectors ─────────────────────────────────────────────────────
# The number of leaf fields of a state whose field containers are tuples or
# NamedTuples all the way down to the single arrays (a MultiHaloArray, nested
# or not): every container is a type parameter, so the count is a property of
# the type. `_n_leaf_type` walks the type tree; `:foldable` lets the compiler
# evaluate the call on the concrete type at compile time, so `SVector` gets a
# static length. Without it, inference follows the recursion itself and gives
# up on nested collections (recursion limit), leaving the length dynamic. A
# collection with an Array-backed field container anywhere in the tree has a
# runtime field count: the walk reaches that container and throws a clear
# ArgumentError, with the explicit form to use instead.
Base.@assume_effects :foldable _n_leaf_type(::Type{<:AbstractSingleHaloArray}) = 1
Base.@assume_effects :foldable _n_leaf_type(::Type{<:FieldCollection{T,D,S,C}}) where
        {T,D,S,C<:Union{Tuple,NamedTuple}} =
    sum(_n_leaf_type, fieldtypes(C); init = 0)
_n_leaf_type(::Type{S}) where {S<:FieldCollection} = throw(ArgumentError(
    "sitevector needs the field count in the type: `$(nameof(S))` with field container " *
    "`$(nameof(fieldtype(S, :arrays)))` has a runtime number of fields (an Array-backed " *
    "ArrayOfHaloArray, possibly nested inside a MultiHaloArray). Use " *
    "`SVector{N}(siteview(state, I))` with an explicit N."))

"""
    sitevector(state, I::CartesianIndex[, tile]) -> SVector{N}
    sitevector(tile_parent(state, tile), I::CartesianIndex) -> SVector{N}

The `N` leaf fields of `state` at local padded-storage index `I` as a **static
vector**, in the order of [`siteview`](@ref). `N` is read off the type of
`state`, so it is a compile-time constant: a `MultiHaloArray` with fields
`(:E, :Mx, :My, :D)` gives an `SVector{4}`, a nested `MultiHaloArray((; c, w))`
of two such collections with 4 + 6 fields an `SVector{10}`, and a single halo
array an `SVector{1}`. The result is a copy of the values, not a view; write
back through `siteview(state, I) .= v`.

Supported: single halo arrays, and collections whose field containers are
tuples or `NamedTuple`s down to the single arrays, i.e. a `MultiHaloArray`
whose fields are single arrays or again such collections. An `ArrayOfHaloArray`
backed by an `Array` has a runtime field count, so it is not supported, whether
on its own or nested inside a `MultiHaloArray`; the call throws an
`ArgumentError` saying so, and `SVector{N}(siteview(state, I))` with an
explicit `N` is the form to use there.

Compiles to one load per field, no allocation, the same code as writing
`SVector{N}(siteview(state, I))` by hand. Index and tile conventions are those
of [`siteview`](@ref); `@inbounds` at the call site elides the storage bounds
checks.

```julia
q = sitevector(u, I)           # SVector{4,Float64} for a four-field u
F = flux(q)                    # static-array arithmetic, no allocation
siteview(du, I) .-= F          # write back through the view
```

The second form reads from the storage of one tile, resolved once with
[`tile_parent`](@ref): the form for a tile loop that reads and writes, on every
backend (see [`siteview`](@ref)).

```julia
tile_foreach(u) do t
    s, d = tile_parent(u, t), tile_parent(du, t)       # the tile, looked up once
    @inbounds for I in interior_cells(CellRanges(u, t))
        siteview(d, I) .= rhs(sitevector(s, I))
    end
end
```
"""
Base.@propagate_inbounds sitevector(state::AbstractHaloArray, I::CartesianIndex,
        tile::Union{Nothing,Integer}=nothing) =
    SVector{_n_leaf_type(typeof(state))}(siteview(state, I, tile))

# ─── Site access on the storage of one tile ──────────────────────────────────
# `tile_parent(state, t)` resolves the tile once: a padded array for a single
# field, a tuple or NamedTuple of them for a collection, nested like the
# collection. The accessors below read and write a site of that storage, so a
# tile loop looks the tile up once instead of at every access. On threaded
# storage the lookup is a load from the tile vector, which the compiler cannot
# hoist out of a loop that also stores into field arrays.
const _TileFields = Union{Tuple,NamedTuple}
const _TileStorage = Union{DenseArray,_TileFields}

# Wrapper that marks resolved tile storage as the `state` of a SiteView.
struct _TileSite{S<:_TileStorage}
    storage::S
end

@inline _n_leaves(::DenseArray) = 1
@inline _n_leaves(s::NamedTuple) = _n_leaves(Tuple(s))
@inline _n_leaves(::Tuple{}) = 0
@inline _n_leaves(s::Tuple) = _n_leaves(first(s)) + _n_leaves(Base.tail(s))

@inline _leaf_eltype(a::DenseArray) = eltype(a)
# The tile storage of an Array-backed ArrayOfHaloArray is an array of arrays: a
# runtime field count, which the tuple walks below cannot unroll.
_leaf_eltype(::DenseArray{<:DenseArray}) = _throw_array_of_storages()
@noinline _throw_array_of_storages() = throw(ArgumentError(
    "site access on tile storage needs a padded array or a tuple/NamedTuple of them; " *
    "an array of arrays (the tile storage of an Array-backed ArrayOfHaloArray) has a " *
    "runtime number of fields. Use `siteview(state, I, tile)` on the collection."))
@inline _leaf_eltype(s::_TileFields) = promote_type(map(_leaf_eltype, Tuple(s))...)

# The values of every leaf at `I`, as one flat tuple in declaration order.
Base.@propagate_inbounds _site_values(a::DenseArray, I) = (a[I],)
_site_values(::DenseArray{<:DenseArray}, I) = _throw_array_of_storages()
Base.@propagate_inbounds _site_values(s::NamedTuple, I) = _site_values(Tuple(s), I)
@inline _site_values(::Tuple{}, I) = ()
Base.@propagate_inbounds _site_values(s::Tuple, I) =
    (_site_values(first(s), I)..., _site_values(Base.tail(s), I)...)

# Leaf `k` at `I`: the walk of `_concat_get`, on arrays instead of fields.
Base.@propagate_inbounds _leaf_get(a::DenseArray, k::Int, I) = a[I]
Base.@propagate_inbounds _leaf_get(s::NamedTuple, k::Int, I) = _leaf_get(Tuple(s), k, I)
_leaf_get(::Tuple{}, k::Int, I) = throw(BoundsError((), k))
Base.@propagate_inbounds function _leaf_get(s::Tuple, k::Int, I)
    f = first(s); n = _n_leaves(f)
    return k <= n ? _leaf_get(f, k, I) : _leaf_get(Base.tail(s), k - n, I)
end
Base.@propagate_inbounds _leaf_set!(a::DenseArray, v, k::Int, I) = (a[I] = v; nothing)
Base.@propagate_inbounds _leaf_set!(s::NamedTuple, v, k::Int, I) = _leaf_set!(Tuple(s), v, k, I)
_leaf_set!(::Tuple{}, v, k::Int, I) = throw(BoundsError((), k))
Base.@propagate_inbounds function _leaf_set!(s::Tuple, v, k::Int, I)
    f = first(s); n = _n_leaves(f)
    return k <= n ? _leaf_set!(f, v, k, I) : _leaf_set!(Base.tail(s), v, k - n, I)
end

@inline _site_size(s::_TileSite) = (_n_leaves(s.storage),)
Base.@propagate_inbounds _site_get(s::_TileSite, k::Int, ::Nothing, I) = _leaf_get(s.storage, k, I)
Base.@propagate_inbounds _site_set!(s::_TileSite, v, k::Int, ::Nothing, I) = _leaf_set!(s.storage, v, k, I)
Base.parent(q::SiteView{T,N,<:_TileSite}) where {T,N} = q.state.storage

@inline _leaf_dataids(a::DenseArray) = Base.dataids(a)
@inline _leaf_dataids(s::NamedTuple) = _leaf_dataids(Tuple(s))
@inline _leaf_dataids(::Tuple{}) = ()
@inline _leaf_dataids(s::Tuple) = (_leaf_dataids(first(s))..., _leaf_dataids(Base.tail(s))...)
Base.dataids(q::SiteView{T,N,<:_TileSite}) where {T,N} = _leaf_dataids(q.state.storage)
_leaf_mightalias(f::DenseArray, a) = Base.mightalias(f, a)
_leaf_mightalias(s::_TileFields, a) = any(f -> _leaf_mightalias(f, a), Tuple(s))
_storage_mightalias(s::_TileSite, ::Nothing, a) = _leaf_mightalias(s.storage, a)

@inline siteview(storage::_TileStorage, I::CartesianIndex) =
    SiteView{_leaf_eltype(storage),1,_TileSite{typeof(storage)},typeof(I),Nothing}(
        _TileSite(storage), I, nothing)

Base.@propagate_inbounds sitevector(storage::_TileStorage, I::CartesianIndex) =
    SVector(_site_values(storage, I))
