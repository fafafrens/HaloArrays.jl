# Local storage access: no halo exchange or intermediate field-container map.
@inline function _require_flat_fields(state::AbstractHaloCollection)
    all(field -> field isa AbstractSingleHaloArray, eachfield(state)) ||
        throw(ArgumentError("field access requires a flat collection of single halo arrays; nested collections are not supported"))
    return nothing
end

@inline _cell_storage(field::Union{LocalHaloArray,HaloArray}, ::Nothing) = parent(field)
@inline _cell_storage(field::ThreadedHaloArray, ::Nothing) =
    throw(ArgumentError("threaded field access requires an explicit tile id"))
Base.@propagate_inbounds function _cell_storage(field::AbstractSingleHaloArray, tile::Integer)
    @boundscheck 1 <= tile <= tile_count(field) || throw(BoundsError(field, tile))
    return tile_parent(field, tile)
end

function _check_field_access(values::AbstractVector, state::AbstractHaloCollection{T,N,S},
        I::CartesianIndex{D}, tile) where {T,N,S,D}
    D == S || throw(DimensionMismatch("expected a $S-dimensional storage index, got $D"))
    length(values) == prod(field_shape(state)) ||
        throw(DimensionMismatch("vector length must equal the number of fields"))
    # Validate all storage indices before mutating a destination.
    for field in eachfield(state)
        checkbounds(_cell_storage(field, tile), I)
    end
    return nothing
end

"""
    gather_fields!(dest, state, I::CartesianIndex[, tile])

Copy all fields at local padded-storage index `I` into the preallocated vector
`dest`. Accepts flat `ArrayOfHaloArray` and `MultiHaloArray` collections whose
immediate fields are all single halo arrays. Nested collections are rejected
with `ArgumentError` before any writes, even under `@inbounds`. Array field containers
use column-major order; named collections use declaration order. The vector
length must equal `prod(field_shape(state))`.

`I` uses the storage coordinates returned by `interior_cells(CellRanges(state))`
and `interior_faces(FaceRanges(state), dim)`, including allocated halo cells.
The caller must synchronize halos before reading them. No communication is
performed. Threaded collections require an explicit `tile` id; Local/MPI
collections use their local storage (and optionally accept tile id `1`).

Returns `dest`. No intermediate field container or vector is allocated. These
scalar accessors are intended for CPU storage; they do not launch GPU kernels.
The destination must not alias the state storage.

Validation follows Julia's bounds-checking convention: calling with `@inbounds`
skips vector-length, index-dimension, and storage/tile bounds checks. The caller
must ensure these are valid. Ordinary calls validate before writing.

```julia
u = ArrayOfHaloArray(LocalHaloArray, Float64, (3,), (16,), 1;
                     boundary_condition=:periodic)
v = zeros(3)
I = first(interior_cells(CellRanges(u)))
gather_fields!(v, u, I)
```
"""
Base.@propagate_inbounds function gather_fields!(dest::AbstractVector, state::AbstractHaloCollection,
        I::CartesianIndex, tile::Union{Nothing,Integer}=nothing)
    _require_flat_fields(state)
    @boundscheck _check_field_access(dest, state, I, tile)
    @inbounds for (j, field) in zip(eachindex(dest), eachfield(state))
        dest[j] = _cell_storage(field, tile)[I]
    end
    return dest
end

"""
    scatter_fields!(state, I::CartesianIndex, values[, tile])

Overwrite all fields at local padded-storage index `I` from `values` and return
`state`. Uses the field ordering, tile selection, and index conventions of
[`gather_fields!`](@ref). No communication or halo synchronization is performed;
refresh halos before subsequent stencil reads. `values` must not alias state
storage. Element conversion follows ordinary array assignment.
"""
Base.@propagate_inbounds function scatter_fields!(state::AbstractHaloCollection, I::CartesianIndex,
        values::AbstractVector, tile::Union{Nothing,Integer}=nothing)
    _require_flat_fields(state)
    @boundscheck _check_field_access(values, state, I, tile)
    @inbounds for (j, field) in zip(eachindex(values), eachfield(state))
        _cell_storage(field, tile)[I] = values[j]
    end
    return state
end

"""
    add_fields!(state, I::CartesianIndex, values, scale[, tile])

Accumulate `scale * values` into all fields at local padded-storage index `I`
and return `state`. The target entries must already be initialized. Uses the
ordering, tile selection, and non-aliasing contract of [`gather_fields!`](@ref).
No communication or halo synchronization is performed.

For a finite-volume face flux `F`, use `add_fields!(du, IL, F, -invdx)` and
`add_fields!(du, IR, F, invdx)` on initialized storage.
"""
Base.@propagate_inbounds function add_fields!(state::AbstractHaloCollection, I::CartesianIndex,
        values::AbstractVector, scale, tile::Union{Nothing,Integer}=nothing)
    _require_flat_fields(state)
    @boundscheck _check_field_access(values, state, I, tile)
    @inbounds for (j, field) in zip(eachindex(values), eachfield(state))
        storage = _cell_storage(field, tile)
        storage[I] += scale * values[j]
    end
    return state
end
