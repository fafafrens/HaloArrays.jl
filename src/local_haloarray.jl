"""
    LocalHaloArray(T, owned_dims, halo; boundary_condition=:repeating)
    LocalHaloArray(data::AbstractArray, halo, boundary_condition)

A single-process halo array: a plain array padded with `halo` ghost cells on
every side of the `owned_dims` interior, carrying the boundary condition to
apply when the halo is refreshed.

This is the simplest backend — no MPI, no threads. Logical indexing, `axes`,
and `eachindex` address the interior cells only; the ghost padding is hidden.

# Arguments
- `T`: element type (defaults to `Float64` if omitted).
- `owned_dims::NTuple{N,Int}`: size of the interior region.
- `halo::Int`: ghost-cell width on each side, in each dimension.
- `boundary_condition`: how ghosts are filled by [`synchronize_halo!`](@ref) /
  [`boundary_condition!`](@ref). A symbol (`:periodic`, `:repeating`,
  `:reflecting`, `:antireflecting`), a boundary-condition instance, or a
  per-dimension tuple of `(left, right)` pairs. See [`Periodic`](@ref) etc.

# Examples
```julia
u = LocalHaloArray(Float64, (64, 64), 1; boundary_condition=:periodic)
interior_view(u) .= 1.0      # write the interior cells
synchronize_halo!(u)         # fill ghost cells from the boundary condition
```

See also [`HaloArray`](@ref) (MPI), [`ThreadedHaloArray`](@ref) (threads),
[`interior_view`](@ref), [`interior_range`](@ref).
"""
mutable struct LocalHaloArray{T,N,A,Halo,BCondition} <: AbstractSerialHaloArray{T,N}
    data::A
    boundary_condition::BCondition
end

@inline halo_backend(::Type{<:LocalHaloArray}) = LocalHaloBackend()

function LocalHaloArray(data::AbstractArray{T,N}, halo::Int, boundary_condition) where {T,N}
    bc = normalize_boundary_condition(boundary_condition, N)
    return LocalHaloArray{T,N,typeof(data),halo,typeof(bc)}(data, bc)
end

function LocalHaloArray(::Type{T}, owned_dims::NTuple{N,Int}, halo::Int; boundary_condition=:repeating) where {T,N}
    fullsize = ntuple(i -> owned_dims[i] + 2 * halo, Val(N))
    data = zeros(T, fullsize...)
    return LocalHaloArray(data, halo, boundary_condition)
end

function LocalHaloArray(owned_dims::NTuple{N,Int}, halo::Int; boundary_condition=:repeating) where {N}
    return LocalHaloArray(Float64, owned_dims, halo; boundary_condition)
end

# size, axes, length, eltype, ndims, parent, interior_axes, eachindex, iterate
# inherited from AbstractSingleHaloArray / AbstractArray.

is_root(::LocalHaloArray; root::Integer=0) = (root == 0)
# is_active, communicator inherited from AbstractSerialHaloArray

@inline halo_width(::Type{<:LocalHaloArray{T,N,A,Halo}}) where {T,N,A,Halo} = Halo
@inline halo_width(::LocalHaloArray{T,N,A,Halo}) where {T,N,A,Halo} = Halo

# storage_size / interior_size / interior_range come from AbstractSingleHaloArray.

@inline function full_range(halo::LocalHaloArray)
    N = ndims(halo)
    return ntuple(i -> 1:storage_size(halo, i), Val(N))
end

@inline function interior_view(halo::LocalHaloArray)
    ranges = interior_range(halo)
    @views return halo.data[ranges...]
end

function full_view(halo::LocalHaloArray)
    ranges = full_range(halo)
    @views return halo.data[ranges...]
end

@inline edge_view(array::LocalHaloArray, s::Side, d::Dim)  = edge_view(parent(array), s, d, halo_width(array))
@inline ghost_view(array::LocalHaloArray, s::Side, d::Dim) = ghost_view(parent(array), s, d, halo_width(array))
# versors, Base.similar dispatchers, Base.map!/map inherited from AbstractSingleHaloArray

function Base.similar(halo::LocalHaloArray{T,N,A,Halo,BCondition}, ::Type{AA},
        dims::Dims{M}) where {T,N,A,Halo,BCondition,AA,M}
    M == N || throw(DimensionMismatch("LocalHaloArray similar dims must have $N dimensions"))
    fullsize = ntuple(i -> Int(dims[i]) + 2 * halo_width(halo), Val(N))
    data = similar(parent(halo), AA, fullsize)
    return LocalHaloArray(data, halo_width(halo), halo.boundary_condition)
end

# Non-Int dims are normalized to Dims by Base's generic similar fallbacks.

# Base.copy, Base.zero, Base.fill!, Base.copyto!,
# Base.foreach, arithmetic, norm,
# interior_axes, eachindex, iterate, versors, Base.similar dispatchers,
# Base.map!/map — all inherited from AbstractSingleHaloArray

function Base.map(f, src::Vararg{LocalHaloArray,N}) where {N}
    dest = similar(src[1])
    map!(f, dest, src...)
    return dest
end


interior_to_global_index(::LocalHaloArray, owned_idx::NTuple{N,<:Integer}) where {N} = owned_idx
@inline function ghost_origin(halo::LocalHaloArray{T,N}, ::Side{S}, ::Dim{D}) where {T,N,S,D}
    owned = interior_size(halo)
    hw    = halo_width(halo)
    CartesianIndex(ntuple(i -> i == D ? (S == 1 ? 1 - hw : owned[i] + 1) : 1, Val(N)))
end
global_to_storage_index(halo::LocalHaloArray, global_idx::NTuple{N,<:Integer}) where {N} =
    all(i -> 1 <= global_idx[i] <= interior_size(halo, i), 1:N) ? ntuple(i -> global_idx[i] + halo_width(halo), Val(N)) : nothing
global_size(halo::LocalHaloArray) = interior_size(halo)

function Base.getindex(halo::LocalHaloArray, I::Vararg{Integer})
    idx = _check_global_scalar_indices(halo, I)
    storage_idx = global_to_storage_index(halo, idx)
    @inbounds return parent(halo)[storage_idx...]
end

function Base.setindex!(halo::LocalHaloArray, value, I::Vararg{Integer})
    idx = _check_global_scalar_indices(halo, I)
    storage_idx = global_to_storage_index(halo, idx)
    @inbounds parent(halo)[storage_idx...] = value
    return halo
end

function Base.show(io::IO, obj::LocalHaloArray)
    print(io, "LocalHaloArray of global size ", size(obj), " (interior size: ", interior_size(obj), ", storage size: ", storage_size(obj), "), halo width: ", halo_width(obj), "\n")
    print(io, "  eltype: ", eltype(obj), "\n")
    print(io, "  boundary_condition: ", obj.boundary_condition, "\n")
end

function Base.show(io::IO, mime::MIME"text/plain", obj::LocalHaloArray)
    println(io, "LocalHaloArray (storage size: ", storage_size(obj), ", halo width: ", halo_width(obj), ")")
    println(io, "  eltype: ", eltype(obj))
    println(io, "  boundary_condition: ", obj.boundary_condition)
    println(io, "  interior data preview:")
    show(io, mime, interior_view(obj))
end
