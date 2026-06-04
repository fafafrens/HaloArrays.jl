"""
    LocalHaloArray(T, owned_dims, halo; boundary_condition=:repeating)
    LocalHaloArray(data::AbstractArray, halo, boundary_condition)

A single-process halo array: a plain array padded with `halo` ghost cells on
every side of the `owned_dims` interior, carrying the boundary condition to
apply when the halo is refreshed.

This is the simplest backend — no MPI, no threads. Logical indexing, `axes`,
and `eachindex` address the owned cells only; the ghost padding is hidden.

# Arguments
- `T`: element type (defaults to `Float64` if omitted).
- `owned_dims::NTuple{N,Int}`: size of the owned (interior) region.
- `halo::Int`: ghost-cell width on each side, in each dimension.
- `boundary_condition`: how ghosts are filled by [`synchronize_halo!`](@ref) /
  [`boundary_condition!`](@ref). A symbol (`:periodic`, `:repeating`,
  `:reflecting`, `:antireflecting`), a boundary-condition instance, or a
  per-dimension tuple of `(left, right)` pairs. See [`Periodic`](@ref) etc.

# Examples
```julia
u = LocalHaloArray(Float64, (64, 64), 1; boundary_condition=:periodic)
interior_view(u) .= 1.0      # write the owned cells
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

# size, axes, length, eltype, ndims, parent, owned_axes, eachindex, iterate
# inherited from AbstractSingleHaloArray / AbstractArray.

is_root(::LocalHaloArray; root::Integer=0) = (root == 0)
# isactive, get_comm inherited from AbstractSerialHaloArray

@inline halo_width(::Type{<:LocalHaloArray{T,N,A,Halo}}) where {T,N,A,Halo} = Halo
@inline halo_width(::LocalHaloArray{T,N,A,Halo}) where {T,N,A,Halo} = Halo

@inline function interior_size(halo::LocalHaloArray{T,N}) where {T,N}
    h = halo_width(halo)
    return ntuple(i -> size(halo.data, i) - 2 * h, Val(N))
end

@inline storage_size(halo::LocalHaloArray) = size(halo.data)
@inline storage_size(halo::LocalHaloArray, i::Int) = size(halo.data, i)

@inline function interior_range(halo::LocalHaloArray)
    h = halo_width(halo)
    N = ndims(halo)
    return ntuple(i -> (h + 1):(storage_size(halo, i) - h), Val(N))
end

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

@inline get_send_view(s, d, array::LocalHaloArray) = get_send_view(s, d, parent(array), halo_width(array))
@inline get_recv_view(s, d, array::LocalHaloArray) = get_recv_view(s, d, parent(array), halo_width(array))
# versors, Base.similar dispatchers, Base.map!/map inherited from AbstractSingleHaloArray

function Base.similar(halo::LocalHaloArray{T,N,A,Halo,BCondition}, ::Type{AA},
        dims::Dims{M}) where {T,N,A,Halo,BCondition,AA,M}
    M == N || throw(DimensionMismatch("LocalHaloArray similar dims must have $N dimensions"))
    fullsize = ntuple(i -> Int(dims[i]) + 2 * halo_width(halo), Val(N))
    data = similar(parent(halo), AA, fullsize)
    return LocalHaloArray(data, halo_width(halo), halo.boundary_condition)
end

Base.similar(halo::LocalHaloArray{T,N,A,Halo,BCondition}, ::Type{AA},
    dims::NTuple{M,<:Integer}) where {T,N,A,Halo,BCondition,AA,M} =
    similar(halo, AA, ntuple(d -> Int(dims[d]), Val(M)))

# Base.copy, Base.zero, Base.fill!, Base.copyto!, fill_interior!,
# fill_from_local_indices!, Base.foreach, arithmetic, norm,
# owned_axes, eachindex, iterate, versors, Base.similar dispatchers,
# Base.map!/map — all inherited from AbstractSingleHaloArray

function Base.map(f, src::Vararg{LocalHaloArray,N}) where {N}
    dest = similar(src[1])
    map!(f, dest, src...)
    return dest
end


function fill_from_global_indices!(f, halo::LocalHaloArray)
    interior = interior_view(halo)
    for I in CartesianIndices(interior)
        interior[I] = f(Tuple(I))
    end
    return halo
end

owned_to_global_index(::LocalHaloArray, owned_idx::NTuple{N,<:Integer}) where {N} = owned_idx
global_to_storage_index(halo::LocalHaloArray, global_idx::NTuple{N,<:Integer}) where {N} =
    all(i -> 1 <= global_idx[i] <= owned_size(halo, i), 1:N) ? ntuple(i -> global_idx[i] + halo_width(halo), Val(N)) : nothing
global_size(halo::LocalHaloArray) = owned_size(halo)

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
    print(io, "LocalHaloArray of global size ", size(obj), " (owned size: ", owned_size(obj), ", storage size: ", storage_size(obj), "), halo width: ", halo_width(obj), "\n")
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
