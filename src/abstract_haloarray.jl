abstract type AbstractHaloArray{T,N} <: AbstractArray{T,N} end

abstract type AbstractSingleHaloArray{T,N} <: AbstractHaloArray{T,N} end
abstract type AbstractDistributedHaloArray{T,N} <: AbstractSingleHaloArray{T,N} end
abstract type AbstractSerialHaloArray{T,N} <: AbstractSingleHaloArray{T,N} end

abstract type AbstractHaloCollection{T,N} <: AbstractHaloArray{T,N} end

abstract type AbstractHaloBackend end
struct MPIHaloBackend <: AbstractHaloBackend end
struct LocalHaloBackend <: AbstractHaloBackend end
struct ThreadedHaloBackend <: AbstractHaloBackend end

"""
    halo_backend(x) -> AbstractHaloBackend

Return a singleton trait describing the storage/execution backend of a halo
array or halo collection. Use this for dispatch when code needs separate MPI,
local, or threaded implementations while still accepting collection wrappers.
"""
function halo_backend end

@inline halo_backend(halo::AbstractSingleHaloArray) = halo_backend(typeof(halo))

"""
    owned_size(halo)

Return the owned interior size of a halo container on the current process.

For `HaloArray` this is the owned MPI subdomain size, not the global
distributed size. For serial containers it is equal to their full logical
interior size.
"""
@inline owned_size(halo::AbstractHaloArray) = interior_size(halo)
@inline owned_size(halo::AbstractHaloArray, i::Int) = owned_size(halo)[i]

"""
    owned_axes(halo)

Return the owned-cell axes of a halo container on the current process.

Use `axes(halo)` for the global logical axes and `owned_axes(halo)` when
looping over data that this process can update directly.
"""
@inline owned_axes(halo::AbstractHaloArray) = map(Base.OneTo, owned_size(halo))
@inline owned_axes(halo::AbstractHaloArray, i::Int) = owned_axes(halo)[i]

function storage_size end
function owned_to_global_index end
function global_to_storage_index end
function is_root end

@inline halo_width(arr::AbstractArray{<:AbstractSingleHaloArray}) = halo_width(first(arr))

@inline function _check_global_scalar_indices(halo::AbstractHaloArray, I::Tuple)
    length(I) == ndims(halo) || throw(BoundsError(halo, I))
    all(d -> first(axes(halo, d)) <= I[d] <= last(axes(halo, d)), eachindex(I)) ||
        throw(BoundsError(halo, I))
    return I
end

Base.getindex(halo::AbstractHaloArray, I::CartesianIndex) = getindex(halo, Tuple(I)...)
Base.setindex!(halo::AbstractHaloArray, value, I::CartesianIndex) =
    setindex!(halo, value, Tuple(I)...)
