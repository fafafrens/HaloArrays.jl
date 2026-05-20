abstract type AbstractHaloArray{T,N} <: AbstractArray{T,N} end

abstract type AbstractSingleHaloArray{T,N} <: AbstractHaloArray{T,N} end
abstract type AbstractDistributedHaloArray{T,N} <: AbstractSingleHaloArray{T,N} end
abstract type AbstractSerialHaloArray{T,N} <: AbstractSingleHaloArray{T,N} end

abstract type AbstractHaloCollection{T,N} <: AbstractHaloArray{T,N} end

"""
    local_size(halo)

Return the owned interior size of a halo container on the current process.

For `HaloArray` this is the local MPI subdomain size, not the global
distributed size. For serial containers it is equal to their full logical
interior size.
"""
@inline local_size(halo::AbstractHaloArray) = interior_size(halo)
@inline local_size(halo::AbstractHaloArray, i::Int) = local_size(halo)[i]

"""
    local_axes(halo)

Return the local owned-cell axes of a halo container on the current process.

Use `axes(halo)` for the global logical axes and `local_axes(halo)` when
looping over data that this process can update directly.
"""
@inline local_axes(halo::AbstractHaloArray) = map(Base.OneTo, local_size(halo))
@inline local_axes(halo::AbstractHaloArray, i::Int) = local_axes(halo)[i]

@inline function _check_global_scalar_indices(halo::AbstractHaloArray, I::Tuple)
    length(I) == ndims(halo) || throw(BoundsError(halo, I))
    all(d -> first(axes(halo, d)) <= I[d] <= last(axes(halo, d)), eachindex(I)) ||
        throw(BoundsError(halo, I))
    return I
end

Base.getindex(halo::AbstractHaloArray, I::CartesianIndex) = getindex(halo, Tuple(I)...)
Base.setindex!(halo::AbstractHaloArray, value, I::CartesianIndex) =
    setindex!(halo, value, Tuple(I)...)
