module HaloArraysKernelAbstractionsExt

# GPU implementations of the face-copy seams of src/boundary.jl and
# src/mpi_support.jl (see the "GPU seams" block in boundary.jl). Loaded with
# `using KernelAbstractions`; host (`Array`) storage keeps the broadcast path.

import HaloArrays
using HaloArrays: LocalHaloArray, Periodic, halo_width
using KernelAbstractions
const KA = KernelAbstractions

# ---- which queue writes this storage ------------------------------------
# The kernels index the storage array by offsets, so a view qualifies only
# when every index is a contiguous range over a dense parent (every HaloArrays
# face view is one). Anything else — stepped views, host arrays, storage with
# no KernelAbstractions backend (e.g. `BitArray`) — keeps the broadcast path.

_storage(x::SubArray) = parent(x)
_storage(x::DenseArray) = x
_offset(x::SubArray) = map(r -> first(r) - 1, x.indices)
_offset(x::DenseArray) = ntuple(_ -> 0, ndims(x))

_unit_indexed(x::DenseArray) = true
_unit_indexed(x::SubArray) = parent(x) isa DenseArray && all(r -> r isa AbstractUnitRange, x.indices)

_ka_backend(::Array) = nothing                                 # host storage: broadcast path of src/
function _ka_backend(x::AbstractArray)
    try
        KA.get_backend(x)
    catch err
        err isa ArgumentError ? nothing : rethrow()            # no backend registered for this storage
    end
end

HaloArrays._slab_backend(dest::Union{DenseArray,SubArray}, src::Union{DenseArray,SubArray}) =
    _unit_indexed(dest) && _unit_indexed(src) ? _ka_backend(_storage(dest)) : nothing

# ---- one slab copy = one kernel ------------------------------------------

HaloArrays._copy_slab!(::KA.CPU, dest, src) = (dest .= src; nothing)

# No `@Const` on `src`: a periodic ghost fill copies between two slabs of the
# same storage array, so `dest` and `src` may be the same object (disjoint
# ranges, but `@Const` promises no aliasing at all).
@kernel function _slab_copy_kernel!(dest, d0, src, s0)
    J = @index(Global, NTuple)
    @inbounds dest[(J .+ d0)...] = src[(J .+ s0)...]
end

function HaloArrays._copy_slab!(backend::KA.Backend, dest, src)
    size(dest) == size(src) || throw(DimensionMismatch(
        "slab copy: destination $(size(dest)), source $(size(src))"))
    isempty(dest) && return nothing
    _slab_copy_kernel!(backend)(_storage(dest), _offset(dest), _storage(src), _offset(src);
                                ndrange=size(dest))
    return nothing
end

# The MPI sends read device memory directly: the pack kernels must have run.
# A backend without `synchronize` (JLArrays' reference backend) executes eagerly.
function HaloArrays._device_synchronize(x::AbstractArray)
    backend = _ka_backend(x)
    backend === nothing && return nothing
    applicable(KA.synchronize, backend) && KA.synchronize(backend)
    return nothing
end

# ---- all-periodic single block: every face in one launch ----------------
# Work item i → (face, cell in the face) from the cumulative face sizes, then
# the ghost cell and its periodic source (the opposite edge) by offsets. The
# faces are disjoint (corner ghosts belong to no face), so the order is free.
# Int32 throughout: the unravelling costs N integer divisions per cell, which
# GPUs emulate in 64-bit (the Int64 kernel is slower than the broadcasts).

@inline function _unravel(j::Int32, ext::NTuple{N,Int32}) where {N}
    c = ntuple(_ -> Int32(0), Val(N))
    r = j
    for μ in 1:N
        c = Base.setindex(c, r % ext[μ], μ)
        r ÷= ext[μ]
    end
    return c
end

@kernel function _periodic_faces_kernel!(a, n::NTuple{N,Int32}, h::Int32, counts, ::Val{N}) where {N}
    i = Int32(@index(Global, Linear)) - Int32(1)
    @inbounds begin
        # cumulative scan: advance to face k+1 only while still at face k (the faces
        # have different sizes on a non-cubic block, so a later count may be smaller)
        f = Int32(1)
        base = Int32(0)
        for k in 1:2N-1
            if f == k && i >= base + counts[k]
                base += counts[k]
                f = Int32(k + 1)
            end
        end
        d = (f + Int32(1)) ÷ Int32(2)                      # dimension of the face
        s = f - Int32(2) * (d - Int32(1))                  # side 1 (low) or 2 (high)
        ext = ntuple(μ -> μ == d ? h : n[μ], Val(N))
        c = _unravel(i - base, ext)                        # 0-based coordinates in the face
        dest = ntuple(μ -> μ == d ? (s == Int32(1) ? Int32(1) + c[μ] : n[μ] + h + Int32(1) + c[μ]) :
                                    h + Int32(1) + c[μ], Val(N))
        src  = ntuple(μ -> μ == d ? (s == Int32(1) ? n[μ] + Int32(1) + c[μ] : h + Int32(1) + c[μ]) :
                                    h + Int32(1) + c[μ], Val(N))
        a[dest...] = a[src...]
    end
end

function HaloArrays._fused_periodic_fill!(halo::LocalHaloArray{T,N}) where {T,N}
    a = parent(halo)
    backend = _ka_backend(a)
    (backend === nothing || backend isa KA.CPU) && return false
    all(bc -> bc[1] isa Periodic && bc[2] isa Periodic, halo.boundary_condition) || return false
    length(a) <= typemax(Int32) || return false            # larger blocks take the per-face kernels
    h = halo_width(halo)
    n = size(halo)
    counts = ntuple(k -> Int32(h * prod(ntuple(μ -> μ == (k + 1) ÷ 2 ? 1 : n[μ], Val(N)))), 2N)
    total = Int(sum(counts))
    total == 0 && return true
    _periodic_faces_kernel!(backend)(a, Int32.(n), Int32(h), counts, Val(N); ndrange=total)
    return true
end

end
