module HaloArraysKernelAbstractionsExt

# GPU implementations of the face-copy seams of src/boundary.jl and
# src/mpi_support.jl (see the "GPU seams" block in boundary.jl). Loads with
# KernelAbstractions and GPUArraysCore, i.e. with any GPU array package; only
# `AbstractGPUArray` storage takes the kernels, everything else keeps the
# broadcast path.

import HaloArrays
using HaloArrays: LocalHaloArray, HaloArray, Periodic, halo_width, Side, Dim
using KernelAbstractions
using GPUArraysCore: AbstractGPUArray
const KA = KernelAbstractions

# ---- which queue writes this storage ------------------------------------
# The kernels index the storage array by offsets, so a view qualifies only
# when every index is a contiguous range over a dense parent (every HaloArrays
# face view is one). Anything else — stepped views, host arrays, storage that
# is not a GPU array (`BitArray`, ...) — keeps the broadcast path.

_storage(x::SubArray) = parent(x)
_storage(x::DenseArray) = x
_offset(x::SubArray) = map(r -> first(r) - 1, x.indices)
_offset(x::DenseArray) = ntuple(_ -> 0, ndims(x))

_unit_indexed(x::DenseArray) = true
_unit_indexed(x::SubArray) = parent(x) isa DenseArray && all(r -> r isa AbstractUnitRange, x.indices)

_ka_backend(x::AbstractGPUArray) = KA.get_backend(x)
_ka_backend(::AbstractArray) = nothing                         # host storage: broadcast path of src/

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

# ---- one face of a rule = one kernel -------------------------------------
# The same `_fill_cell!` as the CPU loop, one work item per ghost cell: the
# prepared rule (an isbits stencil or condition) runs on the device.

HaloArrays._face_backend(a::AbstractGPUArray) = KA.get_backend(a)
HaloArrays._face_backend(a::SubArray) = HaloArrays._face_backend(parent(a))   # view storage, e.g. a stepped view of a GPU array

@kernel function _fill_face_kernel!(a, rule, g, first_ghost)
    J = @index(Global, Cartesian)
    HaloArrays._fill_cell!(a, rule, g, first_ghost + J - oneunit(J))
end

function HaloArrays._traverse!(backend::KA.Backend, a, rule, g::HaloArrays.FaceGeometry)
    _fill_face_kernel!(backend)(a, rule, g, first(g.ghosts); ndrange=size(g.ghosts))
    return nothing
end

# ---- every rule face of a block in one launch ----------------------------
# Work item i → (face f, cell c in the face) from the cumulative face sizes,
# then the face's prepared rule on that ghost cell — the same `_fill_cell!` as
# the per-face kernel and the CPU loop. The faces are disjoint (corner ghosts
# belong to no face), so the order is free. The face selection is unrolled at
# compile time over the face tuple, so each branch sees one concrete
# (geometry, rule) type and the kernel specialises once per combination of
# conditions; the branch is uniform within a face. The scan and the
# unravelling are Int32: N integer divisions per cell, which GPUs emulate in
# 64-bit.

@inline function _unravel(j::Int32, ext::NTuple{N,Int32}) where {N}
    c = ntuple(_ -> Int32(0), Val(N))
    r = j
    for μ in 1:N
        c = Base.setindex(c, r % ext[μ], μ)
        r ÷= ext[μ]
    end
    return c
end

@inline _face_extent(::Nothing) = nothing
@inline _face_extent(face) = Int32.(size(face[1].ghosts))
@inline _face_count(::Nothing) = Int32(0)
@inline _face_count(face) = Int32(length(face[1].ghosts))

# Apply the selected face's rule: one `if f == q` branch per face, emitted at
# compile time (a generated function; a closure over the scan variables would
# be boxed, and a Val recursion is widened by inference — neither compiles on
# a GPU). An inactive face is `nothing` and its branch is dead code.
@generated function _apply_face!(a, faces::NTuple{F,Any}, exts, f, j) where {F}
    branches = map(1:F) do q
        quote
            @inbounds face = faces[$q]
            if face !== nothing && f == $q
                g = face[1]
                @inbounds c = _unravel(j, exts[$q])
                HaloArrays._fill_cell!(a, face[2], g, first(g.ghosts) + CartesianIndex(map(Int, c)))
            end
        end
    end
    return Expr(:block, branches..., :(return nothing))
end

@kernel function _faces_kernel!(a, faces, exts, counts, ::Val{F}) where {F}
    i = Int32(@index(Global, Linear)) - Int32(1)
    # cumulative scan: advance to face k+1 only while still at face k (the faces
    # have different sizes on a non-cubic block, so a later count may be smaller)
    f = Int32(1)
    base = Int32(0)
    for k in 1:F-1
        @inbounds if f == k && i >= base + counts[k]
            base += counts[k]
            f = Int32(k + 1)
        end
    end
    _apply_face!(a, faces, exts, f, i - base)
end

# The prepared face (geometry, rule) for face f of `halo`, or nothing when the
# face is not filled by a rule. `f` is a type parameter so Side/Dim are static.
@inline function _prepared_face(halo, ::Val{f}) where {f}
    d = (f + 1) ÷ 2
    s = f - 2(d - 1)
    bc = HaloArrays._rule_condition(halo, Side(s), Dim(d))
    bc === nothing && return nothing
    return HaloArrays._prepare_face(halo, Side(s), Dim(d), bc, nothing)
end

function HaloArrays._fused_fill!(halo::Union{LocalHaloArray{T,N},HaloArray{T,N}}) where {T,N}
    a = parent(halo)
    backend = _ka_backend(a)
    (backend === nothing || backend isa KA.CPU) && return false
    length(a) <= typemax(Int32) || return false            # larger blocks take the per-face kernels
    halo isa LocalHaloArray && _fused_periodic_fill!(halo, backend, a) && return true
    faces = ntuple(f -> _prepared_face(halo, Val(f)), Val(2N))
    counts = map(_face_count, faces)
    total = Int(sum(counts))
    total == 0 && return true
    exts = map(f -> f === nothing ? ntuple(_ -> Int32(0), Val(N)) : _face_extent(f), faces)
    _faces_kernel!(backend)(a, faces, exts, counts, Val(2N); ndrange=total)
    return true
end


# ---- all-periodic single block: every face in one launch, Int32 ----------
# Work item i → (face, cell in the face) as in the generic kernel, then the
# ghost cell and its periodic source (the opposite edge) as Int32 index
# tuples, no CartesianIndex. Kept beside the generic launch because it is
# measurably faster for this (common) case: 48⁴ Metal, halo 2, 0.114 vs
# 0.139 ms. `false` means "not this case".

@kernel function _periodic_faces_kernel!(a, n::NTuple{N,Int32}, h::Int32, counts, ::Val{N}) where {N}
    i = Int32(@index(Global, Linear)) - Int32(1)
    @inbounds begin
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

function _fused_periodic_fill!(halo::LocalHaloArray{T,N}, backend, a) where {T,N}
    all(bc -> bc[1] isa Periodic && bc[2] isa Periodic, halo.boundary_condition) || return false
    h = halo_width(halo)
    n = size(halo)
    counts = ntuple(k -> Int32(h * prod(ntuple(μ -> μ == (k + 1) ÷ 2 ? 1 : n[μ], Val(N)))), 2N)
    total = Int(sum(counts))
    total == 0 && return true
    _periodic_faces_kernel!(backend)(a, Int32.(n), Int32(h), counts, Val(N); ndrange=total)
    return true
end

end
