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
_ka_backend(x::SubArray) = _ka_backend(parent(x))              # view storage, e.g. a stepped view of a GPU array
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

# ---- the rule faces of a block in one launch -----------------------------
# One kernel for any set of prepared faces: the per-face fill passes one face,
# `_fused_fill!` every rule face of the block. Work item i → (face f, cell j
# in the face) from the cumulative face sizes, then the face's prepared rule
# on that ghost cell — the same `_fill_cell!` as the CPU loop. The faces are
# disjoint (corner ghosts belong to no face), so the order is free. The face
# selection is a recursion on the tuple's tail — one `f == q` branch per face,
# each with a concrete (geometry, rule) type; the tuple shrinks at every call,
# so inference terminates and it inlines to straight-line code (a closure
# over the scan variables would be boxed, a recursion on Val{q} is widened by
# inference; neither compiles on a GPU). The branch is uniform within a face.
# The scan and the unravelling are Int32: N integer divisions per cell, which
# GPUs emulate in 64-bit. (A dedicated all-periodic kernel with Int32 index
# tuples measured 0.062 vs 0.057 ms for this one at 48⁴ on Metal, so there is
# only this one.)

HaloArrays._face_backend(a::AbstractArray) = _ka_backend(a)

@inline function _unravel(j::Int32, ext::NTuple{N,Int32}) where {N}
    c = ntuple(_ -> Int32(0), Val(N))
    r = j
    for μ in 1:N
        c = Base.setindex(c, r % ext[μ], μ)
        r ÷= ext[μ]
    end
    return c
end

@inline _apply_face!(a, ::Tuple{}, f, j, q) = nothing
@inline function _apply_face!(a, faces::Tuple, f, j, q)
    face = first(faces)
    if face !== nothing && f == q          # `face !== nothing` is decided by the element type
        g, rule, ext = face
        c = _unravel(j, ext)
        HaloArrays._fill_cell!(a, rule, g, first(g.ghosts) + CartesianIndex(map(Int, c)))
    end
    return _apply_face!(a, Base.tail(faces), f, j, q + Int32(1))
end

@kernel function _faces_kernel!(a, faces, counts, ::Val{F}) where {F}
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
    _apply_face!(a, faces, f, i - base, Int32(1))
end

# `faces`: a tuple of prepared faces `(geometry, rule)`, `nothing` for a face
# with nothing to fill. Each face is handed to the kernel with its Int32 extent
# (computed once here: in the kernel it would be Int64 arithmetic per cell).
function _launch_faces!(backend::KA.Backend, a, faces::NTuple{F,Any}) where {F}
    counts = map(face -> face === nothing ? Int32(0) : Int32(length(face[1].ghosts)), faces)
    total = Int(sum(counts))
    total == 0 && return nothing
    device_faces = map(face -> face === nothing ? nothing : (face[1], face[2], Int32.(size(face[1].ghosts))), faces)
    _faces_kernel!(backend)(a, device_faces, counts, Val(F); ndrange=total)
    return nothing
end

# one face (the per-face fill on GPU storage)
HaloArrays._traverse!(backend::KA.Backend, a, rule, g::HaloArrays.FaceGeometry) =
    _launch_faces!(backend, a, ((g, rule),))

# The prepared face (geometry, rule) for face f of `halo`, or nothing when the
# face is not filled by a rule. `f` is a type parameter so Side/Dim are static.
@inline function _prepared_face(halo, ::Val{f}) where {f}
    d = (f + 1) ÷ 2
    s = f - 2(d - 1)
    bc = HaloArrays._rule_condition(halo, Side(s), Dim(d))
    bc === nothing && return nothing
    return HaloArrays._prepare_face(halo, Side(s), Dim(d), bc, nothing)
end

# every rule face of the block
function HaloArrays._fused_fill!(halo::Union{LocalHaloArray{T,N},HaloArray{T,N}}) where {T,N}
    a = parent(halo)
    backend = _ka_backend(a)
    (backend === nothing || backend isa KA.CPU) && return false
    length(a) <= typemax(Int32) || return false            # larger blocks take the per-face kernels
    _launch_faces!(backend, a, ntuple(f -> _prepared_face(halo, Val(f)), Val(2N)))
    return true
end

end
