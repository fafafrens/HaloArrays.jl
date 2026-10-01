using StaticArrays

# ============================================================
# Cell geometry: coordinates and widths stored as a MultiHaloArray on the
# layout of a solution array, plus the metric (volume / face-area / normal)
# helpers that a finite-volume or DG kernel evaluates per cell.
#
# The geometry is "just another field": every rank/tile fills its own storage
# — ghost cells included — by evaluating the axis descriptions at the global
# index, so it is complete at construction and needs no exchange. Ghost cells
# outside the domain continue the axis (extended, never wrapped), so distances
# across a periodic edge stay consistent.
# ============================================================

# ---- coordinate systems -----------------------------------------------------

"""
    CoordinateSystem

Supertype of the coordinate-system tags ([`Cartesian`](@ref),
[`Cylindrical`](@ref), [`Spherical`](@ref)) that select the field names and the
metric terms of a [`cell_geometry`](@ref). The geometry stores one coordinate
field and one width field per spatial dimension, in this order; the tag decides
how they combine into [`cell_volume`](@ref) and [`face_area`](@ref).
"""
abstract type CoordinateSystem end

"Cartesian coordinates `x, y, z` (widths `hx, hy, hz`); the metric is the product of widths."
struct Cartesian <: CoordinateSystem end
"""
Cylindrical coordinates with the radius first: `r` (1D), `r, z` (2D,
axisymmetric), `r, θ, z` (3D). Volumes and face areas carry the factor `r`.
Missing angular/axial extents count as unit length.
"""
struct Cylindrical <: CoordinateSystem end
"""
Spherical coordinates with the radius first: `r` (1D), `r, θ` (2D), `r, θ, φ`
(3D), `θ` the polar angle. Volumes and face areas carry `r² sin θ`, integrated
exactly over the cell. Missing angular extents count as unit angle.
"""
struct Spherical <: CoordinateSystem end

"""
    coordinate_names(system, Val(N)) -> NTuple{N,Symbol}

The coordinate field names of an `N`-dimensional [`cell_geometry`](@ref) in the
given coordinate system. The width fields are the same names prefixed by `h`.
"""
coordinate_names(::Cartesian,   ::Val{N}) where {N} = ntuple(d -> (:x, :y, :z)[d], Val(N))
coordinate_names(::Cylindrical, ::Val{1}) = (:r,)
coordinate_names(::Cylindrical, ::Val{2}) = (:r, :z)
coordinate_names(::Cylindrical, ::Val{3}) = (:r, :θ, :z)
coordinate_names(::Spherical,   ::Val{1}) = (:r,)
coordinate_names(::Spherical,   ::Val{2}) = (:r, :θ)
coordinate_names(::Spherical,   ::Val{3}) = (:r, :θ, :φ)
coordinate_names(sys::CoordinateSystem, ::Val{N}) where {N} =
    throw(ArgumentError("$(typeof(sys)) supports 1 to 3 dimensions, got $N"))

@inline _width_names(names::NTuple{N,Symbol}) where {N} = map(s -> Symbol(:h, s), names)

# ---- axis descriptions ------------------------------------------------------

"""
    AbstractAxis

One spatial axis of a [`cell_geometry`](@ref): the cell centre and width as a
function of the **global** cell index along that axis. Outside the domain the
axis is continued with the edge cell's width, so ghost cells get consistent
coordinates. See [`UniformAxis`](@ref), [`EdgeAxis`](@ref).
"""
abstract type AbstractAxis end

"""
    UniformAxis(a, b)

Equal-width cells covering `[a, b]`; the cell count comes from the array the
geometry is built on.
"""
struct UniformAxis{T<:Real} <: AbstractAxis
    a::T
    b::T
end
UniformAxis(a::Real, b::Real) = UniformAxis(promote(float(a), float(b))...)

"""
    EdgeAxis(edges)

Cells bounded by the sorted `edges` vector (or range): cell `i` spans
`edges[i]` to `edges[i+1]`, so `length(edges)` must be the cell count plus one.
See [`cell_edges`](@ref) for a graded distribution.
"""
struct EdgeAxis{V<:AbstractVector} <: AbstractAxis
    edges::V
    function EdgeAxis(edges::AbstractVector)
        length(edges) >= 2 || throw(ArgumentError("EdgeAxis needs at least two edges"))
        issorted(edges) || throw(ArgumentError("EdgeAxis edges must be sorted"))
        new{typeof(edges)}(edges)
    end
end

"""
    cell_edges(breaks, counts) -> Vector

Edges of a piecewise-uniform (graded) axis: `counts[k]` equal cells between
`breaks[k]` and `breaks[k+1]`. For example the element distribution
`cell_edges((0, 0.001, 0.01, 0.15, 1), (5, 15, 50, 50))`.
"""
function cell_edges(breaks, counts)
    length(breaks) == length(counts) + 1 ||
        throw(ArgumentError("need one more break than count"))
    T = promote_type(map(b -> typeof(float(b)), Tuple(breaks))..., Float64)
    edges = T[breaks[1]]
    for k in eachindex(counts)
        counts[k] > 0 || throw(ArgumentError("counts must be positive"))
        a, b = breaks[k], breaks[k + 1]
        for i in 1:counts[k]
            push!(edges, a + (b - a) * i / counts[k])
        end
    end
    return edges
end

@inline _axis_count(ax::UniformAxis, n::Int) = n
@inline function _axis_count(ax::EdgeAxis, n::Int)
    length(ax.edges) == n + 1 ||
        throw(DimensionMismatch("EdgeAxis has $(length(ax.edges) - 1) cells, the array has $n"))
    return n
end

@inline _axis_width(ax::UniformAxis, n::Int, i::Int) = (ax.b - ax.a) / n
@inline _axis_center(ax::UniformAxis, n::Int, i::Int) = ax.a + (i - 0.5) * (ax.b - ax.a) / n

# Outside 1:n continue with the edge cell's width.
@inline function _axis_width(ax::EdgeAxis, n::Int, i::Int)
    j = clamp(i, 1, n)
    @inbounds return ax.edges[j + 1] - ax.edges[j]
end
@inline function _axis_center(ax::EdgeAxis, n::Int, i::Int)
    if 1 <= i <= n
        @inbounds return (ax.edges[i] + ax.edges[i + 1]) / 2
    elseif i < 1
        @inbounds return ax.edges[1] - (0.5 - i) * (ax.edges[2] - ax.edges[1])
    else
        @inbounds return ax.edges[n + 1] + (i - n - 0.5) * (ax.edges[n + 1] - ax.edges[n])
    end
end

# ---- construction -----------------------------------------------------------

# A Float field on `u`'s layout (same backend, halo, tiling, topology, device)
# whose ghost cells are never refreshed: NoBoundaryCondition on physical edges,
# Periodic where the topology is periodic (the topology check requires it).
_geometry_bc(periodic::NTuple{N,Bool}) where {N} =
    ntuple(d -> periodic[d] ? (Periodic(), Periodic()) :
                              (NoBoundaryCondition(), NoBoundaryCondition()), Val(N))

function _geometry_field(u::LocalHaloArray, ::Type{T}) where {T}
    bc = _geometry_bc(infer_periodicity(u.boundary_condition))
    return LocalHaloArray(similar(parent(u), T), halo_width(u), bc)
end
function _geometry_field(u::HaloArray, ::Type{T}) where {T}
    bc = _geometry_bc(u.topology.periodic_boundary_condition)
    return build_haloarray_from_data(similar(parent(u), T), halo_width(u), u.topology, bc)
end
function _geometry_field(u::ThreadedHaloArray{S,N,A,Halo}, ::Type{T}) where {S,N,A,Halo,T}
    bc   = _geometry_bc(u.topology.periodic_boundary_condition)
    data = [similar(tile_parent(u, t), T) for t in 1:tile_count(u)]
    return ThreadedHaloArray{T,N,eltype(data),Halo,typeof(u.topology),typeof(bc),typeof(u.backend)}(
        data, tile_size(u), u.topology, bc, u.backend)
end

# Fill every storage cell (ghosts included) of each tile from `f(I_global)`,
# as one broadcast per tile (device-agnostic: the index offset is a constant).
function _fill_storage_from_global!(f::F, h::AbstractSingleHaloArray{T,N}) where {F,T,N}
    hw = halo_width(h)
    for t in 1:tile_count(h)
        origin = interior_to_global_index(h, t, ntuple(_ -> 1, Val(N)))
        off    = CartesianIndex(ntuple(d -> origin[d] - hw - 1, Val(N)))
        data   = tile_parent(h, t)
        data .= (I -> f(Tuple(I + off))).(CartesianIndices(data))
    end
    return h
end

"""
    cell_geometry(u, axes; system=Cartesian(), T=Float64) -> MultiHaloArray

The cell centres and widths of the grid `u` lives on, as a [`MultiHaloArray`](@ref)
of `2N` scalar fields on `u`'s layout: the coordinates named by
[`coordinate_names`](@ref)`(system, Val(N))` followed by the widths (`hx`, `hr`,
…). `axes` gives one [`AbstractAxis`](@ref) per spatial dimension of `u`, in
terms of global cell indices; `u` may be any single halo array or collection
(the field dimensions of a collection are ignored).

Every cell of every rank and tile — ghost cells included — is filled from the
global index at construction, so the result is complete and must **not** be
passed to [`synchronize_halo!`](@ref) (on a periodic edge that would wrap the
extended ghost coordinates). Read it like any field: `g.x`, `field_storages(g)`,
`tile_parent(g, t)`, and evaluate the metric with [`cell_center`](@ref),
[`cell_volume`](@ref), [`face_area`](@ref), [`face_normal`](@ref), …

# Example
```julia
u = LocalHaloArray(Float64, (64, 32), 1)
g = cell_geometry(u, (UniformAxis(0, 1), EdgeAxis(cell_edges((0, 0.1, 1), (8, 24)))))
gs = field_storages(g)                      # NamedTuple of storages (x, y, hx, hy)
for I in CartesianIndices(interior_range(u))
    V = cell_volume(Cartesian(), gs, I)
end
```
"""
function cell_geometry(u::AbstractHaloArray, axes::Tuple;
        system::CoordinateSystem=Cartesian(), T::Type=Float64)
    ref = _geometry_field_source(u)
    N   = ndims(ref)
    length(axes) == N || throw(DimensionMismatch("$(length(axes)) axes for a $N-dimensional grid"))
    all(a -> a isa AbstractAxis, axes) || throw(ArgumentError("axes must be AbstractAxis values"))
    n = size(ref)
    foreach(d -> _axis_count(axes[d], n[d]), 1:N)
    names = coordinate_names(system, Val(N))
    fields = ntuple(Val(2N)) do k
        d  = k <= N ? k : k - N
        ax = axes[d]
        f  = _geometry_field(ref, T)
        if k <= N
            _fill_storage_from_global!(I -> T(_axis_center(ax, n[d], I[d])), f)
        else
            _fill_storage_from_global!(I -> T(_axis_width(ax, n[d], I[d])), f)
        end
        f
    end
    return MultiHaloArray(NamedTuple{(names..., _width_names(names)...)}(fields))
end
cell_geometry(u::AbstractHaloArray, axes::AbstractAxis...; kwargs...) = cell_geometry(u, axes; kwargs...)

_geometry_field_source(u::AbstractSingleHaloArray) = u
_geometry_field_source(c::AbstractHaloCollection)   = _first_field(c)

# ---- metric helpers ----------------------------------------------------------
# `gs` is the NamedTuple of storages of a geometry — `field_storages(g)` on a
# Local/MPI array, `tile_parent(g, t)` on a threaded one — and `I` a storage index into
# it (CartesianIndex or integers). Fields are positional: coordinate `d` is
# `gs[d]`, width `d` is `gs[N + d]`, so the helpers never touch names.

@inline _geo_ndims(gs::NamedTuple) = length(gs) ÷ 2
@inline _coord(gs, d, I) = @inbounds gs[d][I]
@inline _width(gs, d, I) = @inbounds gs[_geo_ndims(gs) + d][I]
@inline _unit(::Val{N}, ::Val{D}, ::Type{T}) where {N,D,T} =
    SVector(ntuple(j -> j == D ? one(T) : zero(T), Val(N)))

"""
    cell_center(system, gs, I) -> SVector

Coordinates of the cell at storage index `I` of the geometry storages `gs`
(see [`cell_geometry`](@ref)).
"""
@inline cell_center(::CoordinateSystem, gs::NamedTuple, I) =
    SVector(ntuple(d -> _coord(gs, d, I), Val(_geo_ndims(gs))))

"""
    cell_width(system, gs, I) -> SVector

Extent of the cell at storage index `I` along every axis.
"""
@inline cell_width(::CoordinateSystem, gs::NamedTuple, I) =
    SVector(ntuple(d -> _width(gs, d, I), Val(_geo_ndims(gs))))

@inline _prod_widths(gs, I, ::Val{N}) where {N} = prod(ntuple(d -> _width(gs, d, I), Val(N)))
@inline _prod_widths_except(gs, I, ::Val{N}, ::Val{D}) where {N,D} =
    prod(ntuple(d -> d == D ? one(eltype(gs[1])) : _width(gs, d, I), Val(N)))

# Exact angular integrals of the spherical metric over the cell.
@inline _sin_integral(θ, hθ) = cos(θ - hθ / 2) - cos(θ + hθ / 2)          # ∫ sin θ dθ
@inline _r2_integral(r, hr)  = (r * r + hr * hr / 12) * hr                # ∫ r² dr

"""
    cell_volume(system, gs, I) -> Real

Volume of the cell at storage index `I`: the exact integral of the metric over
the cell (`∏ h` in Cartesian; `r hr ∏ h` in cylindrical; `∫r²dr ∫sinθdθ hφ` in
spherical). Missing dimensions count as unit extent.
"""
@inline cell_volume(::Cartesian, gs::NamedTuple, I) = _prod_widths(gs, I, Val(_geo_ndims(gs)))
@inline cell_volume(::Cylindrical, gs::NamedTuple, I) =
    _coord(gs, 1, I) * _prod_widths(gs, I, Val(_geo_ndims(gs)))
@inline function cell_volume(::Spherical, gs::NamedTuple, I)
    N = _geo_ndims(gs)
    v = _r2_integral(_coord(gs, 1, I), _width(gs, 1, I))
    N >= 2 && (v *= _sin_integral(_coord(gs, 2, I), _width(gs, 2, I)))
    N >= 3 && (v *= _width(gs, 3, I))
    return v
end

"""
    face_center(system, gs, Dim(d), I) -> SVector

Coordinates of the **plus** face of cell `I` along axis `d` (the face shared
with `I + e_d`); the minus face of `I` is the plus face of `I - e_d`.
"""
@inline function face_center(sys::CoordinateSystem, gs::NamedTuple, ::Dim{D}, I) where {D}
    c = cell_center(sys, gs, I)
    return c + _unit(Val(length(c)), Val(D), eltype(c)) * (_width(gs, D, I) / 2)
end

"""
    face_distance(gs, Dim(d), I) -> Real

Distance between the centres of cell `I` and its neighbour `I + e_d` across the
plus face along axis `d`: what a face gradient divides by.
"""
@inline function face_distance(gs::NamedTuple, ::Dim{D}, I::CartesianIndex{N}) where {D,N}
    J = I + unit_vector(Val(N), D)
    return (_width(gs, D, I) + _width(gs, D, J)) / 2
end
@inline face_distance(gs::NamedTuple, d::Dim, I::Vararg{Integer}) = face_distance(gs, d, CartesianIndex(I))

"""
    face_area(system, gs, Dim(d), I) -> Real

Area of the plus face of cell `I` along axis `d`, exact for the metric
(`∏_{j≠d} h_j` in Cartesian, with the radius evaluated on the face for radial
faces in cylindrical and spherical coordinates). Missing dimensions count as
unit extent.
"""
@inline face_area(::Cartesian, gs::NamedTuple, ::Dim{D}, I) where {D} =
    _prod_widths_except(gs, I, Val(_geo_ndims(gs)), Val(D))
@inline function face_area(::Cylindrical, gs::NamedTuple, ::Dim{D}, I) where {D}
    N = _geo_ndims(gs)
    r = D == 1 ? _coord(gs, 1, I) + _width(gs, 1, I) / 2 : _coord(gs, 1, I)
    return r * _prod_widths_except(gs, I, Val(N), Val(D))
end
@inline function face_area(::Spherical, gs::NamedTuple, ::Dim{D}, I) where {D}
    N  = _geo_ndims(gs)
    r, hr = _coord(gs, 1, I), _width(gs, 1, I)
    if D == 1
        rp = r + hr / 2
        a  = rp * rp
        N >= 2 && (a *= _sin_integral(_coord(gs, 2, I), _width(gs, 2, I)))
        N >= 3 && (a *= _width(gs, 3, I))
        return a
    elseif D == 2
        a = r * hr * sin(_coord(gs, 2, I) + _width(gs, 2, I) / 2)
        N >= 3 && (a *= _width(gs, 3, I))
        return a
    else
        return r * hr * _width(gs, 2, I)
    end
end

"""
    face_normal(system, gs, Dim(d), I) -> SVector

Outward unit normal of the plus face of cell `I` along axis `d`, in the local
coordinate basis. On a tensor-product grid this is the `d`-th basis vector,
independent of `I`; the argument form is kept so kernels need not change for
grids whose normals vary per face.
"""
@inline face_normal(::CoordinateSystem, gs::NamedTuple, ::Dim{D}, I) where {D} =
    _unit(Val(_geo_ndims(gs)), Val(D), eltype(gs[1]))
