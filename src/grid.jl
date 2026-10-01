using StaticArrays

# ============================================================
# Cell geometry: coordinates and widths stored as a MultiHaloArray on the
# layout of a solution array, plus the metric (volume / face-area / normal)
# helpers that a finite-volume or DG kernel evaluates per cell.
#
# The geometry is "just another field": every rank/tile fills its interior by
# evaluating the axis descriptions at the global index, then one halo
# synchronization completes the ghosts — exchange across ranks/tiles/periodic
# edges, and a FunctionBC continuing the axis on physical edges.
# ============================================================

# ---- coordinate systems -----------------------------------------------------

"""
    CoordinateSystem

Supertype of the coordinate-system tags ([`Cartesian`](@ref), [`Polar`](@ref),
[`Cylindrical`](@ref), [`Spherical`](@ref)) that select the field names and the
metric terms of a [`cell_geometry`](@ref). The geometry stores one coordinate
field and one width field per spatial dimension, in this order; the tag decides
how they combine into [`cell_volume`](@ref) and [`face_area`](@ref).
"""
abstract type CoordinateSystem end

"Cartesian coordinates `x, y, z` (widths `hx, hy, hz`); the metric is the product of widths."
struct Cartesian <: CoordinateSystem end
"Polar coordinates `r, θ` (2D); volumes and face areas carry the factor `r`."
struct Polar <: CoordinateSystem end
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
coordinate_names(::Polar,       ::Val{2}) = (:r, :θ)
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

# A Float field on `u`'s layout (same backend, halo, tiling, topology, device).
# Its boundary condition is `Periodic` where the topology is periodic (the
# topology check requires it; the exchange then wraps the coordinate) and, on
# physical edges, a FunctionBC that evaluates `value(global_index)` on the
# ghost slab — so one `synchronize_halo!` completes the ghosts on every backend.
function _geometry_bc(periodic::NTuple{N,Bool}, value::V) where {N,V}
    fill = FunctionBC((ghost, edge, side, dim, hw, origin) ->
        (ghost .= (J -> value(Tuple(origin - oneunit(origin) + J))).(CartesianIndices(ghost)); nothing))
    return ntuple(d -> periodic[d] ? (Periodic(), Periodic()) : (fill, fill), Val(N))
end

# Zero-initialised storage on the source's device: corner ghosts are never
# written by an exchange or boundary condition (as for every field), so they
# must at least be deterministic.
_zeros_like(a::AbstractArray, ::Type{T}) where {T} = fill!(similar(a, T), zero(T))

function _geometry_field(u::LocalHaloArray, ::Type{T}, value) where {T}
    bc = _geometry_bc(infer_periodicity(u.boundary_condition), value)
    return LocalHaloArray(_zeros_like(parent(u), T), halo_width(u), bc)
end
function _geometry_field(u::HaloArray, ::Type{T}, value) where {T}
    bc = _geometry_bc(u.topology.periodic_boundary_condition, value)
    return build_haloarray_from_data(_zeros_like(parent(u), T), halo_width(u), u.topology, bc)
end
function _geometry_field(u::ThreadedHaloArray{S,N,A,Halo}, ::Type{T}, value) where {S,N,A,Halo,T}
    bc   = _geometry_bc(u.topology.periodic_boundary_condition, value)
    data = [_zeros_like(tile_parent(u, t), T) for t in 1:tile_count(u)]
    return ThreadedHaloArray{T,N,eltype(data),Halo,typeof(u.topology),typeof(bc),typeof(u.backend)}(
        data, tile_size(u), u.topology, bc, u.backend)
end

"""
    cell_geometry(u, axes; system=Cartesian(), T=Float64) -> MultiHaloArray

The cell centres and widths of the grid `u` lives on, as a [`MultiHaloArray`](@ref)
of `2N` scalar fields on `u`'s layout: the coordinates named by
[`coordinate_names`](@ref)`(system, Val(N))` followed by the widths (`hx`, `hr`,
…). `axes` gives one [`AbstractAxis`](@ref) per spatial dimension of `u`, in
terms of global cell indices; `u` may be any single halo array or collection
(the field dimensions of a collection are ignored).

The interior is filled from the global index on every rank and tile and the
ghosts are completed by one [`synchronize_halo!`](@ref): across ranks, tiles
and periodic edges by the exchange (so periodic ghosts wrap like any field),
and on physical edges by a [`FunctionBC`](@ref) that continues the axis
outside the domain. Corner ghosts (outside the domain in more than one
direction) are left at zero, as for every field. Synchronizing it again is
harmless. Read it like any field: `g.x`, `field_storages(g)`, `tile_parent(g, t)`, and evaluate the metric
with [`cell_center`](@ref), [`cell_volume`](@ref), [`face_area`](@ref),
[`face_normal`](@ref), …

# Example
```julia
u = LocalHaloArray(Float64, (64, 32), 1)
g = cell_geometry(u, (UniformAxis(0, 1), EdgeAxis(cell_edges((0, 0.1, 1), (8, 24)))))
for I in CartesianIndices(interior_range(u))      # padded-storage indices
    V = cell_volume(Cartesian(), g, I)
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
        value = k <= N ? (I -> T(_axis_center(ax, n[d], I[d]))) :
                         (I -> T(_axis_width(ax, n[d], I[d])))
        f = _geometry_field(ref, T, value)
        fill_from_global_indices!(value, f)
        synchronize_halo!(f)
        f
    end
    return MultiHaloArray(NamedTuple{(names..., _width_names(names)...)}(fields))
end
cell_geometry(u::AbstractHaloArray, axes::AbstractAxis...; kwargs...) = cell_geometry(u, axes; kwargs...)

_geometry_field_source(u::AbstractSingleHaloArray) = u
_geometry_field_source(c::AbstractHaloCollection)   = _first_field(c)

# ---- metric helpers ----------------------------------------------------------
# Every helper takes the geometry collection `g`, a padded-storage index `I`
# (CartesianIndex) and, for a threaded geometry, the tile id — the same
# arguments as `siteview`, which they use to read the fields at the site. The
# fields are positional in the site vector: coordinate `d` is `q[d]`, width `d`
# is `q[N + d]`, so names never enter the hot path. The spatial dimension `N`
# is a type parameter of the collection, so every `ntuple` below unrolls.

@inline _geo_ndims(::AbstractHaloCollection{T,D,S}) where {T,D,S} = S
@inline _site(g::AbstractHaloCollection, I, tile) = siteview(g, I, tile)
# `length(q)` is the field count (2N), known from the collection type.
@inline _coord(q, d) = @inbounds q[d]
@inline _width(q, d) = @inbounds q[length(q) ÷ 2 + d]

"""
    cell_center(system, g, I[, tile]) -> SVector

Coordinates of the cell at padded-storage index `I` of the geometry `g`
(see [`cell_geometry`](@ref)); `tile` selects the tile of a threaded geometry,
as for [`siteview`](@ref).
"""
@inline function cell_center(::CoordinateSystem, g::AbstractHaloCollection, I, tile=nothing)
    q = _site(g, I, tile)
    return SVector(ntuple(d -> _coord(q, d), Val(_geo_ndims(g))))
end

"""
    cell_width(system, g, I[, tile]) -> SVector

Extent of the cell at padded-storage index `I` along every axis.
"""
@inline function cell_width(::CoordinateSystem, g::AbstractHaloCollection, I, tile=nothing)
    q = _site(g, I, tile)
    N = Val(_geo_ndims(g))
    return SVector(ntuple(d -> _width(q, d), N))
end

@inline _prod_widths(q, ::Val{N}) where {N} = prod(ntuple(d -> _width(q, d), Val(N)))
@inline _prod_widths_except(q, ::Val{N}, ::Val{D}) where {N,D} =
    prod(ntuple(d -> d == D ? one(eltype(q)) : _width(q, d), Val(N)))

# Exact angular integrals of the spherical metric over the cell.
@inline _sin_integral(θ, hθ) = cos(θ - hθ / 2) - cos(θ + hθ / 2)          # ∫ sin θ dθ
@inline _r2_integral(r, hr)  = (r * r + hr * hr / 12) * hr                # ∫ r² dr

"""
    cell_volume(system, g, I[, tile]) -> Real

Volume of the cell at padded-storage index `I`: the exact integral of the
metric over the cell (`∏ h` in Cartesian; `r hr ∏ h` in polar and cylindrical;
`∫r²dr ∫sinθdθ hφ` in spherical). Missing dimensions count as unit extent.
"""
@inline cell_volume(::Cartesian, g::AbstractHaloCollection, I, tile=nothing) =
    _prod_widths(_site(g, I, tile), Val(_geo_ndims(g)))
@inline function cell_volume(::Union{Polar,Cylindrical}, g::AbstractHaloCollection, I, tile=nothing)
    q = _site(g, I, tile)
    return _coord(q, 1) * _prod_widths(q, Val(_geo_ndims(g)))
end
@inline function cell_volume(::Spherical, g::AbstractHaloCollection, I, tile=nothing)
    q = _site(g, I, tile)
    N = _geo_ndims(g)
    v = _r2_integral(_coord(q, 1), _width(q, 1))
    N >= 2 && (v *= _sin_integral(_coord(q, 2), _width(q, 2)))
    N >= 3 && (v *= _width(q, 3))
    return v
end

"""
    face_center(system, g, Dim(d), I[, tile]) -> SVector

Coordinates of the **plus** face of cell `I` along axis `d` (the face shared
with `I + e_d`); the minus face of `I` is the plus face of `I - e_d`.
"""
@inline function face_center(sys::CoordinateSystem, g::AbstractHaloCollection, ::Dim{D}, I, tile=nothing) where {D}
    q = _site(g, I, tile)
    N = _geo_ndims(g)
    c = SVector(ntuple(d -> _coord(q, d), Val(N)))
    return c + SVector{N,eltype(c)}(versors(Val(N))[D]) * (_width(q, D) / 2)
end

"""
    face_distance(g, Dim(d), I[, tile]) -> Real

Distance between the centres of cell `I` and its neighbour `I + e_d` across the
plus face along axis `d`: what a face gradient divides by.
"""
@inline function face_distance(g::AbstractHaloCollection, ::Dim{D}, I::CartesianIndex, tile=nothing) where {D}
    N = _geo_ndims(g)
    J = I + unit_vector(Val(N), D)
    return (_width(_site(g, I, tile), D) + _width(_site(g, J, tile), D)) / 2
end

"""
    face_area(system, g, Dim(d), I[, tile]) -> Real

Area of the plus face of cell `I` along axis `d`, exact for the metric
(`∏_{j≠d} h_j` in Cartesian, with the radius evaluated on the face for radial
faces in cylindrical and spherical coordinates). Missing dimensions count as
unit extent.
"""
@inline face_area(::Cartesian, g::AbstractHaloCollection, ::Dim{D}, I, tile=nothing) where {D} =
    _prod_widths_except(_site(g, I, tile), Val(_geo_ndims(g)), Val(D))
@inline function face_area(::Polar, g::AbstractHaloCollection, ::Dim{D}, I, tile=nothing) where {D}
    q = _site(g, I, tile)
    # radial face: an arc r₊ hθ; θ face: a radial segment hr
    return D == 1 ? (_coord(q, 1) + _width(q, 1) / 2) * _width(q, 2) : _width(q, 1)
end
@inline function face_area(::Cylindrical, g::AbstractHaloCollection, ::Dim{D}, I, tile=nothing) where {D}
    q = _site(g, I, tile)
    N = _geo_ndims(g)
    r = D == 1 ? _coord(q, 1) + _width(q, 1) / 2 : _coord(q, 1)
    return r * _prod_widths_except(q, Val(N), Val(D))
end
@inline function face_area(::Spherical, g::AbstractHaloCollection, ::Dim{D}, I, tile=nothing) where {D}
    q = _site(g, I, tile)
    N = _geo_ndims(g)
    r, hr = _coord(q, 1), _width(q, 1)
    if D == 1
        rp = r + hr / 2
        a  = rp * rp
        N >= 2 && (a *= _sin_integral(_coord(q, 2), _width(q, 2)))
        N >= 3 && (a *= _width(q, 3))
        return a
    elseif D == 2
        a = r * hr * sin(_coord(q, 2) + _width(q, 2) / 2)
        N >= 3 && (a *= _width(q, 3))
        return a
    else
        return r * hr * _width(q, 2)
    end
end

"""
    face_normal(system, g, Dim(d), I[, tile]) -> SVector

Outward unit normal of the plus face of cell `I` along axis `d`, in the local
coordinate basis. On a tensor-product grid this is the `d`-th basis vector,
independent of `I`; the argument form is kept so kernels need not change for
grids whose normals vary per face.
"""
@inline face_normal(::CoordinateSystem, g::AbstractHaloCollection{T}, ::Dim{D}, I, tile=nothing) where {T,D} =
    SVector{_geo_ndims(g),T}(versors(Val(_geo_ndims(g)))[D])
