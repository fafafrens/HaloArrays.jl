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
    CellGeometry{Sys}

The cell geometry of a grid in the coordinate system `Sys`, returned by
[`cell_geometry`](@ref). It holds the cell centres and widths as a
[`MultiHaloArray`](@ref) on the grid's layout, and carries the coordinate
system in its type, so the metric helpers ([`cell_volume`](@ref),
[`face_area`](@ref), …) take the geometry alone and cannot be given the wrong
system.

`parent(g)` is the `MultiHaloArray`. Property access (`g.r`, `g.hθ`),
[`field_storages`](@ref), [`siteview`](@ref), `tile_parent`, `tile_count`,
[`synchronize_halo!`](@ref), `gather_haloarray` and `append_haloarray!` forward
to it. [`coordinate_system`](@ref)`(g)` returns the system.
"""
struct CellGeometry{Sys<:CoordinateSystem,F<:MultiHaloArray}
    fields::F
end
CellGeometry(::Sys, fields::MultiHaloArray) where {Sys<:CoordinateSystem} =
    CellGeometry{Sys,typeof(fields)}(fields)

"""
    coordinate_system(g::CellGeometry) -> CoordinateSystem

The coordinate system the geometry was built in (`Cartesian()`, `Polar()`, …).
"""
@inline coordinate_system(::CellGeometry{Sys}) where {Sys} = Sys()

@inline Base.parent(g::CellGeometry) = getfield(g, :fields)
@inline Base.getproperty(g::CellGeometry, name::Symbol) = getproperty(parent(g), name)
Base.propertynames(g::CellGeometry) = propertynames(parent(g))
@inline field_storages(g::CellGeometry) = field_storages(parent(g))
@inline siteview(g::CellGeometry, I::CartesianIndex, tile::Union{Nothing,Integer}=nothing) =
    siteview(parent(g), I, tile)
@inline tile_parent(g::CellGeometry, tile_id::Integer) = tile_parent(parent(g), tile_id)
@inline tile_count(g::CellGeometry) = tile_count(parent(g))
synchronize_halo!(g::CellGeometry; kwargs...) = (synchronize_halo!(parent(g); kwargs...); g)
gather_haloarray(g::CellGeometry; root::Int=0) = gather_haloarray(parent(g); root=root)
function Base.show(io::IO, g::CellGeometry)
    print(io, "CellGeometry{", nameof(typeof(coordinate_system(g))), "} with fields ",
        propertynames(g), " on a ", join(Base.tail(size(parent(g))), "×"), " grid")
end

"""
    cell_geometry(u, axes; system=Cartesian(), T=Float64) -> CellGeometry{typeof(system)}

The cell centres and widths of the grid `u` lives on, as a [`CellGeometry`](@ref)
holding `2N` scalar fields on `u`'s layout: the coordinates named by
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
harmless. Read it like any field: `g.x`, `field_storages(g)`, `tile_parent(g, t)`,
and evaluate the metric with [`cell_center`](@ref), [`cell_volume`](@ref),
[`face_area`](@ref), [`face_normal`](@ref), … — the geometry carries its
coordinate system, so they take no system argument.

# Example
```julia
u = LocalHaloArray(Float64, (64, 32), 1)
g = cell_geometry(u, (UniformAxis(0, 1), EdgeAxis(cell_edges((0, 0.1, 1), (8, 24)))))
for I in CartesianIndices(interior_range(u))      # padded-storage indices
    V = cell_volume(g, I)
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
    return CellGeometry(system, MultiHaloArray(NamedTuple{(names..., _width_names(names)...)}(fields)))
end
cell_geometry(u::AbstractHaloArray, axes::AbstractAxis...; kwargs...) = cell_geometry(u, axes; kwargs...)

_geometry_field_source(u::AbstractSingleHaloArray) = u
_geometry_field_source(c::AbstractHaloCollection)   = _geometry_field(c)   # the first leaf

# ---- metric formulas -----------------------------------------------------------
# The exact formulas, one method per coordinate system. Each takes the system,
# the geometry's fields `g` (the MultiHaloArray), a padded-storage index `I`
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

@inline function _cell_center(::CoordinateSystem, g::AbstractHaloCollection, I, tile)
    q = _site(g, I, tile)
    return SVector(ntuple(d -> _coord(q, d), Val(_geo_ndims(g))))
end

@inline function _cell_width(::CoordinateSystem, g::AbstractHaloCollection, I, tile)
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

@inline _cell_volume(::Cartesian, g::AbstractHaloCollection, I, tile) =
    _prod_widths(_site(g, I, tile), Val(_geo_ndims(g)))
@inline function _cell_volume(::Union{Polar,Cylindrical}, g::AbstractHaloCollection, I, tile)
    q = _site(g, I, tile)
    return _coord(q, 1) * _prod_widths(q, Val(_geo_ndims(g)))
end
@inline function _cell_volume(::Spherical, g::AbstractHaloCollection, I, tile)
    q = _site(g, I, tile)
    N = _geo_ndims(g)
    v = _r2_integral(_coord(q, 1), _width(q, 1))
    N >= 2 && (v *= _sin_integral(_coord(q, 2), _width(q, 2)))
    N >= 3 && (v *= _width(q, 3))
    return v
end

@inline function _face_center(sys::CoordinateSystem, g::AbstractHaloCollection, ::Dim{D}, I, tile) where {D}
    q = _site(g, I, tile)
    N = _geo_ndims(g)
    c = SVector(ntuple(d -> _coord(q, d), Val(N)))
    return c + SVector{N,eltype(c)}(versors(Val(N))[D]) * (_width(q, D) / 2)
end

# Lamé coefficient h_D at the cell centre: the physical length of a unit
# coordinate step along axis D (1 along a radius; r along an angle in the
# plane; r sin θ along the azimuth of a sphere).
@inline _scale_factor(::Cartesian,   q, ::Val{N}, ::Val{D}) where {N,D} = one(eltype(q))
@inline _scale_factor(::Polar,       q, ::Val{N}, ::Val{D}) where {N,D} = D == 2 ? _coord(q, 1) : one(eltype(q))
@inline _scale_factor(::Cylindrical, q, ::Val{N}, ::Val{D}) where {N,D} =
    (N == 3 && D == 2) ? _coord(q, 1) : one(eltype(q))
@inline _scale_factor(::Spherical,   q, ::Val{N}, ::Val{D}) where {N,D} =
    D == 1 ? one(eltype(q)) : D == 2 ? _coord(q, 1) : _coord(q, 1) * sin(_coord(q, 2))

@inline function _face_distance(sys::CoordinateSystem, g::AbstractHaloCollection, ::Dim{D}, I::CartesianIndex, tile) where {D}
    N = _geo_ndims(g)
    J = I + unit_vector(Val(N), D)
    q = _site(g, I, tile)
    return _scale_factor(sys, q, Val(N), Val(D)) * (_width(q, D) + _width(_site(g, J, tile), D)) / 2
end

@inline _face_area(::Cartesian, g::AbstractHaloCollection, ::Dim{D}, I, tile) where {D} =
    _prod_widths_except(_site(g, I, tile), Val(_geo_ndims(g)), Val(D))
@inline function _face_area(::Polar, g::AbstractHaloCollection, ::Dim{D}, I, tile) where {D}
    q = _site(g, I, tile)
    # radial face: an arc r₊ hθ; θ face: a radial segment hr
    return D == 1 ? (_coord(q, 1) + _width(q, 1) / 2) * _width(q, 2) : _width(q, 1)
end
@inline function _face_area(::Cylindrical, g::AbstractHaloCollection, ::Dim{D}, I, tile) where {D}
    q = _site(g, I, tile)
    N = _geo_ndims(g)
    r = D == 1 ? _coord(q, 1) + _width(q, 1) / 2 : _coord(q, 1)
    return r * _prod_widths_except(q, Val(N), Val(D))
end
@inline function _face_area(::Spherical, g::AbstractHaloCollection, ::Dim{D}, I, tile) where {D}
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

@inline _face_normal(::CoordinateSystem, g::AbstractHaloCollection{T}, ::Dim{D}, I, tile) where {T,D} =
    SVector{_geo_ndims(g),T}(versors(Val(_geo_ndims(g)))[D])


# ---- metric helpers --------------------------------------------------------------
# The public helpers take the geometry, which supplies its own system, plus a
# padded-storage index `I` and, on a threaded geometry, the tile id — the same
# arguments as `siteview`.

"""
    cell_center(g::CellGeometry, I[, tile]) -> SVector

Coordinates of the cell at padded-storage index `I` of the geometry `g`
(see [`cell_geometry`](@ref)); `tile` selects the tile of a threaded geometry,
as for [`siteview`](@ref).
"""
@inline cell_center(g::CellGeometry, I, tile=nothing) = _cell_center(coordinate_system(g), parent(g), I, tile)

"""
    cell_width(g::CellGeometry, I[, tile]) -> SVector

Extent of the cell at padded-storage index `I` along every axis.
"""
@inline cell_width(g::CellGeometry, I, tile=nothing) = _cell_width(coordinate_system(g), parent(g), I, tile)

"""
    cell_volume(g::CellGeometry, I[, tile]) -> Real

Volume of the cell at padded-storage index `I`: the exact integral of the
metric over the cell (`∏ h` in Cartesian; `r hr ∏ h` in polar and cylindrical;
`∫r²dr ∫sinθdθ hφ` in spherical). Missing dimensions count as unit extent.
"""
@inline cell_volume(g::CellGeometry, I, tile=nothing) = _cell_volume(coordinate_system(g), parent(g), I, tile)

"""
    face_center(g::CellGeometry, Dim(d), I[, tile]) -> SVector

Coordinates of the **plus** face of cell `I` along axis `d` (the face shared
with `I + e_d`); the minus face of `I` is the plus face of `I - e_d`.
"""
@inline face_center(g::CellGeometry, d::Dim, I, tile=nothing) = _face_center(coordinate_system(g), parent(g), d, I, tile)

"""
    face_distance(g::CellGeometry, Dim(d), I[, tile]) -> Real

Physical distance between the centres of cell `I` and its neighbour `I + e_d`
across the plus face along axis `d`: what a face gradient divides by. It is the
mean of the two cells' coordinate widths times the metric scale factor along
`d` (`1` along a radius, `r` along an angle in the plane, `r sin θ` along the
azimuth of a sphere), so an angular gradient is `Δu / (r Δθ)`, not `Δu / Δθ`.
"""
@inline face_distance(g::CellGeometry, d::Dim, I::CartesianIndex, tile=nothing) =
    _face_distance(coordinate_system(g), parent(g), d, I, tile)

"""
    face_area(g::CellGeometry, Dim(d), I[, tile]) -> Real

Area of the plus face of cell `I` along axis `d`, exact for the metric
(`∏_{j≠d} h_j` in Cartesian, with the radius evaluated on the face for radial
faces in cylindrical and spherical coordinates). Missing dimensions count as
unit extent.
"""
@inline face_area(g::CellGeometry, d::Dim, I, tile=nothing) = _face_area(coordinate_system(g), parent(g), d, I, tile)

"""
    face_normal(g::CellGeometry, Dim(d), I[, tile]) -> SVector

Outward unit normal of the plus face of cell `I` along axis `d`, in the local
coordinate basis. On a tensor-product grid this is the `d`-th basis vector,
independent of `I`; the argument form is kept so kernels need not change for
grids whose normals vary per face.
"""
@inline face_normal(g::CellGeometry, d::Dim, I, tile=nothing) = _face_normal(coordinate_system(g), parent(g), d, I, tile)

# ---- deprecated: the coordinate system as an argument (0.10) ----------------------
# The system-first forms keep working for one release: on a raw MultiHaloArray
# they wrap it with the given system, on a CellGeometry they check that the
# given system is the geometry's own and throw otherwise.
@inline _with_system(sys::CoordinateSystem, g::MultiHaloArray) = CellGeometry(sys, g)
function _with_system(sys::CoordinateSystem, g::CellGeometry)
    sys === coordinate_system(g) || throw(ArgumentError(
        "coordinate system $(nameof(typeof(sys))) does not match the geometry's $(nameof(typeof(coordinate_system(g))))"))
    return g
end
for f in (:cell_center, :cell_width, :cell_volume)
    @eval function $f(sys::CoordinateSystem, g::Union{MultiHaloArray,CellGeometry}, I, tile=nothing)
        Base.depwarn(string("`", $(QuoteNode(f)), "(system, g, …)` is deprecated: the geometry from ",
            "`cell_geometry` carries its system, call `", $(QuoteNode(f)), "(g, …)`."), $(QuoteNode(f)))
        return $f(_with_system(sys, g), I, tile)
    end
end
for f in (:face_center, :face_distance, :face_area, :face_normal)
    @eval function $f(sys::CoordinateSystem, g::Union{MultiHaloArray,CellGeometry}, d::Dim, I, tile=nothing)
        Base.depwarn(string("`", $(QuoteNode(f)), "(system, g, …)` is deprecated: the geometry from ",
            "`cell_geometry` carries its system, call `", $(QuoteNode(f)), "(g, …)`."), $(QuoteNode(f)))
        return $f(_with_system(sys, g), d, I, tile)
    end
end

# ---- direction iteration -------------------------------------------------------

"""
    map_dims(f, Val(N)) -> NTuple{N}

`(f(Dim(1)), f(Dim(2)), …, f(Dim(N)))`, unrolled at compile time. Each call
sees its direction as a static [`Dim`](@ref)`{D}`, so [`face_area`](@ref),
[`face_distance`](@ref), [`unit_vector`](@ref) and every other `Dim`-dispatched
helper resolve statically, and `f` is inlined at the call — a do-block costs
the same as a hand-written recursion over the directions (an `ntuple(d -> …)`
closure would see an `Int` and dispatch dynamically on every face). Sum or
multiply the result for a reduction over directions:

```julia
flux = sum(map_dims(Val(N)) do D
    e = unit_vector(Val(N), D)
    face_area(g, D, I, tile) * (u[I + e] - u[I]) / face_distance(g, D, I, tile)
end)
```
"""
@inline map_dims(f::F, ::Val{0}) where {F} = ()
@inline map_dims(f::F, ::Val{N}) where {F,N} = (map_dims(f, Val(N - 1))..., @inline(f(Dim(N))))
