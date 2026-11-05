using StaticArrays

# ============================================================
# The cell rule: a boundary condition is callable as `bc(u, k)`
#
# One ghost cell at a time. `u` is a `FaceCells` — the interior as seen from
# this cell's wall: `u(j)` is the interior cell `j` in from the wall along the
# face normal (`u(0)` touches the wall), `far(u)` the same accessor from the
# opposite wall of the block, `global_index(u)` the ghost cell's global index.
# `k` is the ghost cell's distance from the wall, `1:halo`. The rule returns the
# ghost value; the driver writes it. Every rule reads interior cells only, so
# the fill is order-free: the CPU loops the face, the KernelAbstractions
# extension runs the same rule in a kernel, a ThreadedHaloArray tile is just a
# smaller block. The built-ins below are one line each; a user condition is a
# type with one call method, or a plain function (`FunctionBC`, or bare in
# the condition tuple).
# ============================================================

"""
    FaceGeometry

One face of a block as the fill sees it, built once per face by
[`boundary_condition!`](@ref): the ghost cells to write, the storage index of
the interior cell touching the wall, the interior extent along the normal and
the offset from storage to global indices. Type parameters: `N` dimensions,
`D` the normal, `S` the side (1 low, 2 high), `H` the halo width, `T` the
field's element type. A prepared rule ([`prepare_boundary`](@ref)) is built
from it.
"""
struct FaceGeometry{N,D,S,H,T}         # T: the field's element type (the stencil weights follow it)
    ghosts::CartesianIndices{N,NTuple{N,UnitRange{Int}}}   # the face's ghost cells
    wall::Int                   # storage index along D of the interior cell touching the wall
    n::Int                      # interior cells along D
    shift::CartesianIndex{N}    # global index = storage index + shift
end

function FaceGeometry(a::AbstractArray{T,N}, ::Side{S}, ::Dim{D}, ::Val{H},
        origin::CartesianIndex{N}) where {T,N,S,D,H}
    sd = size(a, D)
    ghosts = CartesianIndices(ntuple(i -> i == D ? _recv_window(Side(S), sd, H) : (H+1):(size(a, i)-H), Val(N)))
    wall = S == 1 ? H + 1 : sd - H
    return FaceGeometry{N,D,S,H,T}(ghosts, wall, sd - 2H, origin - first(ghosts))
end

# the same face with no ghost cells: a face the fill does not own, in a form
# of the same type (so a tuple of faces stays concrete)
@inline _empty_face(g::FaceGeometry{N,D,S,H,T}) where {N,D,S,H,T} =
    FaceGeometry{N,D,S,H,T}(CartesianIndices(ntuple(_ -> 1:0, Val(N))), g.wall, g.n, g.shift)

@inline face_dim(::FaceGeometry{N,D}) where {N,D} = D
@inline face_side(::FaceGeometry{N,D,S}) where {N,D,S} = S
@inline halo_width(::FaceGeometry{N,D,S,H}) where {N,D,S,H} = H
@inline _into(::FaceGeometry{N,D,S}) where {N,D,S} = 3 - 2S            # +1 from the low wall, -1 from the high one
@inline _far_wall(g::FaceGeometry) = g.wall + _into(g) * (g.n - 1)
@inline _distance(g::FaceGeometry{N,D,S}, I::CartesianIndex) where {N,D,S} =
    S == 1 ? g.wall - I[D] : I[D] - g.wall                             # ghost cell's distance from the wall, 1:H

"""
    FaceCells

The interior as seen from one ghost cell, handed to a boundary-condition rule
`bc(u, k)` (see [`AbstractBoundaryCondition`](@ref)):

- `u(j)` — the interior cell `j` steps in from this cell's wall along the face
  normal (`u(0)` is the cell touching the wall), with the ghost cell's
  coordinates in every other direction;
- [`far`](@ref)`(u)` — the same accessor from the opposite wall of the block;
- [`global_index`](@ref)`(u)` — the ghost cell's global `CartesianIndex`;
- `u.a`, `u.I`, [`face_dim`](@ref)`(u)` — the storage array, the ghost cell's
  storage index and the face normal, for rules that need more than the normal
  line (a tangential derivative, say).

A `FaceCells` is an isbits value built per ghost cell by the fill loop or kernel;
after inlining, `u(j)` is one indexed load.
"""
struct FaceCells{A,G<:FaceGeometry,N}
    a::A
    g::G                        # the face: wall, extent, global shift, normal and side as type parameters
    I::CartesianIndex{N}        # the ghost cell (storage index)
end

@inline _setindex(I::CartesianIndex{N}, v, ::Val{D}) where {N,D} =
    CartesianIndex(ntuple(i -> i == D ? v : I[i], Val(N)))

# the cell at storage index `i` along the normal, the ghost's coordinates elsewhere
@inline _cell(u::FaceCells, i) = @inbounds u.a[_setindex(u.I, i, Val(face_dim(u.g)))]
@inline (u::FaceCells)(j) = _cell(u, u.g.wall + _into(u.g) * j)

"""
    face_dim(u::FaceCells) -> Int

The dimension normal to the face being filled.
"""
@inline face_dim(u::FaceCells) = face_dim(u.g)

"""
    far(u::FaceCells) -> FaceCells

The interior as seen from the **opposite** wall of the block: `far(u)(j)` is the
cell `j` in from that wall. `Periodic` is `far(u)(k - 1)`. On an MPI
[`HaloArray`](@ref) the opposite wall of the block is in general not the opposite
boundary of the domain (a periodic face is exchanged, not filled), so `far` is
meaningful on single-block storage only.
"""
@inline far(u::FaceCells) = FaceCells(u.a, _far_geometry(u.g), u.I)
# the same face seen from the opposite wall: side flipped, wall moved across the interior
@inline _far_geometry(g::FaceGeometry{N,D,S,H,T}) where {N,D,S,H,T} =
    FaceGeometry{N,D,3 - S,H,T}(g.ghosts, _far_wall(g), g.n, g.shift)

"""
    global_index(u::FaceCells) -> CartesianIndex

The global index of the ghost cell being filled (the MPI-rank / tile offset
included), for position-dependent rules.
"""
@inline global_index(u::FaceCells) = u.I + u.g.shift

# ---- the built-ins as rules ------------------------------------
# The copy conditions return the cell itself, never `1 * cell`: a copy must
# preserve the element exactly (a BigFloat field keeps its precision whatever
# the current default is; a unit multiply would not).
@inline (::Periodic)(u::FaceCells, k)   = far(u)(k - 1)
@inline (bc::Mirror)(u::FaceCells, k)   = (x = u(k - 1); bc.sign < 0 ? -x : x)
@inline (::Repeating)(u::FaceCells, k)  = u(0)

# Polynomial extrapolation of order p: the degree-p Lagrange polynomial through
# the p + 1 interior cells nearest the wall (distances 0..p), evaluated at -k.
# At integer points the weights are integers, L_j(-k) = (-1)^j C(k+p, p-j) C(k+j-1, j)
# (Newton's forward-difference form), so the rule is exact for integer arrays;
# p = 0 is Repeating. Computed from binomials with overflow checking — the
# plain product of numerator and denominator overflows long before the
# coefficient itself does.
@inline function _lagrange_weight(j::Int, p::Int, k::Int)
    w = Base.checked_mul(binomial(k + p, p - j), binomial(k + j - 1, j))
    return isodd(j) ? -w : w
end
@inline function (::Extrapolation{P})(u::FaceCells, k) where {P}
    v = _lagrange_weight(0, P, k) * u(0)
    for j in 1:P
        v += _lagrange_weight(j, P, k) * u(j)
    end
    return v
end

# FunctionBC: the rule as a plain function (see the docstring in haloarray.jl).
@inline (bc::FunctionBC)(u::FaceCells, k) = bc.f(u, k)
# NoBoundaryCondition: keep the ghost as it is. Never applied (its faces are
# skipped), defined so a face tuple holding it compiles.
@inline (::NoBoundaryCondition)(u::FaceCells, k) = @inbounds u.a[u.I]

# ---- prepared rules --------------------------------------------
"""
    prepare_boundary(bc, g::FaceGeometry) -> rule

The rule the fill applies on face `g`: by default the condition itself
(`bc(u, k)`); `Extrapolation` returns a [`Stencil`](@ref) — the weighted
interior cells of each ghost layer, tabulated once per face. Called once per
face per refresh, so a condition may do per-face work here (tables depending
on the halo width and the geometry) rather than per cell.
"""
@inline prepare_boundary(bc, ::FaceGeometry) = bc

"""
    Stencil{H,C,W}

A weighted rule tabulated per ghost layer: ghost layer `k` (distance `1:H`
from the wall) is `Σ_t w[k][t] · u(t - 1)` over the `C` interior cells nearest
the wall, `W` the weight type (the field's scalar type for a floating-point
field, `Int` otherwise). `Extrapolation(p)` lowers to it in
[`prepare_boundary`](@ref): `C = p + 1` entries with the Lagrange weights.
"""
struct Stencil{H,C,W}
    w::NTuple{H,NTuple{C,W}}
    # explicit parameters only: with H = 0 the implicit constructor could not
    # infer C and W from an empty tuple (Aqua's unbound-parameter check)
    Stencil{H,C,W}(w) where {H,C,W} = new{H,C,W}(w)
end

# The weights are exact integers; they are stored in the type the traversal
# multiplies with — the field's scalar type for a floating-point field (so a
# Float32 field sees Float32 weights, converted once here, not per cell and
# not as a 64-bit integer in a kernel), Int for an integer field — with an
# exactness check, so a weight the type cannot represent fails here.
# By dispatch, not a branch on types: a branch is a runtime type on older
# Julia (1.10 infers a Union and `Stencil{H,C,W}` becomes a dynamic apply_type).
@inline _weight_type(::Type{T}) where {T<:AbstractFloat} = T
@inline _weight_type(::Type{T}) where {T} = _scalar_weight_type(eltype(T))   # a container (SVector, ...): its scalar
@inline _scalar_weight_type(::Type{T}) where {T<:AbstractFloat} = T
@inline _scalar_weight_type(::Type) = Int
@inline function _weight(::Type{W}, w::Int) where {W}
    x = W(w)
    x == w || throw(InexactError(:Stencil, W, w))
    return x
end

@inline function (s::Stencil{H,C})(u::FaceCells, k) where {H,C}
    @inbounds wk = s.w[k]
    v = wk[1] * u(0)
    for t in 2:C
        @inbounds v += wk[t] * u(t - 1)
    end
    return v
end

@inline function prepare_boundary(::Extrapolation{P}, ::FaceGeometry{N,D,S,H,T}) where {P,N,D,S,H,T}
    W = _weight_type(T)
    return Stencil{H,P + 1,W}(_extrapolation_weights(Val(H), Val(P), W))
end

# The H × (P + 1) weight table as literal tuple expressions (a generated
# function): nested `ntuple` closures are allocation-free on current Julia but
# allocated their environment on 1.10 in an earlier form of this code.
@generated function _extrapolation_weights(::Val{H}, ::Val{P}, ::Type{W}) where {H,P,W}
    layer(k) = Expr(:tuple, (:(_weight($W, _lagrange_weight($j, $P, $k))) for j in 0:P)...)
    return Expr(:tuple, (layer(k) for k in 1:H)...)
end

# Interior cells a rule reads along the normal at halo width H — checked once
# per face, so a short interior fails loudly instead of reading ghosts (the
# "rules read interior cells only" contract). A custom rule declares its own
# need by adding a method; the default is the wall cell.
@inline _cells_needed(::AbstractBoundaryCondition, ::Val{H}) where {H} = 1
@inline _cells_needed(::Mirror, ::Val{H}) where {H} = H                 # u(k - 1), k = 1:H
@inline _cells_needed(::Periodic, ::Val{H}) where {H} = H               # the far wall's k - 1
@inline _cells_needed(::Extrapolation{P}, ::Val{H}) where {P,H} = P + 1

# ---- the fill: every rule, every single-block backend ----------
# `tile` selects the storage (`_cell_storage`, field_access.jl): nothing for
# the whole block, a ThreadedHaloArray tile id otherwise.
@inline function boundary_condition!(h::AbstractSingleHaloArray, s::Side, d::Dim,
        bc::AbstractBoundaryCondition, tile=nothing)
    a = _cell_storage(h, tile)
    g = _face_geometry(h, s, d, bc, tile)
    isempty(g.ghosts) && return nothing          # halo width 0, or an empty block: nothing to fill
    _traverse!(_face_backend(a), a, prepare_boundary(bc, g), g)
    return nothing
end

# The face's geometry, validated for the interior extent the rule reads.
# The halo width is a type parameter of `h`: passed as a Val, the geometry and
# the stencil specialise on it without relying on constant propagation.
@inline function _face_geometry(h::AbstractSingleHaloArray, s::Side, d::Dim,
        bc::AbstractBoundaryCondition, tile=nothing)
    g = FaceGeometry(_cell_storage(h, tile), s, d, Val(halo_width(h)), ghost_origin(h, s, d, tile))
    isempty(g.ghosts) && return g
    need = _cells_needed(bc, Val(halo_width(h)))
    g.n >= need || throw(ArgumentError(
        "boundary condition needs $need interior cells along dimension $(face_dim(g)), the interior has $(g.n)"))
    return g
end

# Whether a single-block fill owns face (s, d): every face of a LocalHaloArray;
# on an MPI HaloArray the physical, non-periodic ones (an exchange face has a
# neighbour rank, and periodic faces are exchange faces: the topology wraps
# them). A NoBoundaryCondition face is owned and left alone by its no-op
# method; the fused launch gives it an empty geometry.
@inline _rule_owns(::LocalHaloArray, ::Side, ::Dim) = true
@inline _rule_owns(h::HaloArray, s::Side{S}, d::Dim{D}) where {S,D} =
    !_has_neighbor(h, s, d) && !(h.boundary_condition[D][S] isa Periodic)

# One ghost cell: its accessor and distance, then the rule.
@inline function _fill_cell!(a, rule, g::FaceGeometry, I::CartesianIndex)
    @inbounds a[I] = rule(FaceCells(a, g, I), _distance(g, I))
    return nothing
end

# Seam: the KernelAbstractions extension launches the loop below as a kernel on
# GPU storage (`_face_backend` returns the backend there); host storage loops.
@inline _face_backend(a) = nothing
@inline function _traverse!(::Nothing, a, rule, g::FaceGeometry)      # inlined: a prepared rule stays in the caller's frame
    @inbounds for I in g.ghosts
        _fill_cell!(a, rule, g, I)
    end
    return nothing
end

# ============================================================
# Per-face dispatcher of the single-block types: fill the face if the fill
# owns it (`_rule_owns`). The fused launch of the extension uses the same
# predicate, so face ownership is decided in one place.
# ============================================================

function boundary_condition!(halo::_SingleBlock, s::Side{S}, dim::Dim{D}) where {S,D}
    _rule_owns(halo, s, dim) && boundary_condition!(halo, s, dim, halo.boundary_condition[D][S])
    return halo
end

"""
    boundary_condition!(u)
    boundary_condition!(u, Side(s), Dim(d))

Apply the configured boundary condition to `u`'s ghost cells at the **physical**
domain edges — all sides and dimensions, or just one `(side, dim)`. This is the
edge-filling half of [`synchronize_halo!`](@ref); it does *not* perform the MPI
neighbour / inter-tile exchange ([`halo_exchange!`](@ref) does that).

It only touches faces that are real domain boundaries (for [`HaloArray`](@ref),
faces with no MPI neighbour; for [`ThreadedHaloArray`](@ref), the outer tile
edges), so it is safe to call on a decomposed grid. The condition per
`(dimension, side)` comes from `u`'s stored boundary condition — see
[`Periodic`](@ref), [`Mirror`](@ref) ([`Reflecting`](@ref), [`Antireflecting`](@ref)),
[`Repeating`](@ref), [`Extrapolation`](@ref), [`NoBoundaryCondition`](@ref),
[`FunctionBC`](@ref), or any type implementing the rule
`bc(u, k)` (see [`AbstractBoundaryCondition`](@ref)).
"""
function boundary_condition!(halo::AbstractSingleHaloArray{T,N}; threads::Bool=false) where {T,N}
    _fused_fill!(halo) && return halo              # GPU storage: every rule face in one launch (extension)
    _foreach_face(boundary_condition!, halo, Val(N))
    return halo
end

# ============================================================
# Mode-specific methods
#
# Everything not listed here is a rule and takes the generic fill above:
# `boundary_condition!(halo, side, dim, bc[, tile])`, `tile` a ThreadedHaloArray
# tile id or `nothing` for the single-block backends.
# ============================================================

# ---- Periodic -------------------------------------------------
# HaloArray (MPI exchange) and ThreadedHaloArray (inter-tile exchange) already
# wrap the edges → no-op. LocalHaloArray wraps the interior data through the
# periodic stencil (or, on GPU storage with every face periodic, the fused
# kernel of the extension).
boundary_condition!(::Union{HaloArray,ThreadedHaloArray}, ::Side, ::Dim, ::Periodic, tile=nothing) = nothing

# ============================================================
# GPU seams
#
# The exchange-buffer pack/unpack copies in mpi_support.jl are broadcasts on
# views. On GPU storage a
# broadcast on two SubArrays costs ~20 µs of launch work per face while the
# faces carry almost no data, so a refresh is launch-bound (0.17 ms for the 8
# faces of a 48⁴ block on Metal). The KernelAbstractions extension
# (ext/HaloArraysKernelAbstractionsExt.jl) specialises the seams below for
# non-CPU backends: one kernel per slab copy, one fused kernel for an
# all-periodic single block (0.06 ms for the same block), and a device
# synchronization between packing the send buffers and posting the MPI sends,
# which read device memory directly and are otherwise not ordered after the
# pack kernels. Without the extension, or on CPU storage, everything below is
# the plain broadcast it always was.
# ============================================================

"""
    _copy_slab!(dest, src)

`dest .= src` for an exchange slab (a ghost/edge view of the storage, or an
exchange buffer). The KernelAbstractions extension turns it into one kernel on
GPU storage.
"""
@inline _copy_slab!(dest, src) = _copy_slab!(_slab_backend(dest, src), dest, src)
@inline _slab_backend(dest, src) = nothing
@inline _copy_slab!(::Nothing, dest, src) = (dest .= src; nothing)

# Wait for the device queue that writes `storage` (no-op on the CPU).
@inline _device_synchronize(storage) = nothing

# Fill every rule face of a single block in one launch; `false` means "not
# handled, take the per-face path".
@inline _fused_fill!(halo) = false

# ---- NoBoundaryCondition ----------------------------------------
# Ghost cells are left unchanged; the user fills them via a custom function.
boundary_condition!(::AbstractSingleHaloArray, ::Side, ::Dim, ::NoBoundaryCondition, tile=nothing) = nothing

# ============================================================
# Collection delegators
# ============================================================

function boundary_condition!(c::AbstractHaloCollection; threads::Bool=false)
    # two constant calls rather than a closure over `threads`: the closure
    # allocates 16 bytes per call under coverage tracking on Julia 1.10
    threads ? foreach_field!(_boundary_condition_threaded!, c) : foreach_field!(boundary_condition!, c)
    return c
end
_boundary_condition_threaded!(f) = boundary_condition!(f; threads=true)

# ============================================================
# Coupled boundary conditions
#
# The per-field `boundary_condition!` above fills each field's ghosts
# independently. A *coupled* boundary condition (e.g. characteristic
# reconstruction) instead needs every field's interior edge at once, because the
# ghost state of each field depends on all of them together. Mark the relevant
# `(dim, side)` with `NoBoundaryCondition` so `synchronize_halo!` skips it, then
# call `apply_coupled_bc!` to fill those edges from the whole state.
# ============================================================

"""
    AbstractCoupledBoundaryCondition

Supertype for boundary conditions that act on a whole multi-field state at once
(rather than per field). Subtype it for your scheme, carrying any parameters,
and implement [`apply_coupled_bc!`](@ref). See [`NoBoundaryCondition`](@ref) for
opting the relevant boundaries out of the automatic per-field BC.
"""
abstract type AbstractCoupledBoundaryCondition end

"""
    eachfield(state::AbstractHaloCollection)

Iterate the underlying field arrays of a [`MultiHaloArray`](@ref) (in declared
order) or an [`ArrayOfHaloArray`](@ref) (in index order) uniformly — useful when
writing a coupled boundary condition that reads/writes every field.
"""
@inline eachfield(state::AbstractHaloCollection) = values(state.arrays)

"""
    is_physical_boundary(field, Side(s), Dim(d)) -> Bool
    is_physical_boundary(state::AbstractHaloCollection, Side(s), Dim(d)) -> Bool

Whether the `(side, dim)` face is a real domain edge (no MPI neighbour) rather
than an interior rank face. Always `true` for [`LocalHaloArray`](@ref). For a
collection, delegates to its first field. Use it to apply a physical boundary
condition only where one belongs.
"""
@inline is_physical_boundary(::LocalHaloArray, ::Side, ::Dim) = true
@inline is_physical_boundary(halo::HaloArray, s::Side, d::Dim) = !_has_neighbor(halo, s, d)
# Threaded: a domain edge exists if any boundary tile has no neighbour (id 0).
@inline is_physical_boundary(h::ThreadedHaloArray, ::Side{S}, ::Dim{D}) where {S,D} =
    any(t -> neighbor_tile_id(h, t, D, S) == 0, 1:tile_count(h))
@inline is_physical_boundary(state::AbstractHaloCollection, s::Side, d::Dim) =
    is_physical_boundary(first(eachfield(state)), s, d)


"""
    apply_coupled_bc!(bc::AbstractCoupledBoundaryCondition, state)
    apply_coupled_bc!(bc::AbstractCoupledBoundaryCondition, state, Side(s), Dim(d))

Apply a coupled boundary condition to a multi-field `state` (a
[`MultiHaloArray`](@ref) or [`ArrayOfHaloArray`](@ref)).

You implement **one five-argument method** for your `bc` type — the same method
runs on every backend. It receives the face and a `tile` handle (`nothing` for
`LocalHaloArray`/MPI [`HaloArray`](@ref) fields; the boundary tile id for
[`ThreadedHaloArray`](@ref) fields), which it passes straight through to the
view helpers: read each field's interior edge with [`edge_view`](@ref) and
write each field's ghost slab with [`ghost_view`](@ref) (use
[`eachfield`](@ref) to iterate fields):

```julia
struct MyBC <: AbstractCoupledBoundaryCondition; ... end
function HaloArrays.apply_coupled_bc!(bc::MyBC, state, s::Side{S}, d::Dim{D}, tile) where {S,D}
    for field in eachfield(state)
        edge  = edge_view(field, s, d, tile)    # interior cells adjacent to the boundary
        ghost = ghost_view(field, s, d, tile)   # ghost cells to fill
        # ... transform across fields, write ghost ...
    end
end
```

The **two-argument** driver dispatches your method on every *physical* boundary
([`is_physical_boundary`](@ref)) — under MPI it skips interior rank faces, and
for threaded fields it visits only the boundary tiles (`neighbor_tile_id == 0`).
The spatial dimension is taken from the collection's type
(`AbstractHaloCollection{T,N,S}`), so the loop unrolls and the call is
allocation-free. Mark the coupled edges [`NoBoundaryCondition`](@ref) so
`synchronize_halo!` leaves them for this call; for a *mix* of coupled and
ordinary boundaries, call the per-face form on the specific `(side, dim)`.
"""
function apply_coupled_bc! end

# One face goes straight to the user's `apply_coupled_bc!(bc, state, side, dim,
# tile)`: `tile` is `nothing` on a single block, the tile id on a threaded
# field. A condition without that method is a plain MethodError. (No generic
# throwing fallback: it would be ambiguous with user methods that leave `tile`
# untyped.)

# Apply the coupled BC on one physical face. The per-field path (Local/MPI) makes
# a single whole-array call (tile = nothing); the threaded path visits only the
# boundary tiles.
@inline _apply_coupled_face!(bc, state, side::Side, dim::Dim) =
    _apply_coupled_face!(halo_backend(state), bc, state, side, dim)

@inline function _apply_coupled_face!(::AbstractHaloBackend, bc, state, side::Side, dim::Dim)
    is_physical_boundary(state, side, dim) && apply_coupled_bc!(bc, state, side, dim, nothing)
    return nothing
end

@inline function _apply_coupled_face!(::ThreadedHaloBackend, bc, state,
        side::Side{S}, dim::Dim{D}) where {S,D}
    for tile_id in 1:tile_count(state)
        neighbor_tile_id(state, tile_id, D, S) == 0 &&
            apply_coupled_bc!(bc, state, side, dim, tile_id)
    end
    return nothing
end

# `S` is the spatial dimension carried in the collection's type, so the Val
# loops unroll and Side(s)/Dim(d) are compile-time → no allocation.
function apply_coupled_bc!(bc::AbstractCoupledBoundaryCondition,
        state::AbstractHaloCollection{T,N,S}) where {T,N,S}
    ntuple(Val(S)) do d
        ntuple(Val(2)) do s
            _apply_coupled_face!(bc, state, Side(s), Dim(d))
        end
    end
    return nothing
end

# ============================================================
# Helpers: symbol/type → BC instance, normalization
# ============================================================

# Map a user-supplied boundary condition (a symbol shortcut, a BC type, or a BC
# instance) to a concrete instance. Multiple dispatch instead of an if/isa ladder,
# so it stays open: an extension can register a new shortcut by adding a method
# `to_bc(::Val{:myname}) = MyBC()` without editing this file. Construction-time
# only, so the one dynamic dispatch through `Val(s)` is irrelevant.
to_bc(bc::AbstractBoundaryCondition)                  = bc
to_bc(::Type{T}) where {T<:AbstractBoundaryCondition} = T()
to_bc(::typeof(Reflecting))                           = Reflecting()     # the bare constructor names
to_bc(::typeof(Antireflecting))                       = Antireflecting()
to_bc(f::Function)                                    = FunctionBC(f)     # a bare rule (u, k) -> value
to_bc(s::Symbol)                                      = to_bc(Val(s))
to_bc(::Val{:reflecting})     = Reflecting()
to_bc(::Val{:antireflecting}) = Antireflecting()
to_bc(::Val{:repeating})      = Repeating()
to_bc(::Val{:extrapolating})  = Extrapolation(1)
to_bc(::Val{:periodic})       = Periodic()
to_bc(::Val{:noboundary})     = NoBoundaryCondition()
to_bc(::Val{s}) where {s}     = throw(ArgumentError("Unknown boundary condition symbol: :$s"))
to_bc(x)                      = throw(ArgumentError("Invalid boundary_condition: $x"))

function normalize_one_dim(bc_dim)
    if bc_dim isa Tuple && length(bc_dim) == 2
        return (to_bc(bc_dim[1]), to_bc(bc_dim[2]))
    else
        return (to_bc(bc_dim), to_bc(bc_dim))
    end
end

function normalize_boundary_condition(bc, N::Int)
    if bc isa Tuple
        length(bc) == N ||
            throw(ArgumentError("Boundary condition tuple length $(length(bc)) != $N"))
        return ntuple(i -> normalize_one_dim(bc[i]), N)
    else
        bc_concrete = to_bc(bc)
        return ntuple(_ -> (bc_concrete, bc_concrete), N)
    end
end

function infer_periodicity(boundary_condition::NTuple{N,NTuple{2,AbstractBoundaryCondition}}) where {N}
    ntuple(i -> boundary_condition[i][1] isa Periodic && boundary_condition[i][2] isa Periodic, Val(N))
end
