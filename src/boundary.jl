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
struct FaceGeometry{N,D,S,H,R,T}       # T: the field's element type (the stencil weights follow it)
    ghosts::R                   # CartesianIndices of the face's ghost cells
    wall::Int                   # storage index along D of the interior cell touching the wall
    n::Int                      # interior cells along D
    shift::CartesianIndex{N}    # global index = storage index + shift
end

function FaceGeometry(a::AbstractArray{T,N}, ::Side{S}, ::Dim{D}, ::Val{H},
        origin::CartesianIndex{N}) where {T,N,S,D,H}
    sd = size(a, D)
    ghosts = CartesianIndices(ntuple(i -> i == D ? _recv_window(Side(S), sd, H) : (H+1):(size(a, i)-H), Val(N)))
    wall = S == 1 ? H + 1 : sd - H
    return FaceGeometry{N,D,S,H,typeof(ghosts),T}(ghosts, wall, sd - 2H, origin - first(ghosts))
end

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
struct FaceCells{A,N,D}         # D: the face normal (static, so u(j) is one indexed load)
    a::A
    I::CartesianIndex{N}        # the ghost cell (storage index)
    wall::Int                   # storage index along D of the interior cell touching the wall
    into::Int                   # +1 / -1: from the wall into the interior
    n::Int                      # interior cells along D
    shift::CartesianIndex{N}    # global_index(u) = I + shift
end

@inline FaceCells(a::A, g::FaceGeometry{N,D}, I::CartesianIndex{N}) where {A,N,D} =
    FaceCells{A,N,D}(a, I, g.wall, _into(g), g.n, g.shift)

@inline _setindex(I::CartesianIndex{N}, v, ::Val{D}) where {N,D} =
    CartesianIndex(ntuple(i -> i == D ? v : I[i], Val(N)))

# the cell at storage index `i` along the normal, the ghost's coordinates elsewhere
@inline _cell(u::FaceCells{A,N,D}, i) where {A,N,D} = @inbounds u.a[_setindex(u.I, i, Val(D))]
@inline (u::FaceCells)(j) = _cell(u, u.wall + u.into * j)

"""
    face_dim(u::FaceCells) -> Int

The dimension normal to the face being filled.
"""
@inline face_dim(::FaceCells{A,N,D}) where {A,N,D} = D

"""
    far(u::FaceCells) -> FaceCells

The interior as seen from the **opposite** wall of the block: `far(u)(j)` is the
cell `j` in from that wall. `Periodic` is `far(u)(k - 1)`. On an MPI
[`HaloArray`](@ref) the opposite wall of the block is in general not the opposite
boundary of the domain (a periodic face is exchanged, not filled), so `far` is
meaningful on single-block storage only.
"""
@inline far(u::FaceCells{A,N,D}) where {A,N,D} =
    FaceCells{A,N,D}(u.a, u.I, u.wall + u.into * (u.n - 1), -u.into, u.n, u.shift)

"""
    global_index(u::FaceCells) -> CartesianIndex

The global index of the ghost cell being filled (the MPI-rank / tile offset
included), for position-dependent rules.
"""
@inline global_index(u::FaceCells) = u.I + u.shift

# ---- the built-ins as rules ------------------------------------
# These are the definitions; the fill does not call them directly but through
# `prepare_boundary`, which lowers each to a `Stencil` (same values, one shared
# evaluator). They remain callable for a user rule that composes them.
@inline (::Periodic)(u::FaceCells, k)   = far(u)(k - 1)
@inline (bc::Mirror)(u::FaceCells, k)   = bc.sign * u(k - 1)
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
@inline function (bc::Extrapolation)(u::FaceCells, k)
    p = bc.order
    v = _lagrange_weight(0, p, k) * u(0)
    for j in 1:p
        v += _lagrange_weight(j, p, k) * u(j)
    end
    return v
end

# FunctionBC: the rule as a plain function (see the docstring in haloarray.jl).
@inline (bc::FunctionBC)(u::FaceCells, k) = bc.f(u, k)

# ---- prepared rules --------------------------------------------
"""
    prepare_boundary(bc, g::FaceGeometry) -> rule

The rule the fill applies on face `g`: by default the condition itself
(`bc(u, k)`), for the built-ins a [`Stencil`](@ref) — the weighted interior
cells of each ghost layer, tabulated once per face. Built once per face per
refresh, so a condition may do per-face work here (tables depending on the
halo width and the geometry) rather than per cell.
"""
@inline prepare_boundary(bc, ::FaceGeometry) = bc

"""
    Stencil{H,C,W}

A linear rule tabulated per ghost layer: ghost layer `k` (distance `1:H` from
the wall) is `Σ_t w[k][t] · a[idx[k][t]]`, `idx` a storage index along the face
normal, `C` the capacity (entries past `count` are unused). `Periodic`,
`Repeating`, `Mirror` and `Extrapolation` lower to it in
[`prepare_boundary`](@ref), so every built-in runs through one evaluator —
`Periodic` is one entry at the opposite wall, `Extrapolation(p)` is `p + 1`
entries with the Lagrange weights.
"""
struct Stencil{H,C,W}
    count::Int
    idx::NTuple{H,NTuple{C,Int}}
    w::NTuple{H,NTuple{C,W}}
end
const _STENCIL_CAPACITY = 8

# The weights are exact integers; they are stored in the type the traversal
# multiplies with — the field's scalar type for a floating-point field (so a
# Float32 field sees Float32 weights, converted once here, not per cell and
# not as a 64-bit integer in a kernel), Int for an integer field — with an
# exactness check, so a weight the type cannot represent fails here.
@inline _weight_type(::Type{T}) where {T} = eltype(T) <: AbstractFloat ? eltype(T) : Int
@inline function _weight(::Type{W}, w::Int) where {W}
    x = W(w)
    x == w || throw(InexactError(:Stencil, W, w))
    return x
end

@inline function (s::Stencil)(u::FaceCells, k)
    @inbounds ik = s.idx[k]
    @inbounds wk = s.w[k]
    v = wk[1] * _cell(u, ik[1])
    for t in 2:s.count
        @inbounds v += wk[t] * _cell(u, ik[t])
    end
    return v
end

# One entry per layer: index `at(k)` with weight `w`, in the weight type of `T`.
@inline function _stencil1(::Type{T}, ::Val{H}, at::F, w::Int) where {T,H,F}
    W = _weight_type(T)
    return Stencil{H,_STENCIL_CAPACITY,W}(1,
        ntuple(k -> ntuple(t -> t == 1 ? at(k) : 0, Val(_STENCIL_CAPACITY)), Val(H)),
        ntuple(k -> ntuple(t -> t == 1 ? _weight(W, w) : zero(W), Val(_STENCIL_CAPACITY)), Val(H)))
end

@inline prepare_boundary(::Repeating, g::FaceGeometry{N,D,S,H,R,T}) where {N,D,S,H,R,T} =
    _stencil1(T, Val(H), k -> g.wall, 1)
@inline prepare_boundary(bc::Mirror, g::FaceGeometry{N,D,S,H,R,T}) where {N,D,S,H,R,T} =
    _stencil1(T, Val(H), k -> g.wall + _into(g) * (k - 1), Int(bc.sign))
@inline prepare_boundary(::Periodic, g::FaceGeometry{N,D,S,H,R,T}) where {N,D,S,H,R,T} =
    _stencil1(T, Val(H), k -> _far_wall(g) - _into(g) * (k - 1), 1)
@inline function prepare_boundary(bc::Extrapolation, g::FaceGeometry{N,D,S,H,R,T}) where {N,D,S,H,R,T}
    p = bc.order
    p < _STENCIL_CAPACITY || throw(ArgumentError(
        "Extrapolation order $p exceeds the stencil capacity ($(_STENCIL_CAPACITY) cells, order ≤ $(_STENCIL_CAPACITY - 1))"))
    W = _weight_type(T)
    return Stencil{H,_STENCIL_CAPACITY,W}(p + 1,
        ntuple(k -> ntuple(t -> g.wall + _into(g) * (t - 1), Val(_STENCIL_CAPACITY)), Val(H)),
        ntuple(k -> ntuple(t -> t - 1 <= p ? _weight(W, _lagrange_weight(t - 1, p, k)) : zero(W), Val(_STENCIL_CAPACITY)), Val(H)))
end

# Interior cells a rule reads along the normal at halo width H — checked once
# per face, so a short interior fails loudly instead of reading ghosts (the
# "rules read interior cells only" contract). A custom rule declares its own
# need by adding a method; the default is the wall cell.
@inline _cells_needed(::AbstractBoundaryCondition, ::Val{H}) where {H} = 1
@inline _cells_needed(::Mirror, ::Val{H}) where {H} = H                 # u(k - 1), k = 1:H
@inline _cells_needed(::Periodic, ::Val{H}) where {H} = H               # the far wall's k - 1
@inline _cells_needed(bc::Extrapolation, ::Val{H}) where {H} = bc.order + 1

# ---- the fill: every rule, every single-block backend ----------
# `tile` selects the storage: nothing (whole block) or a ThreadedHaloArray tile.
@inline _tile_storage(h::AbstractSingleHaloArray, ::Nothing) = parent(h)
@inline _tile_storage(h::ThreadedHaloArray, tile::Integer) = tile_parent(h, tile)

@inline function boundary_condition!(h::AbstractSingleHaloArray, s::Side, d::Dim,
        bc::AbstractBoundaryCondition, tile=nothing)
    a = _tile_storage(h, tile)
    # the halo width is a type parameter of `h`: passed as a Val, the geometry
    # and the stencil specialise on it without relying on constant propagation
    g = FaceGeometry(a, s, d, Val(halo_width(h)), ghost_origin(h, s, d, tile))
    isempty(g.ghosts) && return nothing          # halo width 0, or an empty block: nothing to fill
    need = _cells_needed(bc, Val(halo_width(h)))
    g.n >= need || throw(ArgumentError(
        "boundary condition needs $need interior cells along dimension $(face_dim(g)), the interior has $(g.n)"))
    _traverse!(_face_backend(a), a, prepare_boundary(bc, g), g)
    return nothing
end

# One ghost cell: its accessor and distance, then the rule.
@inline function _fill_cell!(a, rule, g::FaceGeometry, I::CartesianIndex)
    @inbounds a[I] = rule(FaceCells(a, g, I), _distance(g, I))
    return nothing
end

# Seam: the KernelAbstractions extension launches the loop below as a kernel on
# GPU storage (`_face_backend` returns the backend there); host storage loops.
@inline _face_backend(a) = nothing
function _traverse!(::Nothing, a, rule, g::FaceGeometry)
    @inbounds for I in g.ghosts
        _fill_cell!(a, rule, g, I)
    end
    return nothing
end

# ============================================================
# HaloArray top-level dispatcher
# Only applies BC on faces with no MPI neighbour (physical boundary).
# ============================================================

function boundary_condition!(halo::HaloArray{T,N,A,Halo,B,BCondition},
        s::Side{side}, dim::Dim{d}) where {T,N,A,Halo,B,BCondition,side,d}
    mode   = halo.boundary_condition[d][side]
    nbrank = halo.topology.neighbors[d][side]
    if nbrank < 0   # MPI.PROC_NULL == -1 → physical boundary
        boundary_condition!(halo, s, dim, mode)
    end
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
    _fused_periodic_fill!(halo) && return halo     # GPU storage, every face periodic: one kernel (extension)
    _foreach_face(boundary_condition!, halo, Val(N))
    return halo
end

# ============================================================
# LocalHaloArray per-face dispatcher (no MPI neighbour check needed)
# ============================================================

function boundary_condition!(halo::LocalHaloArray, s::Side{side}, dim::Dim{d}) where {side,d}
    mode = halo.boundary_condition[d][side]
    boundary_condition!(halo, s, dim, mode)
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

# Fill every face of an all-periodic single block in one launch; `false` means
# "not handled, take the per-face path".
@inline _fused_periodic_fill!(halo) = false

# ---- NoBoundaryCondition ----------------------------------------
# Ghost cells are left unchanged; the user fills them via a custom function.
boundary_condition!(::AbstractSingleHaloArray, ::Side, ::Dim, ::NoBoundaryCondition, tile=nothing) = nothing

# ============================================================
# Collection delegators
# ============================================================

function boundary_condition!(c::AbstractHaloCollection; threads::Bool=false)
    foreach_field!(f -> boundary_condition!(f; threads), c)
    return c
end

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
@inline is_physical_boundary(halo::HaloArray, ::Side{S}, ::Dim{D}) where {S,D} =
    halo.topology.neighbors[D][S] < 0
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

!!! compat "Legacy signatures (pre-0.3)"
    The old split signatures still work: a **four-argument** method
    `apply_coupled_bc!(bc, state, s, d)` (Local/MPI whole-array) and a
    **five-argument** `(bc, state, s, d, tile_id::Integer)` (per threaded tile).
    New code should define the single `tile`-generic method above instead.
"""
function apply_coupled_bc!(bc::AbstractCoupledBoundaryCondition,
        state, ::Side{S}, ::Dim{D}) where {S,D}
    throw(ArgumentError(
        "no apply_coupled_bc! for $(typeof(bc)) at side $S dim $D; define " *
        "`HaloArrays.apply_coupled_bc!(bc::$(nameof(typeof(bc))), state, ::Side, ::Dim, tile)` " *
        "(read `edge_view(field, s, d, tile)`, write `ghost_view(field, s, d, tile)`)"))
end

# Route one face to the user's method. The canonical signature is the 5-arg
# `(bc, state, side, dim, tile)`; when only the legacy 4-arg (Local/MPI) method
# exists, fall back to it. The `applicable` check (runtime method lookup) runs
# once per face per application — noise next to the ghost fill itself. There is
# deliberately NO generic throwing 5-arg fallback method: it would be dispatch-
# ambiguous with user methods that leave `tile` untyped, so the missing-method
# error lives here instead.
@inline function _coupled_face_call!(bc, state, side::Side{S}, dim::Dim{D}, tile) where {S,D}
    if applicable(apply_coupled_bc!, bc, state, side, dim, tile)
        apply_coupled_bc!(bc, state, side, dim, tile)
    elseif tile === nothing
        apply_coupled_bc!(bc, state, side, dim)   # legacy 4-arg (or its helpful error)
    else
        throw(ArgumentError(
            "no apply_coupled_bc! for $(typeof(bc)) at side $S dim $D tile $tile; define " *
            "`HaloArrays.apply_coupled_bc!(bc::$(nameof(typeof(bc))), state, ::Side, ::Dim, tile)`"))
    end
    return nothing
end

# Apply the coupled BC on one physical face. The per-field path (Local/MPI) makes
# a single whole-array call (tile = nothing); the threaded path visits only the
# boundary tiles.
@inline _apply_coupled_face!(bc, state, side::Side, dim::Dim) =
    _apply_coupled_face!(halo_backend(state), bc, state, side, dim)

@inline function _apply_coupled_face!(::AbstractHaloBackend, bc, state, side::Side, dim::Dim)
    is_physical_boundary(state, side, dim) && _coupled_face_call!(bc, state, side, dim, nothing)
    return nothing
end

@inline function _apply_coupled_face!(::ThreadedHaloBackend, bc, state,
        side::Side{S}, dim::Dim{D}) where {S,D}
    for tile_id in 1:tile_count(state)
        neighbor_tile_id(state, tile_id, D, S) == 0 &&
            _coupled_face_call!(bc, state, side, dim, tile_id)
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
