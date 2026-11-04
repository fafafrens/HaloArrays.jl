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
    side::Int                   # 1 low, 2 high
    wall::Int                   # storage index along d of the interior cell touching the wall
    into::Int                   # +1 / -1: from the wall into the interior
    n::Int                      # interior cells along d
    shift::CartesianIndex{N}    # global_index(u) = I + shift
end

@inline FaceCells(a::A, I::CartesianIndex{N}, ::Val{D}, side, wall, into, n, shift) where {A,N,D} =
    FaceCells{A,N,D}(a, I, side, wall, into, n, shift)

@inline _setindex(I::CartesianIndex{N}, v, ::Val{D}) where {N,D} =
    CartesianIndex(ntuple(i -> i == D ? v : I[i], Val(N)))

@inline (u::FaceCells{A,N,D})(j) where {A,N,D} =
    @inbounds u.a[_setindex(u.I, u.wall + u.into * j, Val(D))]

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
    FaceCells{A,N,D}(u.a, u.I, 3 - u.side, u.wall + u.into * (u.n - 1), -u.into, u.n, u.shift)

"""
    global_index(u::FaceCells) -> CartesianIndex

The global index of the ghost cell being filled (the MPI-rank / tile offset
included), for position-dependent rules.
"""
@inline global_index(u::FaceCells) = u.I + u.shift

# ---- the built-ins as rules ------------------------------------
@inline (::Periodic)(u::FaceCells, k)   = far(u)(k - 1)
@inline (bc::Mirror)(u::FaceCells, k)   = bc.sign * u(k - 1)
@inline (::Repeating)(u::FaceCells, k)  = u(0)

# Polynomial extrapolation of order p: the degree-p Lagrange polynomial through
# the p + 1 interior cells nearest the wall (distances 0..p), evaluated at -k.
# At integer points the weights are integers (the binomial coefficients of
# Newton's forward-difference formula), computed in integer arithmetic so the
# rule is exact for integer arrays and runs in a kernel; p = 0 is Repeating.
@inline function _lagrange_weight(j::Int, p::Int, x::Int)
    num = 1; den = 1
    for i in 0:p
        i == j && continue
        num *= x - i
        den *= j - i
    end
    return num ÷ den
end
@inline function (bc::Extrapolation)(u::FaceCells, k)
    p = bc.order
    v = _lagrange_weight(0, p, -k) * u(0)
    for j in 1:p
        v += _lagrange_weight(j, p, -k) * u(j)
    end
    return v
end

# Per-face specialisation hook: `_face_rule(bc, Val(halo))` returns the callable
# the fill applies on that face. The default is the condition itself; a
# condition whose rule has per-face setup (tables that depend on k ≤ halo only)
# returns a prepared rule instead, built once per face rather than per cell.
@inline _face_rule(bc, ::Val) = bc

# Extrapolation: the weights depend on (k, j) only, so tabulate them per face.
# The table has a fixed capacity (orders up to `_TABLE_ORDER`) so that building
# it needs no runtime-to-type conversion of the order (which is a field, not a
# type parameter) — no dynamic dispatch, no allocation; higher orders take the
# per-cell rule above.
const _TABLE_ORDER = 7
struct ExtrapolationTable{H}
    order::Int
    w::NTuple{H,NTuple{_TABLE_ORDER + 1,Int}}   # w[k][j+1] = L_j(order, -k), zero past the order
end
@inline function (t::ExtrapolationTable)(u::FaceCells, k)
    @inbounds wk = t.w[k]
    v = wk[1] * u(0)
    for j in 1:t.order
        @inbounds v += wk[j + 1] * u(j)
    end
    return v
end
@inline _extrapolation_table(p::Int, ::Val{H}) where {H} =
    ExtrapolationTable{H}(p, ntuple(k -> ntuple(j -> j - 1 <= p ? _lagrange_weight(j - 1, p, -k) : 0,
                                                Val(_TABLE_ORDER + 1)), Val(H)))

# FunctionBC: the rule as a plain function (see the docstring in haloarray.jl).
@inline (bc::FunctionBC)(u::FaceCells, k) = bc.f(u, k)

# Interior cells a rule needs along the normal (checked once per face, so a
# short interior fails loudly instead of reading ghosts).
@inline _cells_needed(::AbstractBoundaryCondition) = 1
@inline _cells_needed(bc::Extrapolation) = bc.order + 1

# ---- the fill: every rule, every single-block backend ----------
# `tile` selects the storage: nothing (whole block) or a ThreadedHaloArray tile.
@inline _tile_storage(h::AbstractSingleHaloArray, ::Nothing) = parent(h)
@inline _tile_storage(h::ThreadedHaloArray, tile::Integer) = tile_parent(h, tile)

@inline function boundary_condition!(h::AbstractSingleHaloArray, s::Side, d::Dim,
        bc::AbstractBoundaryCondition, tile=nothing)
    # the halo width is a type parameter of `h`: pass it as a Val, so the fill
    # specialises on it without relying on constant propagation
    _fill_face!(_tile_storage(h, tile), bc, s, d, Val(halo_width(h)), ghost_origin(h, s, d, tile))
    return nothing
end

function _fill_face!(a::AbstractArray{T,N}, bc, ::Side{S}, ::Dim{D}, ::Val{hw},
        origin::CartesianIndex{N}) where {T,N,S,D,hw}
    sd = size(a, D)
    n  = sd - 2hw
    n >= _cells_needed(bc) || throw(ArgumentError(
        "boundary condition needs $(_cells_needed(bc)) interior cells along dimension $D, the interior has $n"))
    ghosts = CartesianIndices(ntuple(i -> i == D ? _recv_window(Side(S), sd, hw) : (hw+1):(size(a, i)-hw), Val(N)))
    isempty(ghosts) && return nothing
    wall  = S == 1 ? hw + 1 : sd - hw
    into  = S == 1 ? 1 : -1
    shift = origin - first(ghosts)
    _launch_fill!(a, bc, ghosts, Val(D), S, wall, into, n, shift, Val(hw))
    return nothing
end

# `_face_rule` must be type-stable (one concrete rule type per condition type);
# a condition with a runtime choice of rule branches here instead, so that each
# branch is a concrete launch.
@inline _launch_fill!(a, bc, ghosts, d::Val, side, wall, into, n, shift, hw::Val) =
    _fill_face_cells!(_face_backend(a), a, _face_rule(bc, hw), ghosts, d, side, wall, into, n, shift)
@inline function _launch_fill!(a, bc::Extrapolation, ghosts, d::Val, side, wall, into, n, shift, hw::Val)
    if bc.order <= _TABLE_ORDER
        _fill_face_cells!(_face_backend(a), a, _extrapolation_table(bc.order, hw), ghosts, d, side, wall, into, n, shift)
    else
        _fill_face_cells!(_face_backend(a), a, bc, ghosts, d, side, wall, into, n, shift)
    end
    return nothing
end

# One ghost cell: its distance from the wall and its accessor, then the rule.
@inline function _fill_cell!(a, rule, I::CartesianIndex, ::Val{D}, side, wall, into, n, shift) where {D}
    k = side == 1 ? wall - I[D] : I[D] - wall
    u = FaceCells(a, I, Val(D), side, wall, into, n, shift)
    @inbounds a[I] = rule(u, k)
    return nothing
end

# Seam: the KernelAbstractions extension launches the loop below as a kernel on
# GPU storage (`_face_backend` returns the backend there); host storage loops.
@inline _face_backend(a) = nothing
function _fill_face_cells!(::Nothing, a, rule, ghosts, d::Val, side, wall, into, n, shift)
    @inbounds for I in ghosts
        _fill_cell!(a, rule, I, d, side, wall, into, n, shift)
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
# wrap the edges → no-op. LocalHaloArray must physically wrap the interior data:
# a slab copy rather than the rule, since it is one contiguous copy per face
# (and one kernel, or the fused all-faces kernel, through the extension).
boundary_condition!(::Union{HaloArray,ThreadedHaloArray}, ::Side, ::Dim, ::Periodic, tile=nothing) = nothing
@inline boundary_condition!(halo::LocalHaloArray, ::Side{S}, d::Dim, ::Periodic, tile=nothing) where {S} =
    (_copy_slab!(ghost_view(halo, Side(S), d), edge_view(halo, Side(3 - S), d)); nothing)

# ============================================================
# GPU seams
#
# The face copies — the periodic ghost fill above and the exchange-buffer
# pack/unpack in mpi_support.jl — are broadcasts on views. On GPU storage a
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

`dest .= src` for a face slab (a ghost/edge view of the storage, or an exchange
buffer). The KernelAbstractions extension turns it into one kernel on GPU storage.
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
