"""
    AbstractBoundaryCondition

Supertype for the rule that fills a physical-edge ghost layer when the halo is
refreshed (by [`boundary_condition!`](@ref) / [`synchronize_halo!`](@ref)).

Concrete conditions: [`Periodic`](@ref), [`Repeating`](@ref), [`Mirror`](@ref)
(constructed by [`Reflecting`](@ref) and [`Antireflecting`](@ref)),
[`Extrapolation`](@ref), [`NoBoundaryCondition`](@ref), [`FunctionBC`](@ref).
A condition is given per `(dimension, side)`; constructors
also accept the symbols `:periodic`, `:repeating`, `:reflecting`,
`:antireflecting`, `:extrapolating`, and a bare function `(u, k) -> value`.

# The rule contract

A condition is a callable `bc(u, k)` returning the value of **one ghost cell**:
`u` is a [`FaceCells`](@ref) — `u(j)` reads the interior cell `j` in from the
wall (`u(0)` touches it), [`far`](@ref)`(u)(j)` the same from the opposite
wall, [`global_index`](@ref)`(u)` is the ghost cell's global index — and `k` is
the ghost cell's distance from the wall, `1:halo`. The built-ins are one line
each (`Repeating` is `u(0)`, `Mirror(s)` is `s * u(k - 1)`, `Periodic` is
`far(u)(k - 1)`), and a custom condition is a type with one method:

```julia
struct Dirichlet{T} <: AbstractBoundaryCondition; value::T; end
(bc::Dirichlet)(u, k) = 2bc.value - u(k - 1)      # reflection about the wall value, every ghost layer
```

The same method runs on every backend: a loop over the face on the CPU, a
kernel on GPU storage (through the KernelAbstractions extension), per tile on a
[`ThreadedHaloArray`](@ref), on the physical faces of an MPI [`HaloArray`](@ref).
Rules read interior cells only, so the fill is order-free. Two conditions are
not rules: [`NoBoundaryCondition`](@ref) (leave the ghosts alone) and the
coupled conditions ([`apply_coupled_bc!`](@ref), several fields at once).
"""
abstract type AbstractBoundaryCondition end

"""
    Mirror(sign)

Reflection about the wall with the parity as a value: each ghost cell is `sign`
times its mirror image across the wall. `Mirror(1)` is the even reflection
(`ghost = +interior`, zero-gradient), `Mirror(-1)` the odd one (`ghost = -interior`,
the field vanishes at the wall); [`Reflecting`](@ref)`()` and
[`Antireflecting`](@ref)`()` construct these two.

The sign is a field, not a type parameter, so fields of both parities have the
same array type. A collection that mixes them — the components of a vector or
tensor at a symmetry axis, with one parity per component — then keeps a
concretely typed field container, and its halo refresh dispatches statically
instead of once per field at run time.
"""
struct Mirror <: AbstractBoundaryCondition
    sign::Int8
    function Mirror(sign::Integer)
        abs(sign) == 1 || throw(ArgumentError("Mirror sign must be +1 or -1, got $sign"))
        return new(sign)
    end
end

"""
    Reflecting()

Even/mirror reflection about the wall: `ghost = +interior` (zero-gradient).
Returns [`Mirror`](@ref)`(1)`.
"""
Reflecting() = Mirror(1)

"""
    Antireflecting()

Odd reflection about the wall: `ghost = -interior` (the field vanishes at the
wall) — e.g. a velocity component normal to a solid wall. Returns
[`Mirror`](@ref)`(-1)`. Cf. [`Reflecting`](@ref).
"""
Antireflecting() = Mirror(-1)
"Zero-gradient: each ghost cell copies the nearest interior value."
struct Repeating       <: AbstractBoundaryCondition end

"""
    Extrapolation(order = 1)

Polynomial extrapolation into the ghost cells: along the boundary normal, each
ghost cell takes the value of the polynomial of degree `order` through the
`order + 1` interior cells nearest the wall (open or outflow boundaries,
far-field edges). `Extrapolation(0)` is [`Repeating`](@ref); `Extrapolation(1)`
continues the edge slope. The interior (on a threaded array, the tile) must hold
at least `order + 1` cells along that dimension, or `synchronize_halo!` throws.
The weights are integers, so integer arrays are extrapolated exactly. The order
is a type parameter, `Extrapolation{order}`, so the stencil it lowers to is
sized at compile time for any order; `Extrapolation(order)` constructs it. The
symbol `:extrapolating` gives the linear case.
"""
struct Extrapolation{P} <: AbstractBoundaryCondition
    function Extrapolation{P}() where {P}
        P isa Integer && P >= 0 || throw(ArgumentError("Extrapolation order must be >= 0, got $P"))
        return new{P}()
    end
end
Extrapolation(order::Integer = 1) = Extrapolation{Int(order)}()
Base.show(io::IO, ::Extrapolation{P}) where {P} = print(io, "Extrapolation(", P, ")")
"Ghost cells wrap around from the opposite side of the domain (periodic)."
struct Periodic        <: AbstractBoundaryCondition end
"Leave ghost cells untouched; the caller fills them manually (e.g. a custom or characteristic BC)."
struct NoBoundaryCondition <: AbstractBoundaryCondition end

"""
    FunctionBC(f)

A boundary condition from a per-cell function `f(u, k)` — the rule contract of
[`AbstractBoundaryCondition`](@ref) without defining a type: `u` is the
[`FaceCells`](@ref) accessor (`u(j)` the interior cell `j` in from the wall,
`far(u)` the same from the opposite wall, `global_index(u)` the ghost cell's
global index) and `k` the ghost cell's distance from the wall; `f` returns the
ghost value. A bare function in a boundary-condition tuple is wrapped
automatically. It runs on every backend like a built-in — a loop on the CPU, a
kernel on GPU storage, per tile, on the physical faces of an MPI `HaloArray`.
In a kernel `f` runs with its captures as kernel arguments: scalars, or arrays
on the device — an array captured by `f` (a lookup table, an inflow profile)
moves with the halo array under `adapt(CuArray, u)` and is converted to its
device form at launch, like the storage itself.

```julia
dirichlet(g)  = FunctionBC((u, k) -> 2g - u(k - 1))             # wall value g, any halo width
neumann(q, Δ) = (u, k) -> u(0) + k * Δ * q                       # outward flux q (bare function)
inflow(prof)  = (u, k) -> prof(Tuple(global_index(u)))           # position-dependent
u = LocalHaloArray(Float64, (64, 64), 1;
        boundary_condition = ((dirichlet(300.0), neumann(0.0, dx)), (:periodic, :periodic)))
```
For cross-field (characteristic) boundaries, see [`apply_coupled_bc!`](@ref) instead.
"""
struct FunctionBC{F} <: AbstractBoundaryCondition
    f::F
end

"""
    Side{S}()  (S = 1 low, 2 high);  Side(s::Int)

Type-level tag for the low (`1`) or high (`2`) end of a dimension, used to
dispatch boundary/halo operations. See [`Dim`](@ref).
"""
struct Side{S}; end
@inline Side(s::Int) = Side{s}()

"""
    Dim{D}();  Dim(d::Int)

Type-level tag for spatial dimension `D`, used to dispatch boundary/halo
operations on a specific axis. See [`Side`](@ref).
"""
struct Dim{D}; end
@inline Dim(d::Int) = Dim{d}()

# Apply `f` to every `(Dim, Side)` face of an `N`-dimensional halo array,
# unrolled at compile time by recursing on the dimension count. State is passed
# positionally (no closure) and `f` is a named function (a singleton), so this
# stays type-stable and allocation-free on every Julia version — the single
# face-iteration primitive shared by the boundary, halo-exchange, and threaded
# paths (replacing the per-site `ntuple(Val(N)) do D … end` / hand-rolled
# recursions). Two arities: whole-array `f(x, Side, Dim)` and per-tile
# `f(x, tile, Side, Dim)` — the same (Side, Dim) order as every public helper.
@inline _foreach_face(f::F, x, ::Val{0}) where {F} = nothing
@inline function _foreach_face(f::F, x, ::Val{D}) where {F,D}
    _foreach_face(f, x, Val(D - 1))
    f(x, Side(1), Dim(D)); f(x, Side(2), Dim(D))
    return nothing
end

@inline _foreach_face(f::F, x, tile, ::Val{0}) where {F} = nothing
@inline function _foreach_face(f::F, x, tile, ::Val{D}) where {F,D}
    _foreach_face(f, x, tile, Val(D - 1))
    f(x, tile, Side(1), Dim(D)); f(x, tile, Side(2), Dim(D))
    return nothing
end

# HaloArray type params:
#   T           element type
#   N           spatial dimensions
#   A           storage array type
#   Halo        halo width (compile-time Int encoded as type param)
#   B           buffer type (recv_bufs / send_bufs)
#   BCondition  boundary condition type
#   Topo        topology type (CartesianTopology{N,C} when MPI is loaded)
#   CS          comm-state type (HaloCommState)
"""
    HaloArray(T, owned_dims, halo, topology; boundary_condition=:repeating)
    HaloArray(T, owned_dims, halo; boundary_condition=:repeating)   # builds a topology

The MPI-distributed halo array. Each rank stores an `owned_dims` interior patch of the
global grid surrounded by `halo` ghost cells; [`synchronize_halo!`](@ref)
exchanges those ghosts with neighbouring ranks over an MPI Cartesian topology
and applies the boundary condition at the physical domain edges.

`owned_dims` is the size of *this rank's interior* patch — the global grid is
`owned_dims .* topology.dims`. Reductions (`sum`, `dot`, `norm`, …) are global
(MPI `Allreduce`); scalar indexing is local-only and warns.

# Arguments
- `T`: element type (defaults to `Float64`).
- `owned_dims::NTuple{N,Int}`: this rank's interior extent.
- `halo::Int`: ghost-cell width on each side.
- `topology`: a [`CartesianTopology`](@ref). If omitted, one is built over
  `MPI.COMM_WORLD` with periodicity inferred from the boundary condition.
- `boundary_condition`: applied at physical edges (see [`LocalHaloArray`](@ref)
  for the accepted forms).

See also [`LocalHaloArray`](@ref), [`ThreadedHaloArray`](@ref),
[`halo_exchange!`](@ref); `gather_haloarray` to collect a global array on the root.
"""
mutable struct HaloArray{T,N,A,Halo,B,BCondition,Topo,CS} <: AbstractDistributedHaloArray{T,N}
    data::A
    topology::Topo
    comm_state::CS
    receive_bufs::B
    send_bufs::B
    boundary_condition::BCondition
end

@inline halo_backend(::Type{<:HaloArray}) = MPIHaloBackend()

# ---- basic accessors --------------------------------------------------

"""
    halo_width(u) -> Int

The ghost-cell width on each side (the same in every dimension). Encoded in the
type, so it is a compile-time constant. Also callable on the type.
"""
@inline halo_width(::Type{<:HaloArray{T,N,A,Halo}}) where {T,N,A,Halo} = Halo

# eltype/ndims come from AbstractArray{T,N}; parent from AbstractSingleHaloArray.
# storage_size / interior_size / interior_range come from AbstractSingleHaloArray;
# interior_view, edge_view/ghost_view, scalar indexing and show from the
# single-block union in local_haloarray.jl.

# ---- global / topology accessors (pure field access, no MPI calls) ----

#     size(u) -> dims   (the per-backend primitive behind Base.size)
#
# Size of the **whole** grid across all ranks / tiles (`interior_size .* topology.dims`).
# For [`LocalHaloArray`](@ref) it equals [`interior_size`](@ref); for [`HaloArray`](@ref)
# (MPI) and [`ThreadedHaloArray`](@ref) it is larger than this rank's/tile's share.
function _global_size(halo::HaloArray{T,N}) where {T,N}
    local_interior = interior_size(halo)
    dims = halo.topology.dims
    ntuple(i -> local_interior[i] * dims[i], Val(N))
end

@inline communicator(halo::HaloArray) = halo.topology.cart_comm
@inline is_active(a::HaloArray)    = is_active(a.topology)
@inline is_root(a::HaloArray; root::Integer=0) = is_root(a.topology; root=root)

function interior_to_global_index(halo::HaloArray{T,N}, owned_idx::NTuple{N,<:Integer}) where {T,N}
    coords     = halo.topology.cart_coords
    owned_dims = interior_size(halo)
    all(i -> 1 <= owned_idx[i] <= owned_dims[i], 1:N) ||
        throw(BoundsError(halo, owned_idx))
    ntuple(i -> coords[i]*owned_dims[i] + owned_idx[i], Val(N))
end

# Global CartesianIndex of the first ghost cell (`ghost_view(·,s,d[,tile])[1]`)
# on the (side, dim) face — the ghost analog of `interior_to_global_index`,
# for every single-block kind and for a ThreadedHaloArray tile: the block's
# first interior cell, moved outside the interior along `dim` (`1-hw` on the
# low side, `n+1` on the high side). `ghost[J]` then maps to `origin - oneunit + J`.
@inline function ghost_origin(halo::AbstractSingleHaloArray{T,N}, ::Side{S}, ::Dim{D},
        tile=nothing) where {T,N,S,D}
    t = tile === nothing ? 1 : tile
    first_cell = interior_to_global_index(halo, t, ntuple(_ -> 1, Val(N)))
    shift = S == 1 ? -halo_width(halo) : length(interior_range(halo, t)[D])
    return CartesianIndex(Base.setindex(first_cell, first_cell[D] + shift, D))
end

function global_to_storage_index(halo::HaloArray{T,N}, global_idx::NTuple{N,<:Integer}) where {T,N}
    owned_dims   = interior_size(halo)
    coords       = halo.topology.cart_coords
    h            = halo_width(halo)
    owner_coords = ntuple(i -> (global_idx[i]-1) ÷ owned_dims[i], Val(N))
    owner_coords != coords && return nothing
    interior_idx = ntuple(i -> global_idx[i] - coords[i]*owned_dims[i], Val(N))
    ntuple(i -> interior_idx[i] + h, Val(N))
end

# A distributed HaloArray reports its GLOBAL shape (`size`/`axes`/`length`) for
# the vector-space interface, but only this rank's block lives here — so a
# generic whole-array materialization is ambiguous and, done naively, fills a
# global-shaped array from local iteration (half garbage across ranks). Rather
# than lie, `iterate`/`collect` error and name the two explicit escape hatches.
# (Broadcast, reductions, `dot`/`norm`, and the BLAS-1 ops are all overridden,
# so they never reach here; `foreach` iterates `interior_view` directly.)
const _HALOARRAY_MATERIALIZE_MSG =
    "cannot iterate/collect a distributed HaloArray: it reports its global " *
    "shape but holds only this rank's block. Use `gather_haloarray(u)` for the " *
    "global array (collective, root-only) or `interior_view(u)` / " *
    "`collect(interior_view(u))` for this rank's local block."
Base.iterate(::HaloArray, args...) = error(_HALOARRAY_MATERIALIZE_MSG)
Base.collect(::HaloArray)          = error(_HALOARRAY_MATERIALIZE_MSG)

# ---- versors ----------------------------------------------------------

@inline function versors(::Val{N}) where {N}
    ntuple(i -> ntuple(j -> ifelse(i==j, 1, 0), Val(N)), Val(N))
end

# ---- send/recv buffer views -------------------------------------------
#
# Along one dimension D (storage size `sd = interior + 2*halo`), the storage is
#
#     index:   1 .. halo | halo+1 ..  sd-halo  | sd-halo+1 .. sd
#              ghost(lo)  |       interior      |    ghost(hi)
#
# Two width-`halo` slabs matter at each boundary side:
#   • the GHOST slab  — the cells to WRITE   → `_recv_window` ("receive")
#   • the EDGE slab   — the `halo` interior cells touching that ghost slab,
#                       the cells to READ    → `_send_window` ("send")
#
#   Side 1 (low):   ghost = 1:halo            edge = halo+1 : 2*halo
#   Side 2 (high):  ghost = sd-halo+1 : sd    edge = sd-2*halo+1 : sd-halo
#
# `edge_view`/`ghost_view` return *views* (writing into a ghost view
# mutates the array) of width `halo` along D, spanning the interior extent in
# every other dimension. They behave identically for HaloArray (MPI) and
# LocalHaloArray.
#
# Halo exchange uses them as the names suggest: copy this rank's `send` slab to
# a neighbour, receive into the `recv` slab. For a *physical* boundary condition
# the same views are the natural primitives: write `ghost_view(field, s, d)`
# (the ghosts) from `edge_view(field, s, d)` (the adjacent interior edge) —
# both same-shaped, so per-cell it is just `recv[k] = g(edge[k])`. The two slabs
# straddle the boundary, so their indices run TOWARD each other: on side 1 the
# innermost ghost recv[halo] is adjacent to the edge cell send[1] (a mirror BC
# pairs recv[halo+1-k] with send[k]; a zeroth-order copy fills every ghost from
# the edge cell send[1]). A coupled BC across several fields does the same, but
# gathers send-edge values from all fields, transforms them together, and writes
# the result into each field's recv slab.

@inline _send_window(::Side{1}, sd::Int, halo::Int) = (halo+1):(2*halo)
@inline _send_window(::Side{2}, sd::Int, halo::Int) = (sd-2*halo+1):(sd-halo)
@inline _recv_window(::Side{1}, sd::Int, halo::Int) = 1:halo
@inline _recv_window(::Side{2}, sd::Int, halo::Int) = (sd-halo+1):sd

# Select `window` along dimension D and the interior span (halo+1 : size-halo)
# in every other dimension.
@inline function _halo_window_view(window, arr::AbstractArray{<:Any,N}, D::Integer, halo::Int) where {N}
    view(arr, ntuple(I -> I == D ? window : (halo+1):(size(arr,I)-halo), Val(N))...)
end

# HaloArray dispatch — the dimension is always a compile-time `Dim{D}` so the
# window view specialises statically (the exchange and BC paths supply it typed).
"""
    edge_view(u, side, dim[, tile]) -> view

A view of the `halo`-wide band of interior cells adjacent to the
`(side, dim)` boundary — the cells a halo exchange reads (and sends), and the
interior edge a boundary condition reads. Paired with [`ghost_view`](@ref)
(same shape) when writing a boundary condition.

For a [`ThreadedHaloArray`](@ref) pass the tile id as the last argument;
`tile = nothing` means the whole array, so one coupled-BC method can pass its
`tile` straight through on every backend.
"""
function edge_view end

"""
    ghost_view(u, side, dim[, tile]) -> view

A mutable view of the `halo`-wide *ghost* band on the `(side, dim)` boundary —
the cells a halo exchange (or a boundary condition) writes. Same shape as
[`edge_view`](@ref) on that side, so a BC can pair them cell-by-cell.
`tile` as in [`edge_view`](@ref).
"""
function ghost_view end

# tile = nothing ⇒ whole array (single-block backends), so coupled-BC code can
# pass its tile argument through uniformly (ghost_origin takes `tile` itself).
@inline edge_view(a::AbstractHaloArray, s::Side, d::Dim, ::Nothing)  = edge_view(a, s, d)
@inline ghost_view(a::AbstractHaloArray, s::Side, d::Dim, ::Nothing) = ghost_view(a, s, d)

# Plain-array dispatch (used during buffer construction, before the HaloArray exists).
@inline edge_view(arr::AbstractArray, s::Side, ::Dim{D}, halo::Int) where {D} =
    _halo_window_view(_send_window(s, size(arr, D), halo), arr, D, halo)

@inline ghost_view(arr::AbstractArray, s::Side, ::Dim{D}, halo::Int) where {D} =
    _halo_window_view(_recv_window(s, size(arr, D), halo), arr, D, halo)

# ---- buffer allocation ------------------------------------------------

function make_recv_buffers(data::AbstractArray{T,N}, halo::Int) where {T,N}
    ntuple(D -> ntuple(S -> similar(ghost_view(data, Side(S), Dim(D), halo)), Val(2)), Val(N))
end

function make_send_buffers(data::AbstractArray{T,N}, halo::Int) where {T,N}
    ntuple(D -> ntuple(S -> similar(edge_view(data, Side(S), Dim(D), halo)), Val(2)), Val(N))
end

# validate_boundary_condition is inherited from AbstractCartesianTopology (abstract_haloarray.jl)

# ---- local-interior-dims helper (used by Base.similar in mpi_support.jl) ------

function _global_to_owned_dims(halo::HaloArray{T,N}, dims::NTuple{M,<:Integer}) where {T,N,M}
    M == N || throw(DimensionMismatch("HaloArray similar dims must have $N dimensions"))
    topo_dims = is_active(halo) ? halo.topology.dims : ntuple(_ -> 1, Val(N))
    all(d -> Int(dims[d]) % topo_dims[d] == 0, 1:N) ||
        throw(DimensionMismatch("HaloArray global similar dims $dims not divisible by topology dims $topo_dims"))
    ntuple(d -> Int(dims[d]) ÷ topo_dims[d], Val(N))
end

# ---- mutation ---------------------------------------------------------

# Base.copy, Base.zero, Base.fill!, Base.copyto! inherited from AbstractSingleHaloArray

# Base.foreach, arithmetic,
# LinearAlgebra.norm inherited from AbstractSingleHaloArray

# Base.map!/map inherited from AbstractSingleHaloArray


# ---- fill helpers -----------------------------------------------------

# fill_from_global_indices! is generic over the one-tile decomposition
# (abstract_haloarray.jl); interior_to_global_index below supplies the rank offset.


# show: the single-block method in local_haloarray.jl (with the topology line).
