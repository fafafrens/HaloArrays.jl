using StaticArrays

@inline _slice_index(::Val{N}, dim, i) where {N} =
    ntuple(j -> j == dim ? (i:i) : Colon(), Val(N))

# ============================================================
# Ghost-fill kernels (one source of truth for the index math)
#
# Each takes the ghost slab to fill and the interior to read from, both as
# already-resolved views, plus the compile-time `Side`/`Dim`. The single-array
# and threaded backends differ only in *which* views they pass (whole-array vs
# per-tile), so they all delegate to these.
#
# `N` is captured as a type parameter (from the ghost-slab view), so `Val(N)` is
# a static-parameter splat with no runtime→`Val` conversion — guaranteeing the
# `_slice_index` calls stay type-stable and allocation-free on every Julia
# version (rather than relying on `Val(ndims(x))` constant-folding).
# ============================================================

# Mirror the interior into the ghost layer. `scale = 1` → Reflecting (keep sign),
# `scale = -1` → Antireflecting (flip sign). The source index is the mirror of
# the ghost index about the wall.
@inline function _reflect_into!(halo_region::AbstractArray{<:Any,N}, interior_region,
        ::Side{S}, ::Dim{dim}, h, scale) where {N,S,dim}
    n = size(interior_region, dim)
    for i in 1:size(halo_region, dim)
        src_i = S == 1 ? h - i + 1 : n - (i - 1)
        @views halo_region[_slice_index(Val(N), dim, i)...] .=
               scale .* interior_region[_slice_index(Val(N), dim, src_i)...]
    end
    return nothing
end

# Zero-gradient: copy the nearest interior edge cell into every ghost cell.
@inline function _repeating_into!(halo_region::AbstractArray{<:Any,N}, interior_region,
        ::Side{S}, ::Dim{dim}) where {N,S,dim}
    edge_i = S == 1 ? 1 : size(interior_region, dim)
    edge = @view interior_region[_slice_index(Val(N), dim, edge_i)...]
    for i in 1:size(halo_region, dim)
        @views halo_region[_slice_index(Val(N), dim, i)...] .= edge
    end
    return nothing
end

# Wrap the opposite interior edge into the ghost layer (single-process periodic).
@inline function _periodic_into!(halo_region::AbstractArray{<:Any,N}, interior_region,
        ::Side{S}, ::Dim{dim}, h) where {N,S,dim}
    n = size(interior_region, dim)
    for i in 1:size(halo_region, dim)
        src_i = S == 1 ? n - h + i : i
        @views halo_region[_slice_index(Val(N), dim, i)...] .=
               interior_region[_slice_index(Val(N), dim, src_i)...]
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
[`Periodic`](@ref), [`Reflecting`](@ref), [`Repeating`](@ref),
[`Antireflecting`](@ref), [`NoBoundaryCondition`](@ref).
"""
function boundary_condition!(halo::AbstractSingleHaloArray{T,N}) where {T,N}
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
# Shared mode implementations on AbstractSingleHaloArray
#
# ThreadedHaloArray dispatches all take tile_id as the second
# argument and will win over these fallbacks — no conflict.
# HaloArray and LocalHaloArray both have a 3-arg ghost_view
# dispatch, so the body is identical for both types.
# ============================================================

# ---- Reflecting / Antireflecting / Repeating ------------------
# Delegate to the shared kernels with the whole-array views. `S`/`scale` are
# compile-time, so these inline to the same code as a hand-written per-side method.
@inline boundary_condition!(halo::AbstractSingleHaloArray, s::Side, d::Dim, ::Reflecting) =
    _reflect_into!(ghost_view(halo, s, d), interior_view(halo), s, d, halo_width(halo), 1)
@inline boundary_condition!(halo::AbstractSingleHaloArray, s::Side, d::Dim, ::Antireflecting) =
    _reflect_into!(ghost_view(halo, s, d), interior_view(halo), s, d, halo_width(halo), -1)
@inline boundary_condition!(halo::AbstractSingleHaloArray, s::Side, d::Dim, ::Repeating) =
    _repeating_into!(ghost_view(halo, s, d), interior_view(halo), s, d)

# ---- Periodic -------------------------------------------------
# HaloArray: MPI exchange fills halos → no-op here.
# LocalHaloArray: must physically wrap the interior data (both sides in one method).
boundary_condition!(::HaloArray, ::Side, ::Dim, ::Periodic) = nothing
@inline boundary_condition!(halo::LocalHaloArray, s::Side, d::Dim, ::Periodic) =
    _periodic_into!(ghost_view(halo, s, d), interior_view(halo), s, d, halo_width(halo))

# ---- NoBoundaryCondition ----------------------------------------
# Ghost cells are left unchanged; the user fills them via a custom function.

boundary_condition!(::AbstractSingleHaloArray, ::Side, ::Dim, ::NoBoundaryCondition) = nothing
boundary_condition!(::ThreadedHaloArray, ::Integer, ::Side, ::Dim, ::NoBoundaryCondition) = nothing

# ---- FunctionBC (custom per-field) ------------------------------
# Hand the user's closure the resolved ghost/edge views, the face, the halo width,
# and the global origin of the ghost slab (for position-dependent / GPU-safe BCs).
# Backend-uniform: the threaded method (threaded_haloarray.jl) passes tile-local
# views and the per-tile origin, so one closure runs on every backend.
@inline boundary_condition!(h::AbstractSingleHaloArray, s::Side, d::Dim, bc::FunctionBC) =
    bc.f(ghost_view(h, s, d), edge_view(h, s, d), s, d, halo_width(h), ghost_origin(h, s, d))

# ============================================================
# Collection delegators
# ============================================================

function boundary_condition!(c::AbstractHaloCollection)
    foreach_field!(boundary_condition!, c)
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
to_bc(s::Symbol)                                      = to_bc(Val(s))
to_bc(::Val{:reflecting})     = Reflecting()
to_bc(::Val{:antireflecting}) = Antireflecting()
to_bc(::Val{:repeating})      = Repeating()
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
