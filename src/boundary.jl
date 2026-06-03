using StaticArrays

@inline _slice_index(::Val{N}, dim, i) where {N} =
    ntuple(j -> j == dim ? (i:i) : Colon(), Val(N))

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
    return nothing
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
function boundary_condition!(halo::HaloArray{T,N,A,Halo,B,BCondition}) where {
        T,N,A,Halo,B,BCondition}
    ntuple(Val(N)) do D
        ntuple(Val(2)) do S
            boundary_condition!(halo, Side(S), Dim(D))
        end
    end
    return nothing
end

# ============================================================
# LocalHaloArray top-level dispatcher
# ============================================================

function boundary_condition!(halo::LocalHaloArray, s::Side{side}, dim::Dim{d}) where {side,d}
    mode = halo.boundary_condition[d][side]
    boundary_condition!(halo, s, dim, mode)
    return nothing
end

function boundary_condition!(halo::LocalHaloArray{T,N}) where {T,N}
    ntuple(Val(N)) do D
        ntuple(Val(2)) do S
            boundary_condition!(halo, Side(S), Dim(D))
        end
    end
    return nothing
end

# ============================================================
# Shared mode implementations on AbstractSingleHaloArray
#
# ThreadedHaloArray dispatches all take tile_id as the second
# argument and will win over these fallbacks — no conflict.
# HaloArray and LocalHaloArray both have a 3-arg get_recv_view
# dispatch, so the body is identical for both types.
# ============================================================

# ---- Reflecting -----------------------------------------------

function boundary_condition!(halo::AbstractSingleHaloArray,
        s::Side{1}, d::Dim{dim}, ::Reflecting) where {dim}
    h = halo_width(halo)
    N = ndims(halo)
    interior_region = interior_view(halo)
    halo_region     = get_recv_view(s, d, halo)
    for i in 1:size(halo_region, dim)
        src_i = h - i + 1
        @views halo_region[_slice_index(Val(N), dim, i)...] .=
               interior_region[_slice_index(Val(N), dim, src_i)...]
    end
    return nothing
end

function boundary_condition!(halo::AbstractSingleHaloArray,
        s::Side{2}, d::Dim{dim}, ::Reflecting) where {dim}
    N = ndims(halo)
    n = size(interior_view(halo), dim)
    interior_region = interior_view(halo)
    halo_region     = get_recv_view(s, d, halo)
    for i in 1:size(halo_region, dim)
        src_i = n - (i - 1)
        @views halo_region[_slice_index(Val(N), dim, i)...] .=
               interior_region[_slice_index(Val(N), dim, src_i)...]
    end
    return nothing
end

# ---- Antireflecting -------------------------------------------

function boundary_condition!(halo::AbstractSingleHaloArray,
        s::Side{1}, d::Dim{dim}, ::Antireflecting) where {dim}
    h = halo_width(halo)
    N = ndims(halo)
    interior_region = interior_view(halo)
    halo_region     = get_recv_view(s, d, halo)
    for i in 1:size(halo_region, dim)
        src_i = h - i + 1
        @views halo_region[_slice_index(Val(N), dim, i)...] .=
               .- interior_region[_slice_index(Val(N), dim, src_i)...]
    end
    return nothing
end

function boundary_condition!(halo::AbstractSingleHaloArray,
        s::Side{2}, d::Dim{dim}, ::Antireflecting) where {dim}
    N = ndims(halo)
    n = size(interior_view(halo), dim)
    interior_region = interior_view(halo)
    halo_region     = get_recv_view(s, d, halo)
    for i in 1:size(halo_region, dim)
        src_i = n - (i - 1)
        @views halo_region[_slice_index(Val(N), dim, i)...] .=
               .- interior_region[_slice_index(Val(N), dim, src_i)...]
    end
    return nothing
end

# ---- Repeating ------------------------------------------------

function boundary_condition!(halo::AbstractSingleHaloArray,
        s::Side{1}, d::Dim{dim}, ::Repeating) where {dim}
    N = ndims(halo)
    interior_region = interior_view(halo)
    halo_region     = get_recv_view(s, d, halo)
    edge = @view interior_region[_slice_index(Val(N), dim, 1)...]
    for i in 1:size(halo_region, dim)
        @views halo_region[_slice_index(Val(N), dim, i)...] .= edge
    end
    return nothing
end

function boundary_condition!(halo::AbstractSingleHaloArray,
        s::Side{2}, d::Dim{dim}, ::Repeating) where {dim}
    N = ndims(halo)
    n = size(interior_view(halo), dim)
    interior_region = interior_view(halo)
    halo_region     = get_recv_view(s, d, halo)
    edge = @view interior_region[_slice_index(Val(N), dim, n)...]
    for i in 1:size(halo_region, dim)
        @views halo_region[_slice_index(Val(N), dim, i)...] .= edge
    end
    return nothing
end

# ---- Periodic -------------------------------------------------
# HaloArray: MPI exchange fills halos → no-op here.
# LocalHaloArray: must physically wrap the interior data.

boundary_condition!(::HaloArray, ::Side, ::Dim, ::Periodic) = nothing

function boundary_condition!(halo::LocalHaloArray, s::Side{1}, d::Dim{dim}, ::Periodic) where {dim}
    N = ndims(halo)
    h = halo_width(halo)
    interior_region = interior_view(halo)
    halo_region     = get_recv_view(s, d, halo)
    n = size(interior_region, dim)
    for i in 1:size(halo_region, dim)
        src_i = n - h + i
        @views halo_region[_slice_index(Val(N), dim, i)...] .=
               interior_region[_slice_index(Val(N), dim, src_i)...]
    end
    return nothing
end

function boundary_condition!(halo::LocalHaloArray, s::Side{2}, d::Dim{dim}, ::Periodic) where {dim}
    N = ndims(halo)
    interior_region = interior_view(halo)
    halo_region     = get_recv_view(s, d, halo)
    for i in 1:size(halo_region, dim)
        @views halo_region[_slice_index(Val(N), dim, i)...] .=
               interior_region[_slice_index(Val(N), dim, i)...]
    end
    return nothing
end

# ---- NoBoundaryCondition ----------------------------------------
# Ghost cells are left unchanged; the user fills them via a custom function.

boundary_condition!(::AbstractSingleHaloArray, ::Side, ::Dim, ::NoBoundaryCondition) = nothing
boundary_condition!(::ThreadedHaloArray, ::Integer, ::Side, ::Dim, ::NoBoundaryCondition) = nothing

# ============================================================
# Collection delegators
# ============================================================

function boundary_condition!(mha::MultiHaloArray{T,N,A}) where {T,N,A}
    foreach_field!(boundary_condition!, mha)
    return nothing
end

function boundary_condition!(mha::ArrayOfHaloArray)
    foreach_field!(boundary_condition!, mha)
    return nothing
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
@inline is_physical_boundary(state::AbstractHaloCollection, s::Side, d::Dim) =
    is_physical_boundary(first(eachfield(state)), s, d)

@inline _field_boundary_condition(field, d::Integer, side::Integer) =
    field.boundary_condition[d][side]

"""
    apply_coupled_bc!(bc::AbstractCoupledBoundaryCondition, state)
    apply_coupled_bc!(bc::AbstractCoupledBoundaryCondition, state, Side(s), Dim(d))

Apply a coupled boundary condition to a multi-field `state` (a
[`MultiHaloArray`](@ref) or [`ArrayOfHaloArray`](@ref)).

You implement the **four-argument** method for your `bc` type; it reads each
field's interior edge with [`get_send_view`](@ref) and writes each field's ghost
slab with [`get_recv_view`](@ref) (use [`eachfield`](@ref) to iterate fields):

```julia
struct MyBC <: AbstractCoupledBoundaryCondition; ... end
function HaloArrays.apply_coupled_bc!(bc::MyBC, state, s::Side{S}, d::Dim{D}) where {S,D}
    for field in eachfield(state)
        edge  = get_send_view(s, d, field)   # interior cells adjacent to the boundary
        ghost = get_recv_view(s, d, field)   # ghost cells to fill
        # ... transform across fields, write ghost ...
    end
end
```

The **two-argument** driver visits every face that is both a physical boundary
([`is_physical_boundary`](@ref)) and configured [`NoBoundaryCondition`](@ref),
and dispatches your method there — i.e. it fills exactly the edges that
`synchronize_halo!` left untouched. Call it after `synchronize_halo!(state)`.

Currently supports [`LocalHaloArray`](@ref) and MPI [`HaloArray`](@ref) fields.
"""
function apply_coupled_bc!(bc::AbstractCoupledBoundaryCondition, state::AbstractHaloCollection)
    ref = first(eachfield(state))
    for d in 1:ndims(ref), side in 1:2
        is_physical_boundary(ref, Side(side), Dim(d)) || continue
        _field_boundary_condition(ref, d, side) isa NoBoundaryCondition || continue
        applicable(apply_coupled_bc!, bc, state, Side(side), Dim(d)) ||
            throw(ArgumentError(
                "no apply_coupled_bc! for $(typeof(bc)) at side $side dim $d; define " *
                "`HaloArrays.apply_coupled_bc!(bc::$(nameof(typeof(bc))), state, ::Side, ::Dim)`"))
        apply_coupled_bc!(bc, state, Side(side), Dim(d))
    end
    return state
end

# ============================================================
# Helpers: symbol/type → BC instance, normalization
# ============================================================

function to_bc(x)
    if x isa Symbol
        x == :reflecting       && return Reflecting()
        x == :antireflecting   && return Antireflecting()
        x == :repeating        && return Repeating()
        x == :periodic         && return Periodic()
        x == :noboundary       && return NoBoundaryCondition()
        throw(ArgumentError("Unknown boundary condition symbol: $x"))
    elseif x isa DataType && x <: AbstractBoundaryCondition
        return x()
    elseif x isa AbstractBoundaryCondition
        return x
    else
        throw(ArgumentError("Invalid boundary_condition: $x"))
    end
end

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
