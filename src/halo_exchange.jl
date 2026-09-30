# MPI message tags — shared with the MPI exchange functions in mpi_support.jl
@inline tag_send(dim::Int, side::Int) = 2 * (dim - 1) + side
@inline tag_recv(dim::Int, side::Int) = tag_send(dim, 3 - side)

@inline tag_send(::Val{D}, ::Val{S}) where {D,S} = 2 * (D - 1) + S
@inline tag_recv(::Val{D}, ::Val{S}) where {D,S} = tag_send(Val{D}(), Val{3 - S}())

# HaloArray (MPI) exchange functions live in mpi_support.jl:
#   halo_exchange!(::HaloArray)
#   start_halo_exchange!(::HaloArray)  /  finish_halo_exchange!(::HaloArray)
#   synchronize_halo!(::HaloArray)

# ---- LocalHaloArray (no-ops, BC only) ---------------------------------

"""
    halo_exchange!(u)

Fill `u`'s ghost cells from neighbouring data: MPI neighbour exchange for
[`HaloArray`](@ref) and inter-tile copies for [`ThreadedHaloArray`](@ref). It
does **not** apply the physical-edge boundary condition — use
[`synchronize_halo!`](@ref) for the full refresh, or call
[`boundary_condition!`](@ref) afterwards. A no-op for [`LocalHaloArray`](@ref).
Returns `u` on every backend.

For overlapping communication with computation, use the split
[`start_halo_exchange!`](@ref) / [`finish_halo_exchange!`](@ref) pair.
"""
halo_exchange!(halo::LocalHaloArray; threads::Bool=false) = halo

"""
    start_halo_exchange!(u)
    finish_halo_exchange!(u)

Non-blocking form of [`halo_exchange!`](@ref): `start_` posts the exchange and
returns immediately so you can do ghost-free interior work, then `finish_` waits
for it to complete. Apply [`boundary_condition!`](@ref) afterwards if needed.
Both return `u`.
"""
start_halo_exchange!(halo::LocalHaloArray)  = halo
finish_halo_exchange!(halo::LocalHaloArray) = halo
@doc (@doc start_halo_exchange!) finish_halo_exchange!

"""
    synchronize_halo!(u)

Fully refresh `u`'s ghost cells: run the [`halo_exchange!`](@ref) (MPI neighbours
or thread tiles) **and** apply the [`boundary_condition!`](@ref) at the physical
domain edges. Call this before any stencil that reads ghost cells. Works on a
single array or a collection (refreshing every field). Returns `u`.

`threads=true` runs the per-tile work of a [`ThreadedHaloArray`](@ref) in
parallel through its [`thread_backend`](@ref) (also for [`halo_exchange!`](@ref)
and [`boundary_condition!`](@ref)); the default serial tile loop is
allocation-free and usually wins for the common case (halo width 1, tiles ≈
threads). Other backends accept and ignore the keyword.
"""
function synchronize_halo!(halo::LocalHaloArray; threads::Bool=false)
    boundary_condition!(halo)
    return halo
end

# ---- AbstractHaloCollection (covers MultiHaloArray + ArrayOfHaloArray) --

function halo_exchange!(halo::AbstractHaloCollection; threads::Bool=false)
    foreach_field!(f -> halo_exchange!(f; threads), halo)
    return halo
end

function start_halo_exchange!(halo::AbstractHaloCollection)
    foreach_field!(start_halo_exchange!, halo)
    return halo
end

function finish_halo_exchange!(halo::AbstractHaloCollection)
    foreach_field!(finish_halo_exchange!, halo)
    return halo
end

function synchronize_halo!(halo::AbstractHaloCollection; threads::Bool=false)
    halo_exchange!(halo; threads)
    boundary_condition!(halo; threads)
    return halo
end

function start_halo_exchange_async_unsafe!(halo::AbstractHaloCollection)
    foreach_field!(start_halo_exchange_async_unsafe!, halo)
    return nothing
end

function end_halo_exchange_async_wait_unsafe!(halo::AbstractHaloCollection)
    foreach_field!(end_halo_exchange_async_wait_unsafe!, halo)
    return nothing
end
