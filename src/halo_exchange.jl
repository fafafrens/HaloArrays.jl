# MPI message tags: a message carries the SENDER's face number, which is also
# that face's request slot (mpi_support.jl); the receiver of face (D, S)
# expects the neighbour's opposite face.
@inline tag_send(::Val{D}, ::Val{S}) where {D,S} = 2 * (D - 1) + S
@inline tag_recv(::Val{D}, ::Val{S}) where {D,S} = tag_send(Val{D}(), Val{3 - S}())

# The HaloArray (MPI) exchange lives in mpi_support.jl.

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
    # two constant calls rather than a closure over `threads`, as in the
    # collection `boundary_condition!`: the closure allocates 16 bytes per
    # call under coverage tracking on Julia 1.10 (compile-order dependent)
    threads ? foreach_field!(_halo_exchange_threaded!, halo) : foreach_field!(halo_exchange!, halo)
    return halo
end
_halo_exchange_threaded!(f) = halo_exchange!(f; threads=true)

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
