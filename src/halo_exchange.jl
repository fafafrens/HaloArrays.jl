# MPI message tags — shared with HaloArraysMPIExt
@inline tag_send(dim::Int, side::Int) = 2 * (dim - 1) + side
@inline tag_recv(dim::Int, side::Int) = tag_send(dim, 3 - side)

@inline tag_send(::Val{D}, ::Val{S}) where {D,S} = 2 * (D - 1) + S
@inline tag_recv(::Val{D}, ::Val{S}) where {D,S} = tag_send(Val{D}(), Val{3 - S}())

# HaloArray (MPI) exchange functions live in ext/HaloArraysMPIExt.jl:
#   halo_exchange!(::HaloArray)
#   start_halo_exchange!(::HaloArray)  /  finish_halo_exchange!(::HaloArray)
#   synchronize_halo!(::HaloArray)

# ---- LocalHaloArray (no-ops, BC only) ---------------------------------

halo_exchange!(halo::LocalHaloArray) = halo

start_halo_exchange!(halo::LocalHaloArray)  = halo
finish_halo_exchange!(halo::LocalHaloArray) = halo

function synchronize_halo!(halo::LocalHaloArray)
    boundary_condition!(halo)
    return halo
end

# ---- MultiHaloArray (delegates to individual fields) ------------------

function halo_exchange!(halo::MultiHaloArray)
    foreach_field!(halo_exchange!, halo)
    return halo
end

function start_halo_exchange!(halo::MultiHaloArray)
    foreach_field!(start_halo_exchange!, halo)
    return halo
end

function finish_halo_exchange!(halo::MultiHaloArray)
    foreach_field!(finish_halo_exchange!, halo)
    return halo
end

function synchronize_halo!(halo::MultiHaloArray)
    halo_exchange!(halo)
    boundary_condition!(halo)
    return halo
end

# ---- ArrayOfHaloArray (delegates to individual fields) ----------------

function halo_exchange!(halo::ArrayOfHaloArray)
    foreach_field!(halo_exchange!, halo)
    return halo
end

function start_halo_exchange!(halo::ArrayOfHaloArray)
    foreach_field!(start_halo_exchange!, halo)
    return halo
end

function finish_halo_exchange!(halo::ArrayOfHaloArray)
    foreach_field!(finish_halo_exchange!, halo)
    return halo
end

function synchronize_halo!(halo::ArrayOfHaloArray)
    halo_exchange!(halo)
    boundary_condition!(halo)
    return halo
end

# Internal async helpers that delegate (used by MultiHaloArray / ArrayOfHaloArray
# when their fields are HaloArrays; the HaloArray methods are in the MPI extension)
function start_halo_exchange_async_unsafe!(halo::MultiHaloArray)
    foreach_field!(start_halo_exchange_async_unsafe!, halo)
    return nothing
end

function end_halo_exchange_async_wait_unsafe!(halo::MultiHaloArray)
    foreach_field!(end_halo_exchange_async_wait_unsafe!, halo)
    return nothing
end

function start_halo_exchange_async_unsafe!(halo::ArrayOfHaloArray)
    foreach_field!(start_halo_exchange_async_unsafe!, halo)
    return nothing
end

function end_halo_exchange_async_wait_unsafe!(halo::ArrayOfHaloArray)
    foreach_field!(end_halo_exchange_async_wait_unsafe!, halo)
    return nothing
end
