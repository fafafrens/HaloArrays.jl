# Adapt.jl integration — move a halo array between host and device, e.g.
# `cu(halo)`, `adapt(CuArray, halo)`, `adapt(MtlArray, halo)`.
#
# The critical point: the MPI send/recv buffers that live inside a `HaloArray`
# MUST end up on the same device as `data`. If they stayed on the host, every
# exchange would copy device↔host across the PCIe/NVLink boundary and hand *host*
# pointers to MPI — exactly what a GPU-aware-MPI, one-rank-per-GPU run is meant to
# avoid. Rather than adapt the existing buffers piecewise, we adapt `data` and then
# REBUILD the buffers from it: `make_*_buffers` does `similar(data-view, …)`, so a
# device parent yields device buffers automatically. This reuses the exact
# construction path, so the result is indistinguishable from a natively
# device-constructed `HaloArray`. Topology / boundary condition are device-agnostic
# metadata and pass through unchanged; the transient `comm_state` is recreated.

Adapt.adapt_structure(to, h::LocalHaloArray) =
    LocalHaloArray(Adapt.adapt(to, h.data), halo_width(h), h.boundary_condition)

Adapt.adapt_structure(to, h::HaloArray) =
    build_haloarray_from_data(Adapt.adapt(to, h.data), halo_width(h),
                              h.topology, h.boundary_condition)

# ThreadedHaloArray: adapt each tile's storage. The outer `Vector` stays a host
# vector — it just holds device-array handles, one per tile.
function Adapt.adapt_structure(to, h::ThreadedHaloArray{T,N,A,Halo,Topo,BC,TB}) where {T,N,A,Halo,Topo,BC,TB}
    newdata = map(t -> Adapt.adapt(to, t), h.data)
    return ThreadedHaloArray{T,N,eltype(newdata),Halo,Topo,BC,TB}(
        newdata, h.tile_size, h.topology, h.boundary_condition, h.backend)
end
