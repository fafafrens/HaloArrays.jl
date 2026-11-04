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
# device-constructed `HaloArray`. The topology is device-agnostic metadata and
# passes through unchanged; the transient `comm_state` is recreated.
#
# The boundary conditions move too: a rule runs inside the fill kernel on GPU
# storage, so an array it captures (a lookup table, an inflow profile) has to
# be a device array. `adapt` rebuilds a closure with adapted captures, and the
# `FunctionBC` method below passes that through; the built-in conditions hold
# no arrays and come back unchanged. A custom rule type with array fields
# defines its own `Adapt.adapt_structure` method, like any other struct.

Adapt.adapt_structure(to, bc::FunctionBC) = FunctionBC(Adapt.adapt(to, bc.f))

Adapt.adapt_structure(to, h::LocalHaloArray) =
    LocalHaloArray(Adapt.adapt(to, h.data), halo_width(h), Adapt.adapt(to, h.boundary_condition))

Adapt.adapt_structure(to, h::HaloArray) =
    build_haloarray_from_data(Adapt.adapt(to, h.data), halo_width(h),
                              h.topology, Adapt.adapt(to, h.boundary_condition))

# ThreadedHaloArray: adapt each tile's storage. The outer `Vector` stays a host
# vector — it just holds device-array handles, one per tile.
function Adapt.adapt_structure(to, h::ThreadedHaloArray{T,N,A,Halo,Topo,BC,TB}) where {T,N,A,Halo,Topo,BC,TB}
    newdata = map(t -> Adapt.adapt(to, t), h.data)
    newbc = Adapt.adapt(to, h.boundary_condition)
    return ThreadedHaloArray{T,N,eltype(newdata),Halo,Topo,typeof(newbc),TB}(
        newdata, h.tile_size, h.topology, newbc, h.backend)
end

# Collections and Maybe wrappers: without these, Adapt's generic AbstractArray
# recursion strips the wrapper and returns a bare device array — losing the halo
# metadata (BCs, topology, field names). Adapt each field/inner array through
# the methods above and rebuild the same wrapper.
Adapt.adapt_structure(to, c::FieldCollection) =
    _rebuild_collection(map(a -> Adapt.adapt(to, a), getfield(c, :arrays)))
# Preserve the stored flag rather than recomputing it — adaptation must not
# change whether a rank's value is active.
Adapt.adapt_structure(to, m::MaybeHaloArray) =
    MaybeHaloArray(Adapt.adapt(to, m.data), m.active)

# A cell geometry moves with its coordinate system: adapt the fields (through
# the collection rule above) and rewrap. Without this, Adapt's generic fallback
# returns the wrapper unchanged — still on host arrays, with no error.
Adapt.adapt_structure(to, g::CellGeometry) =
    CellGeometry(coordinate_system(g), Adapt.adapt(to, parent(g)))
