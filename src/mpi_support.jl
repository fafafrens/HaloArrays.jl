# MPI-backed types and functions for HaloArray.
# Included from HaloArrays.jl after using MPI.

# ============================================================
# HaloCommState — MPI request bookkeeping
# ============================================================

# One `UnsafeMultiRequest` per direction, slot `tag_send(D, S)` per face: no
# per-request Julia object, so posting and waiting allocate nothing. The
# blocking exchange waits on all slots at once; the split exchange waits on
# one face's slot (an item of a multi-request is itself a waitable request),
# unpacking each face as it arrives. Both share the slots, since the blocking
# exchange completes before it returns. The buffers and requests are fields
# of the HaloArray, pinned with `GC.@preserve halo` around each MPI call.
struct HaloCommState
    recv_reqs::MPI.UnsafeMultiRequest
    send_reqs::MPI.UnsafeMultiRequest
end
HaloCommState(N::Int) = HaloCommState(MPI.UnsafeMultiRequest(2N), MPI.UnsafeMultiRequest(2N))

# ============================================================
# CartesianTopology constructors
# ============================================================

function inactive_cartesian_topology(dims::NTuple{N,<:Integer}) where {N}
    dims_int = ntuple(i -> Int(dims[i]), Val(N))
    neighbors = ntuple(_ -> (MPI.PROC_NULL, MPI.PROC_NULL), Val(N))
    cart_coords = ntuple(_ -> MPI.PROC_NULL, Val(N))
    CartesianTopology{N,MPI.Comm}(0, dims_int, MPI.PROC_NULL, cart_coords,
        neighbors, MPI.COMM_NULL, MPI.COMM_NULL,
        ntuple(_ -> false, Val(N)), false)
end

inactive_cartesian_topology(n::Int) =
    inactive_cartesian_topology(ntuple(_ -> 0, n))

inactive_cartesian_topology(::Val{N}) where {N} =
    inactive_cartesian_topology(ntuple(_ -> 0, Val(N)))

function CartesianTopology(comm::MPI.Comm, dims::NTuple{N,<:Integer};
        periodic=ntuple(_ -> true, Val(N)), active::Bool=true) where {N}
    dims_int = ntuple(i -> Int(dims[i]), Val(N))
    (comm == MPI.COMM_NULL || !active) &&
        return inactive_cartesian_topology(dims_int)

    nprocs      = MPI.Comm_size(comm)
    dims_int    = MPI.Dims_create(nprocs, dims_int) |> Tuple
    cart_comm   = MPI.Cart_create(comm, dims_int; periodic=periodic, reorder=false)
    global_rank = MPI.Comm_rank(cart_comm)
    cart_coords = MPI.Cart_coords(cart_comm, global_rank) |> Tuple
    neighbors   = ntuple(Val(N)) do dim
        MPI.Cart_shift(cart_comm, dim-1, 1)
    end
    CartesianTopology{N,MPI.Comm}(nprocs, dims_int, global_rank, cart_coords,
        neighbors, comm, cart_comm, Tuple(periodic), true)
end

function CartesianTopology(comm::MPI.Comm, n_dimension::Integer;
        periodic=ntuple(_ -> true, Int(n_dimension)), active::Bool=true)
    CartesianTopology(comm, ntuple(_ -> 0, Int(n_dimension)); periodic=periodic, active=active)
end

# ============================================================
# HaloArray construction
# ============================================================

function build_haloarray_from_data(data::AbstractArray{T,N}, halo::Int,
        topology::CartesianTopology{N}, boundary_condition_raw) where {T,N}
    bc        = normalize_boundary_condition(boundary_condition_raw, N)
    recv_bufs = make_recv_buffers(data, halo)
    send_bufs = make_send_buffers(data, halo)
    validate_boundary_condition(topology, bc)
    comm_state = HaloCommState(N)
    HaloArray{T,N,typeof(data),halo,typeof(recv_bufs),typeof(bc),
              typeof(topology),typeof(comm_state)}(
        data, topology, comm_state, recv_bufs, send_bufs, bc)
end

# Three forms, like the other single kinds: wrap existing storage, allocate on
# a given topology, or allocate on a topology built over COMM_WORLD (periodic
# where the boundary condition is). `Float64` when the element type is omitted.
function HaloArray(data::AbstractArray{T,N}, halo::Integer,
        topology::CartesianTopology{N}, boundary_condition) where {T,N}
    build_haloarray_from_data(data, Int(halo), topology, boundary_condition)
end

function HaloArray(::Type{T}, owned_dims::NTuple{N,<:Integer}, halo::Integer,
        topology::CartesianTopology{N};
        boundary_condition=:repeating, storage=zeros) where {T,N}
    HaloArray(_padded_storage(storage, T, owned_dims, halo), halo, topology, boundary_condition)
end

function HaloArray(::Type{T}, owned_dims::NTuple{N,<:Integer}, halo::Integer;
        boundary_condition=:repeating, kwargs...) where {T,N}
    MPI.Initialized() || MPI.Init()
    bc       = normalize_boundary_condition(boundary_condition, N)
    topology = CartesianTopology(MPI.COMM_WORLD, ntuple(_ -> 0, Val(N)); periodic=infer_periodicity(bc))
    HaloArray(T, owned_dims, halo, topology; boundary_condition=bc, kwargs...)
end

HaloArray(owned_dims::NTuple{N,<:Integer}, halo::Integer, args...; kwargs...) where {N} =
    HaloArray(Float64, owned_dims, halo, args...; kwargs...)

function HaloArray{T,N,arraytype,Halo}(::UndefInitializer,
        boundary_condition) where {T,N,arraytype<:AbstractArray{T,N},Halo}
    data       = arraytype(undef, ntuple(_ -> 2*Halo, Val(N))...)
    topology   = inactive_cartesian_topology(Val(N))
    comm_state = HaloCommState(N)
    recv_bufs  = make_recv_buffers(data, Halo)
    send_bufs  = make_send_buffers(data, Halo)
    bc = normalize_boundary_condition(boundary_condition, N)
    HaloArray{T,N,typeof(data),Halo,typeof(recv_bufs),typeof(bc),
              typeof(topology),typeof(comm_state)}(
        data, topology, comm_state, recv_bufs, send_bufs, bc)
end

function Base.similar(halo::HaloArray{T,N,A,H,B,BC,Topo,CS},
        ::Type{AA}, dims::Dims{M}) where {T,N,A,H,B,BC,Topo,CS,AA,M}
    owned_dims = _global_to_owned_dims(halo, dims)
    fullsize   = ntuple(i -> owned_dims[i] + 2*halo_width(halo), Val(N))
    data       = similar(parent(halo), AA, fullsize)
    build_haloarray_from_data(data, halo_width(halo), halo.topology, halo.boundary_condition)
end

# Non-Int dims are normalized to Dims by Base's generic similar fallbacks.
# Base.similar dispatchers inherited from AbstractSingleHaloArray

# LinearAlgebra.norm inherited from AbstractSingleHaloArray (abstract_haloarray.jl)

# ============================================================
# Halo exchange
#
# Four per-face steps, each a `(halo, Side, Dim)` method that `_foreach_face`
# (haloarray.jl: a compile-time-unrolled, closure-free recursion) applies to
# every face, so Side/Dim are static and the views specialise. Messages are
# tagged by the sender's face, `tag_send(D, S)`, which is also the face's
# request slot; the receive tag is the neighbour's send tag. The pack pass
# runs before any post: on GPU storage the sends read device memory
# directly, and only an explicit device synchronization orders them after
# the pack kernels (`_device_synchronize`, a no-op on CPU).
# ============================================================

# Whether face (s, d) has a neighbour rank (periodic faces do: the topology
# wraps them); the complement is a physical boundary.
@inline _has_neighbor(halo::HaloArray, ::Side{S}, ::Dim{D}) where {S,D} =
    halo.topology.neighbors[D][S] != MPI.PROC_NULL

# edge slab → send buffer / recv buffer → ghost slab (`_copy_slab!`: a kernel
# on GPU storage)
@inline function _pack_face!(halo, s::Side{S}, d::Dim{D}) where {S,D}
    _has_neighbor(halo, s, d) || return nothing
    _copy_slab!(halo.send_bufs[D][S], edge_view(halo, s, d))
    return nothing
end
@inline function _unpack_face!(halo, s::Side{S}, d::Dim{D}) where {S,D}
    _has_neighbor(halo, s, d) || return nothing
    _copy_slab!(ghost_view(halo, s, d), halo.receive_bufs[D][S])
    return nothing
end

# post the face's receive and send on its slot
@inline function _post_face!(halo, s::Side{S}, d::Dim{D}) where {S,D}
    _has_neighbor(halo, s, d) || return nothing
    nbrank = halo.topology.neighbors[D][S]
    comm = halo.topology.cart_comm
    slot = tag_send(Val(D), Val(S))
    GC.@preserve halo begin
        MPI.Irecv!(halo.receive_bufs[D][S], comm, halo.comm_state.recv_reqs[slot];
            source=nbrank, tag=tag_recv(Val(D), Val(S)))
        MPI.Isend(halo.send_bufs[D][S], comm, halo.comm_state.send_reqs[slot];
            dest=nbrank, tag=tag_send(Val(D), Val(S)))
    end
    return nothing
end

# wait for the face's receive, unpack it, wait for its send (split finish)
@inline function _finish_face!(halo, s::Side{S}, d::Dim{D}) where {S,D}
    _has_neighbor(halo, s, d) || return nothing
    slot = tag_send(Val(D), Val(S))
    GC.@preserve halo begin
        MPI.Wait(halo.comm_state.recv_reqs[slot])
        _unpack_face!(halo, s, d)
        MPI.Wait(halo.comm_state.send_reqs[slot])
    end
    return nothing
end

# The blocking and the split exchange share the request slots, so a slot still
# holding a live request means a split exchange was started and not finished.
# Both drivers check this BEFORE packing: the outstanding sends may still be
# reading `send_bufs`, and a second pack would overwrite them under MPI; posting
# over the slots would leak the handles and leave an unmatched receive for a
# later exchange to pick up. An error, with the in-flight exchange untouched.
# (Read on the raw handle vectors: `MPI.isnull` per item converts through a
# preserved pointer and cost ~80 ns per exchange on a 1-rank 3-D block.)
function _check_idle(halo::HaloArray)
    cs = halo.comm_state
    null = MPI.API.MPI_REQUEST_NULL[]
    idle = all(==(null), cs.recv_reqs.vals) && all(==(null), cs.send_reqs.vals)
    idle || throw(ArgumentError("a halo exchange is already in progress on this HaloArray; " *
                                "call finish_halo_exchange! before starting another"))
    return nothing
end

# ---- public exchange API ----------------------------------------------
# All return `halo`, like every other backend's mutating driver.

function halo_exchange!(halo::HaloArray{T,N}; threads::Bool=false) where {T,N}
    _check_idle(halo)
    _foreach_face(_pack_face!, halo, Val(N))
    _device_synchronize(parent(halo))
    _foreach_face(_post_face!, halo, Val(N))
    GC.@preserve halo begin
        MPI.Waitall(halo.comm_state.recv_reqs)
        _foreach_face(_unpack_face!, halo, Val(N))
        MPI.Waitall(halo.comm_state.send_reqs)
    end
    return halo
end

function start_halo_exchange!(halo::HaloArray{T,N}) where {T,N}
    _check_idle(halo)
    _foreach_face(_pack_face!, halo, Val(N))
    _device_synchronize(parent(halo))
    _foreach_face(_post_face!, halo, Val(N))
    return halo
end

function finish_halo_exchange!(halo::HaloArray{T,N}) where {T,N}
    _foreach_face(_finish_face!, halo, Val(N))
    return halo
end

function synchronize_halo!(halo::HaloArray; threads::Bool=false)
    halo_exchange!(halo)
    boundary_condition!(halo)
    return halo
end

# ============================================================
# Reductions
# ============================================================

# Every reduction (`mapreduce` and through it `sum`/`maximum`/…, the folds,
# `any`/`all`/`==`, `dot`/`norm`/`dots!`) is written once in reduction.jl /
# vector_space.jl on the local part plus `_global`, which on a HaloArray is
# the `_allreduce` below. `mapreduce(…; dims=)` runs a transient
# `DimReductionPlan` (reduction.jl): the result has the reduced dimensions
# DROPPED and lives on the coordinate-0 slice of the topology (a
# `MaybeHaloArray`, inactive elsewhere); it owns its sub-communicator, `free!`
# it when reducing in a loop.
#
# The op is normalized AFTER the local part (add_sum's integer widening has
# happened): the builtin MPI_SUM/MPI_PROD then applies — required on
# non-Intel, where MPI.jl cannot register custom reduction ops.

# MPI.jl sends an AbstractArray value down its strided-buffer path, which a
# static array cannot serve (`strides(::SVector)` is undefined), so `sum` of
# SVector cells failed on this backend while `norm`/`dot` (scalar results)
# worked. Reduce a static value inside an isbits wrapper with a named `+`:
# MPI.jl then takes its scalar path with its statically registered `+`, which
# works on non-Intel too (a closure as the reduction op does not). Only `+` is
# defined on the wrapper; no other reduction op is defined on static arrays.
struct _MPICell{T}
    x::T
end
Base.:+(a::_MPICell, b::_MPICell) = _MPICell(a.x + b.x)

@inline _allreduce(x, op, comm; iscommutative) =
    MPI.Allreduce(x, MPI.Op(op, typeof(x); iscommutative), comm)
# A contiguous vector of partial results (e.g. the `j` coefficients of `dots!`):
# reduced in place, elementwise, with MPI's built-in `+` — no custom op, no
# allocation. `StridedVector` (not `AbstractVector`) so a static vector keeps
# taking the cell path below.
@inline _allreduce(x::StridedVector, ::typeof(+), comm; iscommutative) =
    MPI.Allreduce!(x, +, comm)
function _allreduce(x::StaticArray, op, comm; iscommutative)
    _normalize_reduction_op(op) === (+) || throw(ArgumentError(
        "only `+` reductions of static-array cells are supported over MPI (got `$op`)"))
    return MPI.Allreduce(_MPICell(x), MPI.Op(+, _MPICell{typeof(x)}; iscommutative=true), comm).x
end

# Base's reduction kwargs arrive as its internal wrappers; unwrap them so
# MPI.Op resolves to the builtin MPI_SUM/MPI_PROD instead of registering a
# custom callback op on every call.
@inline _normalize_reduction_op(op) = op
@inline _normalize_reduction_op(::typeof(Base.add_sum)) = +
@inline _normalize_reduction_op(::typeof(Base.mul_prod)) = *


# ============================================================
# DimReductionPlan — reusable dims-reduction over the topology
#
# The slice/root communicators are built once with `MPI.Cart_sub` (the
# purpose-built call for extracting sub-grids from a Cartesian communicator —
# no hand-rolled color/key splits), the reduced output array is preallocated,
# and each `reduce!` costs a single `MPI.Reduce`. `free!` releases the
# communicators deterministically. The one-shot forms (`sum(u; dims=…)`,
# `mapreduce(…; dims)`) run a transient plan per call and transfer the
# output — with its sub-communicator — to the caller; reusing a plan skips the
# per-call communicator construction entirely.
# ============================================================

# The MPI plan (see the `DimReductionPlan` docstring on the abstract type in
# reduction.jl): sub-communicators built once, one MPI.Reduce per reduce!.
struct MPIDimReductionPlan{K,N,Topo,B,O<:MaybeHaloArray} <: DimReductionPlan
    dims_to_remove::NTuple{K,Int}
    source_topology::Topo
    source_interior_size::NTuple{N,Int}
    reduce_comm::MPI.Comm       # Cart_sub over the removed dims (this rank's slice group)
    is_slice_root::Bool         # this rank's coords are 0 along every removed dim
    recv_buf::B                 # Reduce! target; interior-sized with singleton removed dims
    output::O                   # preallocated reduced array (inactive off the root slice)
    freed::Base.RefValue{Bool}
end

# Validate and canonicalize a `dims` argument (Int, tuple, or iterable) into a
# sorted duplicate-free tuple — the form the plan stores and the cache keys on.
function _normalize_reduce_dims(::Val{N}, dims) where {N}
    dims_t = _canonical_dims(dims)
    all(d -> 1 <= d <= N, dims_t) ||
        throw(ArgumentError("dims $dims out of range for a $N-dimensional array"))
    isempty(dims_t) && throw(ArgumentError("dims must select at least one dimension"))
    length(dims_t) < N ||
        throw(ArgumentError("Reducing all dimensions to a scalar is not supported; " *
                            "use the no-dims reduction (`sum(u)`, `mapreduce(f, op, u)`, …)"))
    return dims_t
end

DimReductionPlan(u::HaloArray, dims; kwargs...) = MPIDimReductionPlan(u, dims; kwargs...)

function MPIDimReductionPlan(u::HaloArray{T,N,A,Halo}, dims;
        output_eltype=T) where {T,N,A,Halo}
    topo = u.topology
    is_active(topo) ||
        throw(ArgumentError("DimReductionPlan requires an active topology on every rank"))
    dims_to_remove = _normalize_reduce_dims(Val(N), dims)
    K = length(dims_to_remove)
    dims_to_keep = Tuple(d for d in 1:N if !(d in dims_to_remove))
    M = N - K

    # One Cart_sub per role: the REDUCE comm spans the removed dims (each slice
    # group reduces internally, root = sub-rank 0 = coords 0 along removed dims);
    # the KEPT comm spans the kept dims (the topology the reduced array lives on).
    reduce_comm = MPI.Cart_sub(topo.cart_comm, Cint[d in dims_to_remove for d in 1:N])
    kept_comm   = MPI.Cart_sub(topo.cart_comm, Cint[d in dims_to_keep   for d in 1:N])

    coords = topo.cart_coords
    is_slice_root = all(d -> coords[d] == 0, dims_to_remove)

    new_dims     = ntuple(i -> topo.dims[dims_to_keep[i]], Val(M))
    new_periodic = ntuple(i -> topo.periodic_boundary_condition[dims_to_keep[i]], Val(M))
    root_topo = if is_slice_root
        # kept_comm is already Cartesian (Cart_sub keeps grid info), so build the
        # topology from it directly — no extra Cart_create.
        kept_rank   = MPI.Comm_rank(kept_comm)
        kept_coords = Tuple(MPI.Cart_coords(kept_comm, kept_rank))
        neighbors   = ntuple(i -> MPI.Cart_shift(kept_comm, i - 1, 1), Val(M))
        CartesianTopology{M,MPI.Comm}(MPI.Comm_size(kept_comm), new_dims, kept_rank,
            kept_coords, neighbors, kept_comm, kept_comm, new_periodic, true)
    else
        # Every member of a non-root kept-subgrid is non-root, so freeing here is
        # collectively consistent within that subgrid.
        MPI.free(kept_comm)
        inactive_cartesian_topology(new_dims)
    end

    owned         = interior_size(u)
    reduced_owned = ntuple(i -> owned[dims_to_keep[i]], Val(M))
    new_boundary  = ntuple(i -> u.boundary_condition[dims_to_keep[i]], Val(M))
    # Storage allocated like the source's (`similar` on the parent), so a
    # GPU-backed array gets a device-resident reduced output (and, through
    # build_haloarray_from_data, device send/recv buffers).
    out_data = fill!(similar(parent(u), output_eltype,
        ntuple(i -> reduced_owned[i] + 2Halo, Val(M))), zero(output_eltype))
    output = MaybeHaloArray(HaloArray(out_data, Halo, root_topo, new_boundary))

    # Same shape mapreduce(...; dims) produces locally (singleton removed dims);
    # allocated like u's storage so a GPU-backed array gets a device buffer.
    buf_shape = ntuple(d -> d in dims_to_remove ? 1 : owned[d], Val(N))
    recv_buf  = similar(parent(u), output_eltype, buf_shape)

    MPIDimReductionPlan(dims_to_remove, topo, owned, reduce_comm, is_slice_root,
        recv_buf, output, Ref(false))
end

"""
    reduce!(plan::DimReductionPlan, f, op, u) -> reduced array

Map `f` over the interior cells of `u` and reduce with `op` along the plan's
dimensions, into the output array prepared by [`DimReductionPlan`](@ref).
Returns the plan's preallocated output (**overwritten on every call**; `copy`
it to keep a snapshot) — the same kind of reduced array the one-shot forms
produce, except that its element type was fixed at plan construction, so
`f`/`op` must preserve `eltype(u)` (a promoting reduction like `+` on `Bool`
throws with a pointer to the one-shot forms). `u` must share the plan's
geometry (and, on MPI, its topology; there each call costs a single
`MPI.Reduce` and is collective — every rank of the topology must call it).
Same result as `mapreduce(f, op, u; dims=…)`.
"""
function reduce!(plan::MPIDimReductionPlan, f::F, op::OP, u::HaloArray{T,N}) where {F,OP,T,N}
    plan.freed[] && throw(ArgumentError("reduce! on a freed DimReductionPlan"))
    u.topology === plan.source_topology ||
        throw(ArgumentError("array topology does not match the plan's source topology"))
    interior_size(u) == plan.source_interior_size ||
        throw(DimensionMismatch("array interior size $(interior_size(u)) does not match plan's $(plan.source_interior_size)"))

    op_n = _normalize_reduction_op(op)
    local_value = mapreduce(f, op_n, interior_view(u); dims=plan.dims_to_remove)
    _check_plan_eltype(eltype(local_value), eltype(plan.recv_buf))
    op_mpi = MPI.Op(op_n, eltype(plan.recv_buf); iscommutative=true)
    MPI.Reduce!(local_value, plan.recv_buf, op_mpi, plan.reduce_comm; root=0)
    if plan.is_slice_root
        # Same linear order (only singleton dims differ), so a flat copy is exact.
        iv = interior_view(getdata(plan.output))
        iv .= reshape(plan.recv_buf, size(iv))   # broadcast: GPU-safe
    end
    return plan.output
end

"""
    free!(plan::DimReductionPlan) -> plan

Release the MPI communicators owned by `plan` (collective over the plan's
topology; idempotent, and a no-op after `MPI.Finalize`). The plan and its
output array must not be used afterwards.
"""
function free!(plan::MPIDimReductionPlan)
    plan.freed[] && return plan
    plan.freed[] = true
    MPI.Finalized() && return plan
    MPI.free(plan.reduce_comm)
    # The kept-dims communicator lives inside the output's topology (root slice
    # ranks only; elsewhere it was freed at construction).
    plan.is_slice_root && MPI.free(getdata(plan.output).topology.cart_comm)
    return plan
end

# One-shot release (the generic one-shot lives in reduction.jl): free the
# plan's own reduce communicator; the output — with ownership of the
# sub-communicator its topology lives on — is the caller's now.
function _release_transient!(plan::MPIDimReductionPlan)
    plan.freed[] = true
    MPI.free(plan.reduce_comm)   # guards COMM_NULL / Finalized itself
    return nothing
end

"""
    free!(m::MaybeHaloArray) -> m

Release the MPI sub-communicator owned by a reduced array returned by a
`dims=` keyword reduction (a no-op for
serial-backed results, which own no communicator, so backend-generic code can
call it unconditionally). Optional —
unreleased communicators are reclaimed at `MPI.Finalize` — but calling it when
you are done with the result keeps communicator use bounded when reducing in a
loop (MPI implementations cap live communicators). Collective across the ranks
where the result is active; a no-op on inactive ranks, on repeat calls, and
after `MPI.Finalize`. The array's data stays readable; only collective
operations on it (further reductions, gather, parallel HDF5 saves) become
invalid.
"""
free!(m::MaybeHaloArray) = (_free_result_comm!(getdata(m)); m)

# Serial-backed reduction results are bare arrays/collections owning no MPI
# resources — free! is a safe no-op on them, so backend-generic code calls it
# unconditionally. (Never frees a primary array's communicator: distributed
# results are always Maybe-wrapped, and bare HaloArrays are not accepted.)
free!(u::AbstractSerialHaloArray) = u
free!(c::AbstractHaloCollection)  = c

# A bare, distributed `HaloArray` owns its topology's communicator — freeing it
# would break the array. `free!` only releases the sub-communicator a reduction
# hands out (wrapped in a `MaybeHaloArray`), so reject the primary array with a
# story rather than a MethodError if generic code unwraps a result and frees it.
free!(::HaloArray) = throw(ArgumentError(
    "free! releases the sub-communicator of a reduction result (a MaybeHaloArray); " *
    "a primary HaloArray owns its topology and must not be freed. If this is a " *
    "reduction result you unwrapped, call free! on the MaybeHaloArray instead."))

_free_result_comm!(h::HaloArray) = (MPI.free(h.topology.cart_comm); nothing)
_free_result_comm!(::AbstractSerialHaloArray) = nothing   # serial fields own no comm
_free_result_comm!(c::AbstractHaloCollection) =
    (foreach(_free_result_comm!, _fields(c)); nothing)

# Compatibility name; the generic collection method above covers both
# MultiHaloArray and ArrayOfHaloArray.
