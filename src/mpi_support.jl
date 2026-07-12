# MPI-backed types and functions for HaloArray.
# Included from HaloArrays.jl after using MPI.

# ============================================================
# HaloCommState — MPI request bookkeeping
# ============================================================

# Outer dimension is NTuple{N,...} so the compiler can specialize exchange
# loops on N (known at compile time from the HaloArray type parameter).
# Inner dimension stays Vector{MPI.Request} because elements are reassigned
# each exchange via recv_reqs[dim][side] = MPI.Irecv!(...).
# The flat vectors are separate copies used for MPI.Waitall (which is more
# efficient than N*2 individual MPI.Wait calls).
struct HaloCommState{N}
    recv_reqs::NTuple{N, Vector{MPI.Request}}
    send_reqs::NTuple{N, Vector{MPI.Request}}
    unsafe_recv_reqs_vv::NTuple{N, Vector{MPI.UnsafeRequest}}
    unsafe_send_reqs_vv::NTuple{N, Vector{MPI.UnsafeRequest}}
    recv_reqs_flat::Vector{MPI.Request}
    send_reqs_flat::Vector{MPI.Request}
    unsafe_recv_reqs::MPI.UnsafeMultiRequest
    unsafe_send_reqs::MPI.UnsafeMultiRequest
end

function HaloCommState(N::Int)
    recv_reqs         = ntuple(_ -> [MPI.Request()       for _ in 1:2], N)
    send_reqs         = ntuple(_ -> [MPI.Request()       for _ in 1:2], N)
    unsafe_recv_reqs_vv = ntuple(_ -> [MPI.UnsafeRequest() for _ in 1:2], N)
    unsafe_send_reqs_vv = ntuple(_ -> [MPI.UnsafeRequest() for _ in 1:2], N)
    recv_reqs_flat    = reduce(vcat, recv_reqs)
    send_reqs_flat    = reduce(vcat, send_reqs)
    unsafe_recv_reqs  = MPI.UnsafeMultiRequest(length(recv_reqs_flat))
    unsafe_send_reqs  = MPI.UnsafeMultiRequest(length(send_reqs_flat))
    HaloCommState{N}(recv_reqs, send_reqs, unsafe_recv_reqs_vv, unsafe_send_reqs_vv,
        recv_reqs_flat, send_reqs_flat, unsafe_recv_reqs, unsafe_send_reqs)
end

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

function HaloArray(data::AbstractArray{T,N}, halo::Int,
        topology::CartesianTopology{N}, boundary_condition) where {T,N}
    build_haloarray_from_data(data, halo, topology, boundary_condition)
end

function HaloArray(::Type{T}, owned_dims::NTuple{N,Int}, halo::Int,
        topology::CartesianTopology{N};
        boundary_condition=ntuple(_ -> (Repeating(), Repeating()), Val(N))) where {T,N}
    fullsize = ntuple(i -> owned_dims[i] + 2*halo, Val(N))
    data = zeros(T, fullsize...)
    build_haloarray_from_data(data, halo, topology, boundary_condition)
end

function HaloArray(::Type{T}, owned_dims::NTuple{N,Int}, halo::Int,
        topology::CartesianTopology{N}, boundary_condition) where {T,N}
    bc = normalize_boundary_condition(boundary_condition, N)
    HaloArray(T, owned_dims, halo, topology; boundary_condition=bc)
end

function HaloArray(::Type{T}, owned_dims::NTuple{N,Int}, halo::Int,
        boundary_condition::NTuple{N,NTuple{2,AbstractBoundaryCondition}}) where {T,N}
    MPI.Initialized() || MPI.Init()
    comm     = MPI.COMM_WORLD
    periodic = infer_periodicity(normalize_boundary_condition(boundary_condition, N))
    topology = CartesianTopology(comm, ntuple(_ -> 0, Val(N)); periodic=periodic)
    HaloArray(T, owned_dims, halo, topology; boundary_condition=boundary_condition)
end

function HaloArray(::Type{T}, owned_dims::NTuple{N,Int}, halo::Int;
        boundary_condition=ntuple(_ -> (Repeating(), Repeating()), Val(N))) where {T,N}
    HaloArray(T, owned_dims, halo, normalize_boundary_condition(boundary_condition, N))
end

function HaloArray(owned_dims::NTuple{N,Int}, halo::Int;
        boundary_condition=ntuple(_ -> (Repeating(), Repeating()), Val(N))) where {N}
    HaloArray(Float64, owned_dims, halo, normalize_boundary_condition(boundary_condition, N))
end

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
# Halo exchange — per-face helpers
#
# All helpers take ::Val{D} and ::Val{S} so D and S are compile-time
# constants.  This makes Side(S) and Dim(D) concrete types, letting
# edge_view / ghost_view specialise statically (no dynamic
# dispatch).  GC.@preserve is kept inside named functions rather than
# ntuple closures so the pinned roots are unambiguous local parameters.
# ============================================================

# Copy interior edge → send buffer.  Used by every pack path.
@inline function _copy_to_send_buf!(send_bufs, halo, ::Val{D}, ::Val{S}) where {D, S}
    halo.topology.neighbors[D][S] == MPI.PROC_NULL && return nothing
    copyto!(send_bufs[D][S], edge_view(halo, Side(S), Dim(D)))
    return nothing
end

# Copy recv buffer → ghost slab.  Used by every unpack path.
@inline function _copy_from_recv_buf!(recv_bufs, halo, ::Val{D}, ::Val{S}) where {D, S}
    halo.topology.neighbors[D][S] == MPI.PROC_NULL && return nothing
    copyto!(ghost_view(halo, Side(S), Dim(D)), recv_bufs[D][S])
    return nothing
end

# Pack + post using flat request index (waitall safe variant, no GC.@preserve needed).
@inline function _pack_post_flat_safe!(halo, recv_reqs, send_reqs, recv_bufs, send_bufs, comm,
        ::Val{D}, ::Val{S}) where {D, S}
    nbrank = halo.topology.neighbors[D][S]
    nbrank == MPI.PROC_NULL && return nothing
    idx = tag_send(Val(D), Val(S))
    _copy_to_send_buf!(send_bufs, halo, Val(D), Val(S))
    recv_reqs[idx] = MPI.Irecv!(recv_bufs[D][S], comm, recv_reqs[idx];
        source=nbrank, tag=tag_recv(Val(D), Val(S)))
    send_reqs[idx] = MPI.Isend(send_bufs[D][S], comm, send_reqs[idx];
        dest=nbrank, tag=tag_send(Val(D), Val(S)))
    return nothing
end

# Pack + post using flat request index (waitall_unsafe variant).
# recv_state = (UnsafeMultiRequest, recv_bufs),  send_state = (UnsafeMultiRequest, send_bufs)
@inline function _pack_post_flat_unsafe!(halo, recv_state, send_state, comm,
        ::Val{D}, ::Val{S}) where {D, S}
    recv_reqs, recv_bufs = recv_state
    send_reqs, send_bufs = send_state
    nbrank = halo.topology.neighbors[D][S]
    nbrank == MPI.PROC_NULL && return nothing
    idx = tag_send(Val(D), Val(S))
    copyto!(send_bufs[D][S], edge_view(halo, Side(S), Dim(D)))
    GC.@preserve recv_state MPI.Irecv!(recv_bufs[D][S], comm, recv_reqs[idx];
        source=nbrank, tag=tag_recv(Val(D), Val(S)))
    GC.@preserve send_state MPI.Isend(send_bufs[D][S], comm, send_reqs[idx];
        dest=nbrank, tag=tag_send(Val(D), Val(S)))
    return nothing
end

# Pack + post using per-face request index (async_unsafe start variant).
# recv_state = (NTuple{N,Vector{UnsafeRequest}}, recv_bufs),  same for send.
@inline function _pack_post_vv_unsafe!(halo, recv_state, send_state, comm,
        ::Val{D}, ::Val{S}) where {D, S}
    recv_reqs, recv_bufs = recv_state
    send_reqs, send_bufs = send_state
    nbrank = halo.topology.neighbors[D][S]
    nbrank == MPI.PROC_NULL && return nothing
    copyto!(send_bufs[D][S], edge_view(halo, Side(S), Dim(D)))
    GC.@preserve recv_state MPI.Irecv!(recv_bufs[D][S], comm, recv_reqs[D][S];
        source=nbrank, tag=tag_recv(Val(D), Val(S)))
    GC.@preserve send_state MPI.Isend(send_bufs[D][S], comm, send_reqs[D][S];
        dest=nbrank, tag=tag_send(Val(D), Val(S)))
    return nothing
end

# Wait + unpack using per-face unsafe requests (async_unsafe finish variant).
# recv_state = (NTuple{N,Vector{UnsafeRequest}}, recv_bufs),  send_state = NTuple{N,Vector{UnsafeRequest}}
@inline function _wait_unpack_vv_unsafe!(halo, recv_state, send_state,
        ::Val{D}, ::Val{S}) where {D, S}
    recv_reqs, recv_bufs = recv_state
    nbrank = halo.topology.neighbors[D][S]
    nbrank == MPI.PROC_NULL && return nothing
    GC.@preserve recv_state MPI.Wait(recv_reqs[D][S])
    copyto!(ghost_view(halo, Side(S), Dim(D)), recv_bufs[D][S])
    GC.@preserve send_state MPI.Wait(send_state[D][S])
    return nothing
end

# ============================================================
# Halo exchange
#
# Face iteration goes through the shared `_foreach_face` primitive (one
# compile-time-unrolled, closure-free recursion — see haloarray.jl), replacing a
# per-variant `ntuple(Val(N)) do D … end`. Each variant supplies a thin
# `(halo, Side, Dim)` adapter that pulls its request/buffer state from `halo`;
# most delegate to the per-face helpers above (whose MPI calls + `GC.@preserve`
# logic are unchanged), while the two safe-async adapters carry their — formerly
# inline — body directly (there was no extracted helper for that path).
# ============================================================

@inline _post_face_waitall!(halo, ::Side{S}, ::Dim{D}) where {D,S} =
    _pack_post_flat_safe!(halo, halo.comm_state.recv_reqs_flat, halo.comm_state.send_reqs_flat,
        halo.receive_bufs, halo.send_bufs, halo.topology.cart_comm, Val(D), Val(S))

@inline _unpack_face!(halo, ::Side{S}, ::Dim{D}) where {D,S} =
    _copy_from_recv_buf!(halo.receive_bufs, halo, Val(D), Val(S))

@inline _post_face_waitall_unsafe!(halo, ::Side{S}, ::Dim{D}) where {D,S} =
    _pack_post_flat_unsafe!(halo, (halo.comm_state.unsafe_recv_reqs, halo.receive_bufs),
        (halo.comm_state.unsafe_send_reqs, halo.send_bufs), halo.topology.cart_comm, Val(D), Val(S))

@inline _post_face_async_unsafe!(halo, ::Side{S}, ::Dim{D}) where {D,S} =
    _pack_post_vv_unsafe!(halo, (halo.comm_state.unsafe_recv_reqs_vv, halo.receive_bufs),
        (halo.comm_state.unsafe_send_reqs_vv, halo.send_bufs), halo.topology.cart_comm, Val(D), Val(S))

@inline _finish_face_async_unsafe!(halo, ::Side{S}, ::Dim{D}) where {D,S} =
    _wait_unpack_vv_unsafe!(halo, (halo.comm_state.unsafe_recv_reqs_vv, halo.receive_bufs),
        halo.comm_state.unsafe_send_reqs_vv, Val(D), Val(S))

# safe async: the two whose per-face body was previously inline in the do-block.
@inline function _post_face_async_safe!(halo, ::Side{S}, ::Dim{D}) where {D,S}
    topo   = halo.topology
    nbrank = topo.neighbors[D][S]
    nbrank == MPI.PROC_NULL && return nothing
    comm      = topo.cart_comm
    recv_reqs = halo.comm_state.recv_reqs
    send_reqs = halo.comm_state.send_reqs
    recv_bufs = halo.receive_bufs
    send_bufs = halo.send_bufs
    _copy_to_send_buf!(send_bufs, halo, Val(D), Val(S))
    recv_reqs[D][S] = MPI.Irecv!(recv_bufs[D][S], comm, recv_reqs[D][S];
        source=nbrank, tag=tag_recv(Val(D), Val(S)))
    send_reqs[D][S] = MPI.Isend(send_bufs[D][S], comm, send_reqs[D][S];
        dest=nbrank, tag=tag_send(Val(D), Val(S)))
    return nothing
end

@inline function _finish_face_async_safe!(halo, ::Side{S}, ::Dim{D}) where {D,S}
    halo.topology.neighbors[D][S] == MPI.PROC_NULL && return nothing
    recv_reqs = halo.comm_state.recv_reqs
    send_reqs = halo.comm_state.send_reqs
    MPI.Wait(recv_reqs[D][S])
    _copy_from_recv_buf!(halo.receive_bufs, halo, Val(D), Val(S))
    MPI.Wait(send_reqs[D][S])
    return nothing
end

function halo_exchange_waitall!(halo::HaloArray{T,N}) where {T,N}
    _foreach_face(_post_face_waitall!, halo, Val(N))
    MPI.Waitall(halo.comm_state.recv_reqs_flat)
    _foreach_face(_unpack_face!, halo, Val(N))
    MPI.Waitall(halo.comm_state.send_reqs_flat)
    return nothing
end

function halo_exchange_waitall_unsafe!(halo::HaloArray{T,N}) where {T,N}
    recv_state = (halo.comm_state.unsafe_recv_reqs, halo.receive_bufs)
    send_state = (halo.comm_state.unsafe_send_reqs, halo.send_bufs)
    _foreach_face(_post_face_waitall_unsafe!, halo, Val(N))
    GC.@preserve recv_state MPI.Waitall(halo.comm_state.unsafe_recv_reqs)
    _foreach_face(_unpack_face!, halo, Val(N))
    GC.@preserve send_state MPI.Waitall(halo.comm_state.unsafe_send_reqs)
    return nothing
end

function start_halo_exchange_async_unsafe!(halo::HaloArray{T,N}) where {T,N}
    _foreach_face(_post_face_async_unsafe!, halo, Val(N))
    return nothing
end

function end_halo_exchange_async_wait_unsafe!(halo::HaloArray{T,N}) where {T,N}
    _foreach_face(_finish_face_async_unsafe!, halo, Val(N))
    return nothing
end

# ---- safe (non-unsafe-request) async helpers --------------------------

function _start_halo_exchange_safe!(halo::HaloArray{T,N}) where {T,N}
    _foreach_face(_post_face_async_safe!, halo, Val(N))
    return nothing
end

function _finish_halo_exchange_safe!(halo::HaloArray{T,N}) where {T,N}
    _foreach_face(_finish_face_async_safe!, halo, Val(N))
    return nothing
end

# ---- public exchange API ----------------------------------------------
# All return `halo`, like every other backend's mutating driver.

halo_exchange!(halo::HaloArray) = (halo_exchange_waitall_unsafe!(halo); halo)

start_halo_exchange!(halo::HaloArray)  = (start_halo_exchange_async_unsafe!(halo); halo)
finish_halo_exchange!(halo::HaloArray) = (end_halo_exchange_async_wait_unsafe!(halo); halo)

# ---- compatibility wrappers (used by MPI tests and benchmarks) --------

halo_exchange_wait!(halo::HaloArray) = halo_exchange_waitall!(halo)

function start_halo_exchange_async!(halo::HaloArray)
    _start_halo_exchange_safe!(halo)
    return nothing
end

function end_halo_exchange_wait!(halo::HaloArray)
    _finish_halo_exchange_safe!(halo)
    return nothing
end

function halo_exchange_async!(halo::HaloArray)
    start_halo_exchange_async!(halo)
    end_halo_exchange_wait!(halo)
    return nothing
end

function halo_exchange_async_unsafe!(halo::HaloArray)
    start_halo_exchange_async_unsafe!(halo)
    end_halo_exchange_async_wait_unsafe!(halo)
    return nothing
end

halo_exchange_async_wait!(halo::HaloArray)        = end_halo_exchange_wait!(halo)
halo_exchange_async_wait_unsafe!(halo::HaloArray) = end_halo_exchange_async_wait_unsafe!(halo)

function synchronize_halo!(halo::HaloArray)
    halo_exchange!(halo)
    boundary_condition!(halo)
    return halo
end

# ============================================================
# Reductions
# ============================================================

# `mapreduce` (and through it `sum`/`prod`/`maximum`/`minimum`) supports the
# `dims=` keyword: it runs a transient `DimReductionPlan` (built, used, and
# released within the call) and returns a fresh reduced array every time. The
# result has the reduced dimensions DROPPED and lives on the coordinate-0 slice
# of the topology (a `MaybeHaloArray`, inactive elsewhere) — same semantics as
# `mapreduce_haloarray_dims`, unlike Base's kept-singleton-dims shape. The
# result owns its sub-communicator: `free!` it when reducing in a loop.
function Base.mapreduce(
        f::F, op::OP, halo::HaloArray, etc::Vararg{HaloArray}; kws...,
    ) where {F<:Function,OP}
    dims = _dims_kwarg(kws, 1 + length(etc))
    dims === nothing || return mapreduce_haloarray_dims(f, op, halo, dims)
    comm   = communicator(halo)
    rlocal = _local_mapreduce(mapreduce, f, op, (halo, etc...); kws...)  # shared local part
    # Normalize AFTER the local part (add_sum's integer widening already
    # happened in rlocal): the builtin MPI_SUM/MPI_PROD then applies — required
    # on non-Intel, where MPI.jl cannot register custom reduction ops.
    op_mpi = MPI.Op(_normalize_reduction_op(op), typeof(rlocal); iscommutative=true)
    MPI.Allreduce(rlocal, op_mpi, comm)
end

for func in (:mapfoldl, :mapfoldr)
    @eval function Base.$func(
            f::F, op::OP, halo::HaloArray, etc::Vararg{HaloArray}; kws...,
        ) where {F<:Function,OP}
        :dims in keys(kws) && throw(ArgumentError(
            "`$($(string(func)))` with `dims=` is not supported on a distributed HaloArray: " *
            "a per-slice reduction across ranks reorders the fold. Use `mapreduce`/`sum`/… " *
            "with `dims=` (commutative ops only)."))
        comm   = communicator(halo)
        rlocal = _local_mapreduce($func, f, op, (halo, etc...); kws...)  # shared local part
        op_mpi = MPI.Op(op, typeof(rlocal); iscommutative=false)
        MPI.Allreduce(rlocal, op_mpi, comm)
    end
end

for func in (:mapreduce, :mapfoldl, :mapfoldr)
    @eval function Base.$func(
            f::F, op::OP, z::Iterators.Zip{<:Tuple{HaloArray,Vararg{HaloArray}}}; kws...,
        ) where {F<:Function,OP}
        g(args...) = f(args)
        $func(g, op, z.is...; kws...)
    end
end

# Base's reduction kwargs arrive as its internal wrappers; unwrap them so
# MPI.Op resolves to the builtin MPI_SUM/MPI_PROD instead of registering a
# custom callback op on every call.
@inline _normalize_reduction_op(op) = op
@inline _normalize_reduction_op(::typeof(Base.add_sum)) = +
@inline _normalize_reduction_op(::typeof(Base.mul_prod)) = *

function Base.any(f::F, u::HaloArray) where {F<:Function}
    MPI.Allreduce(_local_any(f, u) :: Bool, |, communicator(u))
end

function Base.all(f::F, u::HaloArray) where {F<:Function}
    MPI.Allreduce(_local_all(f, u) :: Bool, &, communicator(u))
end

# Collective equality: every rank compares its own interior, then combines with
# a logical-AND Allreduce, so all ranks agree (see the note in reduction.jl).
function Base.:(==)(x::HaloArray, y::HaloArray)
    size(x) == size(y) || return false
    MPI.Allreduce(_local_equal(x, y) :: Bool, &, communicator(x))
end

# Global inner product / norm / sum: the SAME contiguous-aware local part every
# backend uses (`_local_dot`/`_local_sum`, reduction.jl — for this rank's single
# tile that is `_interior_dot`/`_interior_acc` over the padded parent), then a
# single Allreduce. These stay in every Krylov inner loop, so both the SIMD
# reduction and the single collective matter.
LinearAlgebra.dot(x::HaloArray, y::HaloArray) =
    MPI.Allreduce(_local_dot(x, y), +, communicator(x))
LinearAlgebra.norm(u::HaloArray) =
    sqrt(MPI.Allreduce(_local_sum(abs2, u), +, communicator(u)))
Base.sum(u::HaloArray) =
    MPI.Allreduce(_local_sum(identity, u), +, communicator(u))

# mapreduce_haloarray_dims (all backends + collections) lives in reduction.jl.


# ============================================================
# DimReductionPlan — reusable dims-reduction over the topology
#
# The slice/root communicators are built once with `MPI.Cart_sub` (the
# purpose-built call for extracting sub-grids from a Cartesian communicator —
# no hand-rolled color/key splits), the reduced output array is preallocated,
# and each `reduce!` costs a single `MPI.Reduce`. `free!` releases the
# communicators deterministically. The one-shot forms (`sum(u; dims=…)`,
# `mapreduce_haloarray_dims`) run a transient plan per call and transfer the
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
Same result as [`mapreduce_haloarray_dims`](@ref).
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
`dims=` keyword reduction or [`mapreduce_haloarray_dims`](@ref) (a no-op for
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
mapreduce_mhaloarray_dims(f, op, mha::MultiHaloArray, dims) =
    mapreduce_haloarray_dims(f, op, mha, dims)
