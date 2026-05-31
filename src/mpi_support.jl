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

# ---- sub-communicator helpers -----------------------------------------

function subcomm_for_slices(cart::CartesianTopology{N}, dims_to_reduce) where {N}
    coords = cart.cart_coords
    tuple_dims_to_reduce = Tuple(dims_to_reduce)
    color = coords_to_color_multi(coords, cart.dims, tuple_dims_to_reduce)
    key = 0; mul = 1
    for i in tuple_dims_to_reduce
        key += coords[i] * mul
        mul *= cart.dims[i]
    end
    sub_comm = MPI.Comm_split(cart.cart_comm, color, key)
    subrank  = (sub_comm == MPI.COMM_NULL) ? MPI.PROC_NULL : MPI.Comm_rank(sub_comm)
    return (sub_comm, coords, subrank)
end

function root_topology_multi(cart::CartesianTopology{N}, dims_to_reduce;
        root_coord::Int=0) where {N}
    coords = cart.cart_coords
    tuple_dims_to_reduce = Tuple(dims_to_reduce)
    is_root_rank = all(i -> coords[i] == root_coord, tuple_dims_to_reduce)
    color = is_root_rank ? 0 : nothing
    root_comm = MPI.Comm_split(cart.cart_comm, color, cart.global_rank)
    rem = (i for i in 1:N if !(i in tuple_dims_to_reduce))
    new_dims    = Tuple(cart.dims[i] for i in rem)
    new_periods = Tuple(cart.periodic_boundary_condition[i] for i in rem)
    if !is_root_rank || root_comm == MPI.COMM_NULL
        return CartesianTopology(root_comm, new_dims; periodic=new_periods, active=false)
    else
        return CartesianTopology(root_comm, new_dims; periodic=new_periods, active=true)
    end
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

Base.similar(halo::HaloArray{T,N,A,H,B,BC,Topo,CS}, ::Type{AA},
    dims::NTuple{M,<:Integer}) where {T,N,A,H,B,BC,Topo,CS,AA,M} =
    similar(halo, AA, ntuple(d -> Int(dims[d]), Val(M)))

Base.similar(halo::HaloArray)                       = similar(halo, eltype(halo), size(halo))
Base.similar(halo::HaloArray, ::Type{AA}) where {AA} = similar(halo, AA, size(halo))
Base.similar(halo::HaloArray, dims::Dims{M}) where {M} = similar(halo, eltype(halo), dims)
Base.similar(halo::HaloArray, dims::NTuple{M,<:Integer}) where {M} =
    similar(halo, eltype(halo), dims)

function LinearAlgebra.norm(halo::HaloArray, p::Real=2)
    if p == 2
        return sqrt(mapreduce(abs2, +, halo))
    elseif p == Inf
        return mapreduce(abs, max, halo)
    else
        return mapreduce(x -> abs(x)^p, +, halo)^(1/p)
    end
end

# ============================================================
# Halo exchange
# ============================================================

function halo_exchange_waitall!(halo::HaloArray{T,N}) where {T,N}
    comm      = halo.topology.cart_comm
    topo      = halo.topology
    recv_reqs = halo.comm_state.recv_reqs_flat
    send_reqs = halo.comm_state.send_reqs_flat
    recv_bufs = halo.receive_bufs
    send_bufs = halo.send_bufs
    @inbounds for dim in 1:N, side in 1:2
        nbrank = topo.neighbors[dim][side]
        if nbrank != MPI.PROC_NULL
            idx = tag_send(dim, side)
            copyto!(send_bufs[dim][side], get_send_view(Side(side), dim, halo))
            recv_reqs[idx] = MPI.Irecv!(recv_bufs[dim][side], comm, recv_reqs[idx];
                source=nbrank, tag=tag_recv(dim, side))
            send_reqs[idx] = MPI.Isend(send_bufs[dim][side], comm, send_reqs[idx];
                dest=nbrank, tag=tag_send(dim, side))
        end
    end
    MPI.Waitall(recv_reqs)
    @inbounds for dim in 1:N, side in 1:2
        nbrank = topo.neighbors[dim][side]
        if nbrank != MPI.PROC_NULL
            copyto!(get_recv_view(Side(side), dim, halo), recv_bufs[dim][side])
        end
    end
    MPI.Waitall(send_reqs)
    return nothing
end

function halo_exchange_waitall_unsafe!(halo::HaloArray{T,N}) where {T,N}
    comm      = halo.topology.cart_comm
    topo      = halo.topology
    recv_reqs = halo.comm_state.unsafe_recv_reqs
    send_reqs = halo.comm_state.unsafe_send_reqs
    recv_bufs = halo.receive_bufs
    send_bufs = halo.send_bufs
    recv_state = (recv_reqs, recv_bufs)
    send_state = (send_reqs, send_bufs)
    @inbounds for dim in 1:N, side in 1:2
        nbrank = topo.neighbors[dim][side]
        if nbrank != MPI.PROC_NULL
            idx = tag_send(dim, side)
            copyto!(send_bufs[dim][side], get_send_view(Side(side), dim, halo))
            GC.@preserve recv_state MPI.Irecv!(recv_bufs[dim][side], comm, recv_reqs[idx];
                source=nbrank, tag=tag_recv(dim, side))
            GC.@preserve send_state MPI.Isend(send_bufs[dim][side], comm, send_reqs[idx];
                dest=nbrank, tag=tag_send(dim, side))
        end
    end
    GC.@preserve recv_state MPI.Waitall(recv_reqs)
    @inbounds for dim in 1:N, side in 1:2
        nbrank = topo.neighbors[dim][side]
        if nbrank != MPI.PROC_NULL
            copyto!(get_recv_view(Side(side), dim, halo), recv_bufs[dim][side])
        end
    end
    GC.@preserve send_state MPI.Waitall(send_reqs)
    return nothing
end

function start_halo_exchange_async_unsafe!(halo::HaloArray{T,N}) where {T,N}
    comm      = halo.topology.cart_comm
    topo      = halo.topology
    recv_reqs = halo.comm_state.unsafe_recv_reqs_vv
    send_reqs = halo.comm_state.unsafe_send_reqs_vv
    recv_bufs = halo.receive_bufs
    send_bufs = halo.send_bufs
    recv_state = (recv_reqs, recv_bufs)
    send_state = (send_reqs, send_bufs)
    @inbounds for dim in 1:N, side in 1:2
        nbrank = topo.neighbors[dim][side]
        if nbrank != MPI.PROC_NULL
            copyto!(send_bufs[dim][side], get_send_view(Side(side), dim, halo))
            GC.@preserve recv_state MPI.Irecv!(recv_bufs[dim][side], comm,
                recv_reqs[dim][side]; source=nbrank, tag=tag_recv(dim, side))
            GC.@preserve send_state MPI.Isend(send_bufs[dim][side], comm,
                send_reqs[dim][side]; dest=nbrank, tag=tag_send(dim, side))
        end
    end
    return nothing
end

function end_halo_exchange_async_wait_unsafe!(halo::HaloArray{T,N}) where {T,N}
    topo      = halo.topology
    recv_reqs = halo.comm_state.unsafe_recv_reqs_vv
    send_reqs = halo.comm_state.unsafe_send_reqs_vv
    recv_bufs = halo.receive_bufs
    recv_state = (recv_reqs, recv_bufs)
    send_state = send_reqs
    @inbounds for dim in 1:N, side in 1:2
        nbrank = topo.neighbors[dim][side]
        if nbrank != MPI.PROC_NULL
            GC.@preserve recv_state MPI.Wait(recv_reqs[dim][side])
            copyto!(get_recv_view(Side(side), dim, halo), recv_bufs[dim][side])
            GC.@preserve send_state MPI.Wait(send_reqs[dim][side])
        end
    end
    return nothing
end

# ---- safe (non-unsafe-request) async helpers --------------------------

function _start_halo_exchange_safe!(halo::HaloArray{T,N}) where {T,N}
    comm      = halo.topology.cart_comm
    topo      = halo.topology
    recv_reqs = halo.comm_state.recv_reqs
    send_reqs = halo.comm_state.send_reqs
    recv_bufs = halo.receive_bufs
    send_bufs = halo.send_bufs
    @inbounds for dim in 1:N, side in 1:2
        nbrank = topo.neighbors[dim][side]
        if nbrank != MPI.PROC_NULL
            copyto!(send_bufs[dim][side], get_send_view(Side(side), dim, halo))
            recv_reqs[dim][side] = MPI.Irecv!(recv_bufs[dim][side], comm, recv_reqs[dim][side];
                source=nbrank, tag=tag_recv(dim, side))
            send_reqs[dim][side] = MPI.Isend(send_bufs[dim][side], comm, send_reqs[dim][side];
                dest=nbrank, tag=tag_send(dim, side))
        end
    end
    return nothing
end

function _finish_halo_exchange_safe!(halo::HaloArray{T,N}) where {T,N}
    topo      = halo.topology
    recv_reqs = halo.comm_state.recv_reqs
    send_reqs = halo.comm_state.send_reqs
    recv_bufs = halo.receive_bufs
    @inbounds for dim in 1:N, side in 1:2
        nbrank = topo.neighbors[dim][side]
        if nbrank != MPI.PROC_NULL
            MPI.Wait(recv_reqs[dim][side])
            copyto!(get_recv_view(Side(side), dim, halo), recv_bufs[dim][side])
            MPI.Wait(send_reqs[dim][side])
        end
    end
    return nothing
end

# ---- public exchange API ----------------------------------------------

halo_exchange!(halo::HaloArray) = halo_exchange_waitall_unsafe!(halo)

start_halo_exchange!(halo::HaloArray)  = start_halo_exchange_async_unsafe!(halo)
finish_halo_exchange!(halo::HaloArray) = end_halo_exchange_async_wait_unsafe!(halo)

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

for (func, commutative) in [:mapreduce => true, :mapfoldl => false, :mapfoldr => false]
    @eval function Base.$func(
            f::F, op::OP, halo::HaloArray, etc::Vararg{HaloArray}; kws...,
        ) where {F<:Function,OP}
        comm   = get_comm(halo)
        ups    = map(interior_view, (halo, etc...))
        rlocal = $func(f, op, ups...; kws...)
        op_mpi = MPI.Op(op, typeof(rlocal); iscommutative=$commutative)
        MPI.Allreduce(rlocal, op_mpi, comm)
    end

    @eval function Base.$func(
            f::F, op::OP, z::Iterators.Zip{<:Tuple{Vararg{HaloArray}}}; kws...,
        ) where {F<:Function,OP}
        g(args...) = f(args)
        $func(g, op, z.is...; kws...)
    end
end

function Base.any(f::F, u::HaloArray) where {F<:Function}
    MPI.Allreduce(any(f, interior_view(u)) :: Bool, |, get_comm(u))
end

function Base.all(f::F, u::HaloArray) where {F<:Function}
    MPI.Allreduce(all(f, interior_view(u)) :: Bool, &, get_comm(u))
end

function mapreduce_haloarray_dims(f, op, ha::HaloArray{T,N,A,Halo}, dims) where {T,N,A,Halo}
    topo           = ha.topology
    root_coord     = 0
    dims_to_remove = Tuple(dims)
    dims_to_keep   = Tuple(i for i in 1:N if !(i in dims_to_remove))
    M = length(dims_to_keep)
    M == 0 && throw(ArgumentError("Reducing all dimensions to a scalar is not supported"))

    (sub_comm, coords, subrank) = subcomm_for_slices(topo, dims_to_remove)
    mpi_op      = MPI.Op(op, T; iscommutative=true)
    local_value = dropdims(mapreduce(f, op, interior_view(ha), dims=dims_to_remove),
                           dims=dims_to_remove)
    sum_on_root = MPI.Reduce(local_value, mpi_op, sub_comm, root=root_coord)

    root_topo    = root_topology_multi(topo, dims_to_remove; root_coord=root_coord)
    new_boundary = ntuple(i -> ha.boundary_condition[dims_to_keep[i]], Val(M))
    reduced_size = size(local_value)
    new_ha = HaloArray(T, reduced_size, Halo, root_topo; boundary_condition=new_boundary)

    isactive(root_topo) && (interior_view(new_ha) .= sum_on_root)
    sub_comm != MPI.COMM_NULL && MPI.free(sub_comm)

    return MaybeHaloArray(new_ha)
end

function mapreduce_mhaloarray_dims(f, op, mha::MultiHaloArray, dims)
    names = keys(mha.arrays)
    list_of_maybe = map_over_field(mha) do field
        mapreduce_haloarray_dims(f, op, field, dims)
    end
    active_states = map(isactive, values(list_of_maybe))
    if any(active_states) && !all(active_states)
        error("Inconsistent active state across reduced MultiHaloArray fields")
    end
    nt = NamedTuple{names}(map(getdata, values(list_of_maybe)))
    return MaybeHaloArray(MultiHaloArray(nt))
end
