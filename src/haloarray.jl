abstract type AbstractBoundaryCondition end

struct Reflecting <: AbstractBoundaryCondition end
struct Antireflecting <: AbstractBoundaryCondition end
struct Repeating <: AbstractBoundaryCondition end
struct Periodic <: AbstractBoundaryCondition end

struct Side{S}; end
@inline Side(s::Int) = Side{s}()

struct Dim{D}; end
@inline Dim(d::Int) = Dim{d}()


struct HaloCommState{N}
    recv_reqs::Vector{Vector{MPI.Request}} 
    send_reqs::Vector{Vector{MPI.Request}} 
    unsafe_recv_reqs_vv::Vector{Vector{MPI.UnsafeRequest}} 
    unsafe_send_reqs_vv::Vector{Vector{MPI.UnsafeRequest}} 
    recv_reqs_flat::Vector{MPI.Request} 
    send_reqs_flat::Vector{MPI.Request} 
    unsafe_recv_reqs::MPI.UnsafeMultiRequest
    unsafe_send_reqs::MPI.UnsafeMultiRequest
end


mutable struct HaloArray{T,N,A,Halo,B,BCondition} <: AbstractDistributedHaloArray{T,N}
    data::A
    topology::CartesianTopology{N}
    comm_state::HaloCommState{N}
    receive_bufs::B
    send_bufs::B
    boundary_condition::BCondition
end

@inline halo_backend(::Type{<:HaloArray}) = MPIHaloBackend()

function HaloArray{T,N,arraytype,Halo}(::UndefInitializer,boundary_condition) where {T,N,arraytype<:AbstractArray{T,N},Halo}
    data=arraytype(undef,ntuple(i -> 2*Halo, Val(N))...)
    topology = inactive_cartesian_topology(Val(N))
    comm_state = HaloCommState(N)
    receive_bufs = make_recv_buffers(data,Halo)
    send_bufs = make_send_buffers(data,Halo)
    bc = normalize_boundary_condition(boundary_condition, N)
    return HaloArray{T,N,typeof(data),Halo,typeof(receive_bufs),typeof(bc)}(data, topology, comm_state, receive_bufs, send_bufs, bc)
end


function HaloCommState(N::Int)
    
    recv_reqs = Vector{Vector{MPI.Request}}(undef, N)
    send_reqs = Vector{Vector{MPI.Request}}(undef, N)
    unsafe_recv_reqs_vv = Vector{Vector{MPI.UnsafeRequest}}(undef, N)
    unsafe_send_reqs_vv = Vector{Vector{MPI.UnsafeRequest}}(undef, N)
    for d in 1:N
        recv_reqs[d] = [MPI.Request() for _ in 1:2]
        send_reqs[d] = [MPI.Request() for _ in 1:2]
        unsafe_recv_reqs_vv[d] = [MPI.UnsafeRequest() for _ in 1:2]
        unsafe_send_reqs_vv[d] = [MPI.UnsafeRequest() for _ in 1:2]
    end
    recv_reqs_flat = vcat(recv_reqs...) # Flatten the nested vectors
    send_reqs_flat = vcat(send_reqs...) # Flatten the nested vectors
    unsafe_recv_reqs = MPI.UnsafeMultiRequest(length(recv_reqs_flat))
    unsafe_send_reqs = MPI.UnsafeMultiRequest(length(send_reqs_flat))

    HaloCommState{N}(
    recv_reqs, 
    send_reqs,
    unsafe_recv_reqs_vv,
    unsafe_send_reqs_vv,
    recv_reqs_flat,
    send_reqs_flat,
    unsafe_recv_reqs,
    unsafe_send_reqs
)
end



@inline function get_comm(halo::HaloArray)
    halo.topology.cart_comm
end

@inline Base.length(halo::HaloArray) = prod(size(halo))
@inline Base.size(halo::HaloArray) = global_size(halo)
@inline Base.size(halo::HaloArray, i::Int) = size(halo)[i]
@inline Base.eltype(ha::HaloArray{T,N,A,Halo,B,BCondition}) where {T,N,A,Halo,B,BCondition} = T
@inline Base.axes(halo::HaloArray) = map(Base.OneTo, size(halo))
@inline Base.axes(halo::HaloArray, i::Int) = Base.OneTo(size(halo, i))
@inline owned_axes(halo::HaloArray) = axes(interior_view(halo))
@inline owned_axes(halo::HaloArray, i::Int) = axes(interior_view(halo), i)
@inline Base.eachindex(halo::HaloArray) = eachindex(interior_view(halo))
@inline Base.iterate(halo::HaloArray) = iterate(interior_view(halo))
@inline Base.iterate(halo::HaloArray, state) = iterate(interior_view(halo), state)

@inline function interior_size(halo::HaloArray{T,N,A,Halo,B,BCondition}) where {T,N,A,Halo,B,BCondition}
    h = halo_width(halo)
    ntuple(i -> size(halo.data, i) - 2 * h, Val(N))
end

@inline Base.eltype(::Type{<:HaloArray{T}}) where {T} = T

@inline storage_size(halo::HaloArray) = size(halo.data)
@inline storage_size(halo::HaloArray, i::Int) = size(halo.data,i)


@inline halo_width(::Type{HaloArray{T,N,A,Halo,B,BCondition}}) where {T,N,A,Halo,B,BCondition} = Halo

@inline halo_width(halo::HaloArray{T,N,A,Halo,B,BCondition}) where {T,N,A,Halo,B,BCondition} = Halo

@inline Base.ndims(halo::HaloArray{T,N,A,Halo,B,BCondition}) where {T,N,A,Halo,B,BCondition} = N

@inline Base.ndims(::Type{HaloArray{T,N,A,Halo,B,BCondition}}) where {T,N,A,Halo,B,BCondition} = N

@inline Base.parent(halo::HaloArray) = halo.data


function build_haloarray_from_data(data::AbstractArray{T,N}, halo::Int, topology::CartesianTopology{N}, boundary_condition_raw) where {T,N}
 
    bc = normalize_boundary_condition(boundary_condition_raw, N)
 
    recv_bufs = make_recv_buffers(data, halo)
    send_bufs = make_send_buffers(data, halo)
 
    validate_boundary_condition(topology, bc)
 
    comm_state = HaloCommState(N)
    HaloArray{T, N, typeof(data), halo, typeof(recv_bufs), typeof(bc)}(data, topology, comm_state, recv_bufs, send_bufs, bc)
end

isactive(a::HaloArray) = isactive(a.topology)
is_root(a::HaloArray; root::Integer=0) = is_root(a.topology; root=root)

function HaloArray(data::AbstractArray{T,N}, halo::Int, topology::CartesianTopology{N}, boundary_condition) where {T,N}
    return build_haloarray_from_data(data, halo, topology, boundary_condition)
end

function HaloArray(::Type{T}, owned_dims::NTuple{N,Int}, halo::Int, topology::CartesianTopology{N}; boundary_condition=(ntuple(_ -> (Repeating(), Repeating()), Val(N)))) where {T,N}
    fullsize = ntuple(i -> owned_dims[i] + 2 * halo, Val(N))
    data = zeros(T, fullsize...)
    return build_haloarray_from_data(data, halo, topology, boundary_condition)
end

function HaloArray(::Type{T}, owned_dims::NTuple{N,Int}, halo::Int, boundary_condition::NTuple{N,NTuple{2,AbstractBoundaryCondition}}) where {T,N}
    if !MPI.Initialized()
        MPI.Init()
    end
    comm = MPI.COMM_WORLD
    dims_guess = ntuple(i -> 0, Val(N))
    periodic = infer_periodicity(normalize_boundary_condition(boundary_condition, N))
    topology = CartesianTopology(comm, dims_guess; periodic=periodic)
    return HaloArray(T, owned_dims, halo, topology; boundary_condition=boundary_condition)
end


# Accept legacy positional boundary_condition argument (fifth positional arg)
function HaloArray(::Type{T}, owned_dims::NTuple{N,Int}, halo::Int, topology::CartesianTopology{N}, boundary_condition) where {T,N}
    bc_norm = normalize_boundary_condition(boundary_condition, N)
    return HaloArray(T, owned_dims, halo, topology; boundary_condition=bc_norm)
end

function HaloArray(::Type{T}, owned_dims::NTuple{N,Int}, halo::Int; boundary_condition=(ntuple(_ -> (Repeating(), Repeating()), Val(N)))) where {T,N}
    HaloArray(T, owned_dims, halo, normalize_boundary_condition(boundary_condition, N))
end

function HaloArray(owned_dims::NTuple{N,Int}, halo::Int; boundary_condition=(ntuple(_ -> (Repeating(), Repeating()), Val(N)))) where {N}
    HaloArray(Float64, owned_dims, halo, normalize_boundary_condition(boundary_condition, N))
end



function _global_to_owned_dims(halo::HaloArray{T,N}, dims::NTuple{M,<:Integer}) where {T,N,M}
    M == N || throw(DimensionMismatch("HaloArray similar dims must have $N dimensions"))
    topo_dims = isactive(halo) ? halo.topology.dims : ntuple(_ -> 1, Val(N))
    all(d -> Int(dims[d]) % topo_dims[d] == 0, 1:N) ||
        throw(DimensionMismatch("HaloArray global similar dims $dims are not divisible by topology dims $topo_dims"))
    return ntuple(d -> Int(dims[d]) ÷ topo_dims[d], Val(N))
end

function Base.similar(halo::HaloArray{T,N,A,H,B,BCondition}, ::Type{AA},
        dims::Dims{M}) where {T,N,A,H,B,BCondition,AA,M}
    owned_dims = _global_to_owned_dims(halo, dims)
    fullsize = ntuple(i -> owned_dims[i] + 2 * halo_width(halo), Val(N))
    data = similar(parent(halo), AA, fullsize)
    return build_haloarray_from_data(data, halo_width(halo), halo.topology, halo.boundary_condition)
end

Base.similar(halo::HaloArray{T,N,A,H,B,BCondition}, ::Type{AA},
    dims::NTuple{M,<:Integer}) where {T,N,A,H,B,BCondition,AA,M} =
    similar(halo, AA, ntuple(d -> Int(dims[d]), Val(M)))

Base.similar(halo::HaloArray) = similar(halo, eltype(halo), size(halo))
Base.similar(halo::HaloArray, ::Type{AA}) where {AA} = similar(halo, AA, size(halo))
Base.similar(halo::HaloArray, dims::Dims{M}) where {M} = similar(halo, eltype(halo), dims)
Base.similar(halo::HaloArray, dims::NTuple{M,<:Integer}) where {M} =
    similar(halo, eltype(halo), dims)

@inline function get_send_view(::Side{1}, ::Dim{D}, array::AbstractArray, halo::Int) where {D}
    send_range = (halo + 1):(2 * halo)
    indices = ntuple(I -> I == D ? send_range : (halo+1):(size(array,I)-halo), Val(ndims(array)))
    view(array, indices...)
end

@inline function get_send_view(::Side{2}, ::Dim{D}, array::AbstractArray, halo::Int) where {D}
    send_range = (size(array, D) - 2 * halo + 1):(size(array, D) - halo)
    indices = ntuple(I -> I == D ? send_range : (halo+1):(size(array,I)-halo), Val(ndims(array)))
    view(array, indices...)
end

function get_recv_view(::Side{1}, ::Dim{D}, array::AbstractArray, halo::Int) where {D}
    recv_range = 1:halo
    indices = ntuple(ndims(array)) do I 
        if I == D 
            recv_range
        else 
          (halo+1):(size(array,I)-halo)
        end   
    end 
    view(array, indices...)
end

@inline function get_recv_view(::Side{2}, ::Dim{D}, array::AbstractArray, halo::Int) where {D}
    recv_range = (size(array, D) - halo + 1):(size(array, D))
    indices = ntuple(I -> I == D ? recv_range : (halo+1):(size(array,I)-halo), ndims(array))
    view(array, indices...)
end

@inline function get_send_view(::Side{1}, ::Dim{D}, array::HaloArray{T,N,A,Halo,B,BCondition}) where {D, T,N,A,Halo,B,BCondition}
    send_range = (halo_width(array) + 1):(2 * halo_width(array))
    indices = ntuple(I -> I == D ? send_range : (halo_width(array)+1):(storage_size(array,I)-halo_width(array)), Val(N))
    view(parent(array), indices...)
end

@inline function get_send_view(::Side{2}, ::Dim{D}, array::HaloArray{T,N,A,Halo,B,BCondition}) where {D, T,N,A,Halo,B,BCondition}
    send_range = (storage_size(array, D) - 2 * halo_width(array) + 1):(storage_size(array, D) - halo_width(array))
    indices = ntuple(I -> I == D ? send_range : (halo_width(array)+1):(storage_size(array,I)-halo_width(array)), Val(N))
    view(parent(array), indices...)
end

@inline function get_recv_view(::Side{1}, ::Dim{D}, array::HaloArray{T,N,A,Halo,B,BCondition}) where {D, T,N,A,Halo,B,BCondition}
    recv_range = 1:halo_width(array)
    indices = ntuple(I -> I == D ? recv_range : (halo_width(array)+1):(storage_size(array,I)-halo_width(array)), Val(N))
    view(parent(array), indices...)
end

@inline function get_recv_view(::Side{2}, ::Dim{D}, array::HaloArray{T,N,A,Halo,B,BCondition}) where {D, T, N, A, Halo, B, BCondition}
    recv_range = (storage_size(array, D) - halo_width(array) + 1):(storage_size(array, D))
    indices = ntuple(I -> I == D ? recv_range : (halo_width(array)+1):(storage_size(array,I)-halo_width(array)), Val(N))
    view(parent(array), indices...)
end

@inline function get_send_view(::Side{1}, D::Int, array::HaloArray{T,N,A,Halo,B,BCondition}) where { T, N,A,Halo,B,BCondition}
    send_range = (halo_width(array) + 1):(2 * halo_width(array))
    indices = ntuple(I -> I == D ? send_range : (halo_width(array)+1):(storage_size(array,I)-halo_width(array)), Val(N))
    view(parent(array), indices...)
end

@inline function get_send_view(::Side{2}, D::Int, array::HaloArray{T,N,A,Halo,B,BCondition}) where { T, N,A,Halo,B,BCondition}
    send_range = (storage_size(array, D) - 2 * halo_width(array) + 1):(storage_size(array, D) - halo_width(array))
    indices = ntuple(I -> I == D ? send_range : (halo_width(array)+1):(storage_size(array,I)-halo_width(array)), Val(N))
    view(parent(array), indices...)
end

@inline function get_recv_view(::Side{1}, D::Int, array::HaloArray{T,N,A,Halo,B,BCondition}) where { T,N,A,Halo,B,BCondition}
    recv_range = 1:halo_width(array)
    indices = ntuple(I -> I == D ? recv_range : (halo_width(array)+1):(storage_size(array,I)-halo_width(array)), Val(N))
    view(parent(array), indices...)
end

@inline function get_recv_view(::Side{2}, D::Int, array::HaloArray{T,N,A,Halo,B,BCondition}) where { T, N,A,Halo,B,BCondition}
    recv_range = (storage_size(array, D) - halo_width(array) + 1):(storage_size(array, D))
    indices = ntuple(I -> I == D ? recv_range : (halo_width(array)+1):(storage_size(array,I)-halo_width(array)), Val(N))
    view(parent(array), indices...)
end


@inline function interior_range(halo::HaloArray)
    h = halo_width(halo)
    N=ndims(halo)
    ntuple(i -> (h + 1):(storage_size(halo, i) - h), Val(N))
end

@inline function full_range(halo::HaloArray)
    h = halo_width(halo)
    N=ndims(halo)
    ntuple(i -> ( 1):(storage_size(halo, i) ), Val(N))
end

@inline function interior_view(halo::HaloArray)
    h = halo_width(halo)
    ranges = interior_range(halo)
    @views return halo.data[ranges...]
end



function full_view(halo::HaloArray)
    ranges = full_range(halo)
    @views return halo.data[ranges...]
end



"""
    owned_to_global_index(halo::HaloArray, owned_idx::NTuple{N,<:Integer})

Convert an owned interior index (excluding halo) to global index (1-based).
"""
function owned_to_global_index(halo::HaloArray{T,N,A,Halo,B,BCondition}, owned_idx::NTuple{N,<:Integer}) where {T,N,A,Halo,B,BCondition}
    coords = halo.topology.cart_coords
    owned_dims = interior_size(halo)
    all(i -> 1 <= owned_idx[i] <= owned_dims[i], 1:N) ||
        throw(BoundsError(halo, owned_idx))
    global_idx = ntuple(i -> coords[i] * owned_dims[i] + owned_idx[i], Val(N))
    return global_idx
end

function global_to_storage_index(halo::HaloArray, global_idx::NTuple{N,<:Integer}) where {N}
    owned_dims = interior_size(halo)
    coords = halo.topology.cart_coords
    h = halo_width(halo)

    owner_coords = ntuple(i -> (global_idx[i] - 1) ÷ owned_dims[i], Val(N))

    if owner_coords != coords
        return nothing
    end

    interior_idx = ntuple(i -> global_idx[i] - coords[i] * owned_dims[i], Val(N))

    storage_idx = ntuple(i -> interior_idx[i] + h, Val(N))
    return storage_idx
end

@inline function _owned_global_to_storage_index(halo::HaloArray, I)
    idx = _check_global_scalar_indices(halo, I)
    storage_idx = global_to_storage_index(halo, idx)
    storage_idx === nothing &&
        throw(ArgumentError("Global index $idx is not owned by this MPI rank; HaloArray scalar indexing is local-only and does not communicate."))
    return storage_idx
end

function Base.getindex(halo::HaloArray, I::Vararg{Integer})
    @warn "Global scalar getindex on HaloArray is local-only and intended for diagnostics, not hot loops; use interior_view/owned_axes for kernels." maxlog=1
    storage_idx = _owned_global_to_storage_index(halo, I)
    @inbounds return parent(halo)[storage_idx...]
end

function Base.setindex!(halo::HaloArray, value, I::Vararg{Integer})
    @warn "Global scalar setindex! on HaloArray is local-only and intended for diagnostics, not hot loops; use interior_view/owned_axes for kernels." maxlog=1
    storage_idx = _owned_global_to_storage_index(halo, I)
    @inbounds parent(halo)[storage_idx...] = value
    return halo
end

@inline function is_in_rank(halo::HaloArray, global_idx::NTuple{N,<:Integer}) where {N}
    coords = halo.topology.cart_coords
    owner_coords = owner_coordinates(halo, global_idx)
    return (owner_coords == coords)
end

@inline function owner_coordinates(halo::HaloArray, global_idx::NTuple{N,<:Integer}) where {N}
    owned_dims = interior_size(halo)
    ntuple(i -> (global_idx[i] - 1) ÷ owned_dims[i], Val(N))
end

@inline function versors(::Val{N}) where { N}
    return ntuple(i -> ntuple(j -> ifelse(i == j, 1, 0), Val(N)), Val(N))
end

@inline function versors(::HaloArray{T,N,A,Halo,B,BCondition}) where {T,N,A,Halo,B,BCondition}
    return versors(Val(N))
end

function Base.copyto!(dest::HaloArray, src::HaloArray)
    copyto!(parent(dest), parent(src))
    return dest
end

function Base.copy(src::HaloArray)
    new_halo = similar(src)
    copyto!(new_halo, src)
    return new_halo
end

function Base.zero(halo::HaloArray)
    z = similar(halo)
    fill!(z, zero(eltype(halo)))
    return z
end

function Base.fill!(halo::HaloArray,num)
    Base.fill!(parent(halo), num)
    return halo
end

function fill_interior(halo::HaloArray,num) 
    fill!(interior_view(halo), num)
    return halo
end


function Base.map!(f,dest::HaloArray,src::HaloArray )
    dest_interior = interior_view(dest)
    src_interior = interior_view(src)
    @views map!(f, dest_interior, src_interior)
    return dest
end

function Base.map!(f,dest::HaloArray,src::Vararg{ HaloArray,2} )
    dest_interior = interior_view(dest)
    src_interior_1 = interior_view(src[1])
    src_interior_2 = interior_view(src[2])
    @views map!(f, dest_interior, src_interior_1, src_interior_2)
    return dest
end

function Base.map!(f,dest::HaloArray,src::Vararg{ HaloArray,3} ) 
    dest_interior = interior_view(dest)
    src_interior_1 = interior_view(src[1])
    src_interior_2 = interior_view(src[2])
    src_interior_3 = interior_view(src[3])
    map!(f, dest_interior, src_interior_1, src_interior_2, src_interior_3)
    return dest
end


function Base.map!(f, dest::HaloArray, src::Vararg{ HaloArray, N }) where {N}
    dest_interior = interior_view(dest)
    src_interiors = map(src) do s
        interior_view(s)
    end
    map!(f, dest_interior, src_interiors...)
    return dest
end

function Base.map(f,src::Vararg{HaloArray,N}) where {N} 
    similar_src = similar(src[1])
    map!(f,similar_src,src...)
    return similar_src
end

Base.:/(halo::HaloArray, x::Number) = halo ./ x
Base.:*(halo::HaloArray, x::Number) = halo .* x
Base.:*(x::Number, halo::HaloArray) = x .* halo

function LinearAlgebra.norm(halo::HaloArray, p::Real=2)
    if p == 2
        return sqrt(mapreduce(abs2, +, halo))
    elseif p == Inf
        return mapreduce(abs, max, halo)
    else
        return mapreduce(x -> abs(x)^p, +, halo)^(1 / p)
    end
end


function Base.foreach(f, halo::HaloArray{T,N,A,Halo,B,BCondition}) where {T, N,A,Halo,B,BCondition}
    interior = interior_view(halo)
    foreach(f, interior)

end


function fill_from_global_indices!(f,halo::HaloArray)
    h = halo_width(halo)
    local_shape =interior_range(halo)

    for local_I in CartesianIndices(local_shape)
        full_I = Tuple(local_I)
        local_interior_I = ntuple(i -> full_I[i] - h, Val(ndims(halo)))
        global_I = owned_to_global_index(halo, local_interior_I)
        halo.data[local_I] = f(global_I)
    end

    return halo
end

function fill_from_local_indices!(f,halo::HaloArray)
    interior = interior_view(halo)
    inds = CartesianIndices(interior)
    for I in inds
        interior[I] = f(Tuple(I)...)  # splat indices as arguments
    end
    return nothing
end


function Base.show(io::IO, obj::HaloArray)
    print(io, "HaloArray of global size ", size(obj), " (owned size: ", owned_size(obj), ", storage size: ", storage_size(obj), "), halo width: ", halo_width(obj), "\n")
    print(io, "  eltype: ", eltype(obj), "\n")
    print(io, "  topology: ", obj.topology, "\n")
    print(io, "  boundary_condition: ", obj.boundary_condition, "\n")
end

function Base.show(io::IO, mime::MIME"text/plain", obj::HaloArray)
    println(io, "HaloArray (storage size: ", storage_size(obj), ", halo width: ", halo_width(obj), ")")
    println(io, "  eltype: ", eltype(obj))
    println(io, "  topology: ", obj.topology)
    println(io, "  boundary_condition: ", obj.boundary_condition)
    println(io, "  interior data preview:")
    show(io, mime, interior_view(obj))
end

function global_size(halo::HaloArray)
    N = ndims(halo)
    local_interior = owned_size(halo)     # Tuple with owned interior sizes per dimension
    dims = halo.topology.dims             # Tuple with number of processes per dimension
    
    return ntuple(i -> local_interior[i] * dims[i], Val(N))
end

function validate_boundary_condition(topology::CartesianTopology{N}, boundary_condition) where {N}

    if !isactive(topology)
        return true
    end

    for d in 1:N
        left, right = boundary_condition[d]

        if !(left isa AbstractBoundaryCondition) || !(right isa AbstractBoundaryCondition)
            error("boundary_condition[$d] must be a tuple of AbstractBoundaryCondition (got $(left), $(right))")
        end

        topo_is_periodic = topology.periodic_boundary_condition[d]
        both_periodic = (left isa Periodic) && (right isa Periodic)
        any_periodic = (left isa Periodic) || (right isa Periodic)

        if topo_is_periodic && !both_periodic
            error("Topology is periodic in dimension $d but boundary_condition[$d] is not (both sides must be Periodic).")
        elseif !topo_is_periodic && any_periodic
            error("Boundary condition in dimension $d uses Periodic but topology is not periodic.
            $d: $(topology), boundary_condition[$d]: ($(left), $(right)), isactive: $(isactive(topology))")
        end
    end

    return true
end

function make_recv_buffers(data::AbstractArray{T,N}, halo::Int) where {T,N}
    ntuple(D -> ntuple(S -> similar(get_recv_view(Side(S), Dim(D), data, halo)), Val(2)), Val(N))
end

function make_send_buffers(data::AbstractArray{T,N}, halo::Int) where {T,N}
    ntuple(D -> ntuple(S -> similar(get_send_view(Side(S), Dim(D), data, halo)), Val(2)), Val(N))
end
