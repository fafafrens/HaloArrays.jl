abstract type AbstractBoundaryCondition end

struct Reflecting <: AbstractBoundaryCondition end
struct Antireflecting <: AbstractBoundaryCondition end
struct Repeating <: AbstractBoundaryCondition end
struct Periodic<: AbstractBoundaryCondition end

# -- Helper types --
struct Side{S}; end
@inline Side(s::Int) = Side{s}()

struct Dim{D}; end
@inline Dim(d::Int) = Dim{d}()
# -- HaloArrays module (simplified inline) --


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


mutable struct HaloArray{T,N,A,Halo,B,BCondition}  # removed `Size` type parameter
    data::A
    topology::CartesianTopology{N}
    comm_state::HaloCommState{N}
    receive_bufs::B
    send_bufs::B
    boundary_condition::BCondition
end


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

Base.length(halo::HaloArray) = length(halo.data)
@inline Base.size(halo::HaloArray) = interior_size(halo)
@inline Base.size(halo::HaloArray, i::Int) = interior_size(halo)[i]
@inline Base.eltype(ha::HaloArray{T,N,A,Halo,B,BCondition}) where {T,N,A,Halo,B,BCondition} = T

@inline function interior_size(halo::HaloArray{T,N,A,Halo,B,BCondition}) where {T,N,A,Halo,B,BCondition}
    h = halo_width(halo)
    ntuple(i -> size(halo.data, i) - 2 * h, Val(N))
end

@inline full_size(halo::HaloArray) = size(halo.data)
@inline full_size(halo::HaloArray, i::Int) = size(halo.data,i)

@inline halo_width(halo::HaloArray{T,N,A,Halo,B,BCondition}) where {T,N,A,Halo,B,BCondition} = Halo
@inline Base.ndims(halo::HaloArray{T,N,A,Halo,B,BCondition}) where {T,N,A,Halo,B,BCondition} = N
@inline Base.parent(halo::HaloArray) = halo.data
Base.axes(x::HaloArray) = axes(interior_view(x))


isactive(a::HaloArray) = isactive(a.topology)


function  HaloArray(data::AbstractArray{T,N},halo::Int, topology::CartesianTopology{N},
    boundary_condition) where {T,N}

    # type-stable NTuple{N,NTuple{2,Array}} buffers
    recv_bufs = make_recv_buffers(data, halo)
    send_bufs = make_send_buffers(data, halo)
 
     #we check that the boundary condition is consistent with the topology
     validate_boundary_condition(topology, boundary_condition)
 
     # Create the HaloArray with all necessary fields
 
     comm_state=HaloCommState(N)
    return HaloArray{T, N, typeof(data), halo, typeof(recv_bufs), typeof(boundary_condition)}(
        data, topology, comm_state, recv_bufs, send_bufs, boundary_condition)
 end


# Factory minima: costruisce HaloArray da un array esistente, normalizza BC e crea buffer
function build_haloarray_from_data(data::AbstractArray{T,N}, halo::Int, topology::CartesianTopology{N}, boundary_condition_raw) where {T,N}
 
     # Normalizza la rappresentazione della BC (assume normalize_boundary_condition disponibile)
     bc = normalize_boundary_condition(boundary_condition_raw, N)
 
    # type-stable NTuple buffers
    recv_bufs = make_recv_buffers(data, halo)
   send_bufs = make_send_buffers(data, halo)
 
     # verifica che la BC sia coerente con la topology
     validate_boundary_condition(topology, bc)
 
     comm_state = HaloCommState(N)
    HaloArray{T, N, typeof(data), halo, typeof(recv_bufs), typeof(bc)}(data, topology, comm_state, recv_bufs, send_bufs, bc)
 end

# Costruttore principale da array già pieno (mantiene il parent fornito)
function HaloArray(data::AbstractArray{T,N}, halo::Int, topology::CartesianTopology{N}, boundary_condition) where {T,N}
    return build_haloarray_from_data(data, halo, topology, boundary_condition)
end

# Costruttore da tipo + interior sizes + topology (crea data zeros)
function HaloArray(::Type{T}, local_inner_size::NTuple{N,Int}, halo::Int, topology::CartesianTopology{N}; boundary_condition=(ntuple(_ -> (Repeating(), Repeating()), Val(N)))) where {T,N}
    fullsize = ntuple(i -> local_inner_size[i] + 2 * halo, Val(N))
    data = zeros(T, fullsize...)
    # Delegate normalization to build_haloarray_from_data (single normalization point)
    return build_haloarray_from_data(data, halo, topology, boundary_condition)
end

# Costruttore che sceglie la topologia MPI automaticamente (usa MPI.COMM_WORLD)
function HaloArray(::Type{T}, local_inner_size::NTuple{N,Int}, halo::Int, boundary_condition::NTuple{N,NTuple{2,AbstractBoundaryCondition}}) where {T,N}
    if !MPI.Initialized()
        MPI.Init()
    end
    comm = MPI.COMM_WORLD
    dims_guess = ntuple(i -> 0, Val(N))
    periodic = infer_periodicity(normalize_boundary_condition(boundary_condition, N))
    topology = CartesianTopology(comm, dims_guess; periodic=periodic)
    return HaloArray(T, local_inner_size, halo, topology; boundary_condition=boundary_condition)
end


# Accept legacy positional boundary_condition argument (fifth positional arg)
function HaloArray(::Type{T}, local_inner_size::NTuple{N,Int}, halo::Int, topology::CartesianTopology{N}, boundary_condition) where {T,N}
    # normalize boundary condition (accept Symbols/Types/instances)
    bc_norm = normalize_boundary_condition(boundary_condition, N)
    return HaloArray(T, local_inner_size, halo, topology; boundary_condition=bc_norm)
end

function HaloArray(::Type{T}, local_inner_size::NTuple{N,Int}, halo::Int; boundary_condition=(ntuple(_ -> (Repeating(), Repeating()), Val(N)))) where {T,N}
    # Create a HaloArray without explicit topology (auto-choose topology). The auto-topology
    # variant needs a normalized BC to infer periodicity, so normalize here and delegate.
    HaloArray(T, local_inner_size, halo, normalize_boundary_condition(boundary_condition, N))
end

function HaloArray(local_inner_size::NTuple{N,Int}, halo::Int; boundary_condition=(ntuple(_ -> (Repeating(), Repeating()), Val(N)))) where {N}
    HaloArray(Float64, local_inner_size, halo, normalize_boundary_condition(boundary_condition, N))
end



function Base.similar(halo::HaloArray, element_type=eltype(halo) ,
    dims::NTuple{M,Int64}=interior_size(halo)
    ) where {M}
    # Create a new HaloArray with given interior dims, preserving halo_width and topology
    HaloArray(element_type, dims ,halo_width(halo), halo.topology; boundary_condition=halo.boundary_condition)
end

#this function is to give back a view of a generic array with singleton and on abstract array 
@inline function get_send_view(::Side{1}, ::Dim{D}, array::AbstractArray, halo::Int) where {D}
    send_range = (halo + 1):(2 * halo)
    #indices = ntuple(I -> I == D ? send_range : Colon(), Val(ndims(array)))
    indices = ntuple(I -> I == D ? send_range : (halo+1):(size(array,I)-halo), Val(ndims(array)))
    view(array, indices...)
end

@inline function get_send_view(::Side{2}, ::Dim{D}, array::AbstractArray, halo::Int) where {D}
    send_range = (size(array, D) - 2 * halo + 1):(size(array, D) - halo)
    #indices = ntuple(I -> I == D ? send_range : Colon(), Val(ndims(array)))
    indices = ntuple(I -> I == D ? send_range : (halo+1):(size(array,I)-halo), Val(ndims(array)))
    view(array, indices...)
end

function get_recv_view(::Side{1}, ::Dim{D}, array::AbstractArray, halo::Int) where {D}
    recv_range = 1:halo
    #indices = ntuple(I -> I == D ? recv_range : Colon(), Val(ndims(array)))
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
    #indices = ntuple(I -> I == D ? recv_range : Colon(), Val(ndims(array)))
    indices = ntuple(I -> I == D ? recv_range : (halo+1):(size(array,I)-halo), ndims(array))
    view(array, indices...)
end


#this are the function specialized for HaloArray
@inline function get_send_view(::Side{1}, ::Dim{D}, array::HaloArray{T,N,A,Halo,B,BCondition}) where {D, T,N,A,Halo,B,BCondition}
    send_range = (halo_width(array) + 1):(2 * halo_width(array))
    indices = ntuple(I -> I == D ? send_range : (halo_width(array)+1):(full_size(array,I)-halo_width(array)), Val(N))
    view(parent(array), indices...)
end

@inline function get_send_view(::Side{2}, ::Dim{D}, array::HaloArray{T,N,A,Halo,B,BCondition}) where {D, T,N,A,Halo,B,BCondition}
    send_range = (full_size(array, D) - 2 * halo_width(array) + 1):(full_size(array, D) - halo_width(array))
    indices = ntuple(I -> I == D ? send_range : (halo_width(array)+1):(full_size(array,I)-halo_width(array)), Val(N))
    view(parent(array), indices...)
end

@inline function get_recv_view(::Side{1}, ::Dim{D}, array::HaloArray{T,N,A,Halo,B,BCondition}) where {D, T,N,A,Halo,B,BCondition}
    recv_range = 1:halo_width(array)
    indices = ntuple(I -> I == D ? recv_range : (halo_width(array)+1):(full_size(array,I)-halo_width(array)), Val(N))
    view(parent(array), indices...)
end

@inline function get_recv_view(::Side{2}, ::Dim{D}, array::HaloArray{T,N,A,Halo,B,BCondition}) where {D, T, N, A, Halo, B, BCondition}
    recv_range = (full_size(array, D) - halo_width(array) + 1):(full_size(array, D))
    indices = ntuple(I -> I == D ? recv_range : (halo_width(array)+1):(full_size(array,I)-halo_width(array)), Val(N))
    view(parent(array), indices...)
end

#this are the function specialized for HaloArray and the dimension as Int for perfomance resons
@inline function get_send_view(::Side{1}, D::Int, array::HaloArray{T,N,A,Halo,B,BCondition}) where { T, N,A,Halo,B,BCondition}
    send_range = (halo_width(array) + 1):(2 * halo_width(array))
    indices = ntuple(I -> I == D ? send_range : (halo_width(array)+1):(full_size(array,I)-halo_width(array)), Val(N))
    view(parent(array), indices...)
end

@inline function get_send_view(::Side{2}, D::Int, array::HaloArray{T,N,A,Halo,B,BCondition}) where { T, N,A,Halo,B,BCondition}
    send_range = (full_size(array, D) - 2 * halo_width(array) + 1):(full_size(array, D) - halo_width(array))
    indices = ntuple(I -> I == D ? send_range : (halo_width(array)+1):(full_size(array,I)-halo_width(array)), Val(N))
    view(parent(array), indices...)
end

@inline function get_recv_view(::Side{1}, D::Int, array::HaloArray{T,N,A,Halo,B,BCondition}) where { T,N,A,Halo,B,BCondition}
    recv_range = 1:halo_width(array)
    indices = ntuple(I -> I == D ? recv_range : (halo_width(array)+1):(full_size(array,I)-halo_width(array)), Val(N))
    view(parent(array), indices...)
end

@inline function get_recv_view(::Side{2}, D::Int, array::HaloArray{T,N,A,Halo,B,BCondition}) where { T, N,A,Halo,B,BCondition}
    recv_range = (full_size(array, D) - halo_width(array) + 1):(full_size(array, D))
    indices = ntuple(I -> I == D ? recv_range : (halo_width(array)+1):(full_size(array,I)-halo_width(array)), Val(N))
    view(parent(array), indices...)
end


# --- Helper to get interior or full data for broadcast ---
@inline function interior_range(halo::HaloArray)
    h = halo_width(halo)
    N=ndims(halo)
    ntuple(i -> (h + 1):(full_size(halo, i) - h), Val(N))
end

@inline function full_range(halo::HaloArray)
    h = halo_width(halo)
    N=ndims(halo)
    ntuple(i -> ( 1):(full_size(halo, i) ), Val(N))
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
    local_to_global_index(halo::HaloArray, local_idx::NTuple{N,Int}) -> NTuple{N,Int}

Convert a local interior index (excluding halo) to global index (1-based).
"""
function local_to_global_index(halo::HaloArray{T,N,A,Halo,B,BCondition}, local_idx::NTuple{N,Int}) where {T,N,A,Halo,B,BCondition}
    coords = halo.topology.cart_coords
    size_local = interior_size(halo)
    global_idx = ntuple(i -> coords[i] * size_local[i] + local_idx[i]-halo_width(halo), Val(N))
    return global_idx
end

function global_to_local_index(halo::HaloArray, global_idx::NTuple{N,Int}) where {N}
    size_local = interior_size(halo)
    coords = halo.topology.cart_coords
    h = halo_width(halo)

    # Compute which tile should own this global index
    owner_coords = ntuple(i -> (global_idx[i] - 1) ÷ size_local[i], Val(N))

    # If this rank doesn't own it, return nothing
    if owner_coords != coords
        return nothing
    end

    # Compute the interior local index
    interior_idx = ntuple(i -> global_idx[i] - coords[i] * size_local[i], Val(N))

    # Convert to full local index (i.e., add halo)
    local_idx = ntuple(i -> interior_idx[i] + h, Val(N))
    return local_idx
end

@inline function is_in_rank(halo::HaloArray, global_idx::NTuple{N,Int}) where {N}
    coords = halo.topology.cart_coords
    owner_coords = owner_coordinares(halo,global_idx)
    return (owner_coords == coords)
end 

@inline function owner_coordinares(halo::HaloArray, global_idx::NTuple{N,Int}) where {N}
    size_local = interior_size(halo)
    ntuple(i -> (global_idx[i] - 1) ÷ size_local[i], Val(N))
end 

#function local_offset(halo::HaloArray)
#    size_local =interior_size(halo)
#    return ntuple(i -> halo.topology.cart_coords[i] * size_local[i], Val(ndims(halo)))
#end
#
#
#function owns_global_index(halo::HaloArray, global_idx::NTuple{N,Int}) where {N}
#    offset = local_offset(halo)
#    size =interior_size(halo)
#
#    return all(i -> (offset[i] < global_idx[i] ≤ offset[i] + size[i]), 1:N)
#end
#
#
#function global_to_local_index(halo::HaloArray, global_idx::NTuple{N,Int}) where {N}
#    if !owns_global_index(halo, global_idx)
#        return nothing
#    end
#
#    offset = local_offset(halo)
#    h = halo_width(halo)
#
#    return ntuple(i -> (global_idx[i] - offset[i]) + h[i], Val(N))
#end


@inline function versors(::Val{N}) where { N}
    return ntuple(i -> ntuple(j -> ifelse(i == j, 1, 0), Val(N)), Val(N))
end

@inline function versors(::HaloArray{T,N,A,Halo,B,BCondition}) where {T,N,A,Halo,B,BCondition}
    return versors(Val(N))
end

function Base.copyto!(dest::HaloArray, src::HaloArray)
    @assert size(dest) == size(src) "Incompatible array sizes"
    copyto!(parent(dest), parent(src))
end

function Base.copy(src::HaloArray)
    new_halo = similar(src)
    copyto!(new_halo, src)
    return new_halo
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
    @assert size(dest_interior) == size(src_interior) "Incompatible array sizes"
    @views map!(f, dest_interior, src_interior)
    halo_exchange!(dest) # Use @views to avoid unnecessary copies
    return dest
end

function Base.map!(f,dest::HaloArray,src::Vararg{ HaloArray,2} )
    dest_interior = interior_view(dest)
    src_interior_1 = interior_view(src[1])
    src_interior_2 = interior_view(src[2])
    @assert size(dest_interior) == size(src_interior_1) "Incompatible array sizes"
    @views map!(f, dest_interior, src_interior_1, src_interior_2)
    halo_exchange!(dest) # Use @views to avoid unnecessary copies
    return dest
end

function Base.map!(f,dest::HaloArray,src::Vararg{ HaloArray,3} ) 
    dest_interior = interior_view(dest)
    src_interior_1 = interior_view(src[1])
    src_interior_2 = interior_view(src[2])
    src_interior_3 = interior_view(src[3])
    @assert size(dest_interior) == size(src_interior_1) "Incompatible array sizes"
    map!(f, dest_interior, src_interior_1, src_interior_2, src_interior_3)
    halo_exchange!(dest) # Use @views to avoid unnecessary copies
    return dest
end


function Base.map!(f, dest::HaloArray, src::Vararg{ HaloArray, N }) where {N}
    dest_interior = interior_view(dest)
    src_interiors = map(src) do s
        interior_view(s)
    end
    @assert all(size(dest_interior) .== map(size, src_interiors)) "Incompatible array sizes"
    map!(f, dest_interior, src_interiors...)
    halo_exchange!(dest) # Use @views to avoid unnecessary copies
    return dest
end

function Base.map(f,src::Vararg{HaloArray,N}) where {N} 
    similar_src = similar(src[1])
    map!(f,similar_src,src...)
    return similar_src
end


function Base.foreach(f, halo::HaloArray{T,N,A,Halo,B,BCondition}) where {T, N,A,Halo,B,BCondition}
    interior = interior_view(halo)
    foreach(f, interior)

end


function fill_from_global_indices!(f,halo::HaloArray)
    h = halo_width(halo)
    local_shape =interior_range(halo)
    #@assert length(local_shape) == ndims(halo)

    for local_I in CartesianIndices(local_shape)
        global_I = local_to_global_index(halo, Tuple(local_I))
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


# 2-argument show, used by Array show, print(obj) and repr(obj), keep it short
function Base.show(io::IO, obj::HaloArray)
    print(io, "HaloArray of size ", size(obj), " (full size: ", full_size(obj), "), halo width: ", halo_width(obj), "\n")
    print(io, "  eltype: ", eltype(obj), "\n")
    print(io, "  topology: ", obj.topology, "\n")
    print(io, "  boundary_condition: ", obj.boundary_condition, "\n")
end

# the 3-argument show used by display(obj) on the REPL
function Base.show(io::IO, mime::MIME"text/plain", obj::HaloArray)
    # Show a more detailed, pretty-printed summary for REPL and text/plain
    println(io, "HaloArray (full size: ", full_size(obj), ", halo width: ", halo_width(obj), ")")
    println(io, "  eltype: ", eltype(obj))
    println(io, "  topology: ", obj.topology)
    println(io, "  boundary_condition: ", obj.boundary_condition)
    # Optionally, show a preview of the interior data
    println(io, "  interior data preview:")
    show(io, mime, interior_view(obj))
end

function global_size(halo::HaloArray)
    N = ndims(halo)
    local_interior = interior_size(halo)  # Tuple with local interior sizes per dimension
    dims = halo.topology.dims             # Tuple with number of processes per dimension
    
    return ntuple(i -> local_interior[i] * dims[i], Val(N))
end

# Helper: validate boundary_condition vs topology (semplificata)
function validate_boundary_condition(topology::CartesianTopology{N}, boundary_condition) where {N}
    for d in 1:N
        left, right = boundary_condition[d]

        # type check
        if !(left isa AbstractBoundaryCondition) || !(right isa AbstractBoundaryCondition)
            error("boundary_condition[$d] must be a tuple of AbstractBoundaryCondition (got $(left), $(right))")
        end

        topo_is_periodic = topology.periodic_boundary_condition[d]
        both_periodic = (left isa Periodic) && (right isa Periodic)
        any_periodic = (left isa Periodic) || (right isa Periodic)

        if topo_is_periodic && !both_periodic
            error("Topology is periodic in dimension $d but boundary_condition[$d] is not (both sides must be Periodic).")
        elseif !topo_is_periodic && any_periodic
            error("Boundary condition in dimension $d uses Periodic but topology is not periodic.")
        end
    end
    return true
end

# Helper: crea NTuple{N,NTuple{2,ArrayLike}} di receive/send buffer (type-stable)
function make_recv_buffers(data::AbstractArray{T,N}, halo::Int) where {T,N}
    ntuple(D -> ntuple(S -> similar(get_recv_view(Side(S), Dim(D), data, halo)), Val(2)), Val(N))
end

function make_send_buffers(data::AbstractArray{T,N}, halo::Int) where {T,N}
    ntuple(D -> ntuple(S -> similar(get_send_view(Side(S), Dim(D), data, halo)), Val(2)), Val(N))
end

