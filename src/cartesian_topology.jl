"""
    CartesianTopology{N,C}

Represents an MPI Cartesian topology of dimension `N`.

Fields:
- `nprocs::Int` : number of processes in the reference communicator.
- `dims::NTuple{N,Int}` : grid partition (result of MPI.Dims_create).
- `global_rank::Int` : rank in the original communicator.
- `cart_coords::NTuple{N,Int}` : Cartesian coordinates of this process in `cart_comm`.
- `neighbors::NTuple{N,NTuple{2,Int}}` : for each dimension, `(left, right)` neighbor ranks.
- `comm::C` : reference communicator (e.g. MPI.COMM_WORLD).
- `cart_comm::C` : Cartesian communicator created by MPI.Cart_create.
- `periodic_boundary_condition::NTuple{N,Bool}` : periodicity flags per dimension.
- `active::Bool` : true if the topology is active on this rank.

`C` is the communicator type, typically `MPI.Comm` when MPI is loaded.
"""
struct CartesianTopology{N,C} <: AbstractCartesianTopology{N}
    nprocs::Int
    dims::NTuple{N,Int}
    global_rank::Int
    cart_coords::NTuple{N,Int}
    neighbors::NTuple{N,NTuple{2,Int}}
    comm::C
    cart_comm::C
    periodic_boundary_condition::NTuple{N,Bool}
    active::Bool
end

"""
    is_active(cart::CartesianTopology)

Return `true` if the topology is active on this process.
"""
@inline Base.ndims(::AbstractCartesianTopology{N}) where {N} = N
is_active(cart::CartesianTopology) = cart.active

"""
    is_root(cart::CartesianTopology; root=0)

Return `true` on the root rank (default 0) of an active topology.
"""
is_root(cart::CartesianTopology; root::Integer=0) =
    is_active(cart) && cart.global_rank == root

"""
    coords_to_color_multi(coords, dims, dims_to_remove) -> Int

Compute an integer color for MPI.Comm_split by compressing coordinates
that are NOT in `dims_to_remove`.
"""
function coords_to_color_multi(coords::NTuple{N,Int}, dims::NTuple{N,Int},
        dims_to_remove) where {N}
    tuple_dims_to_remove = Tuple(dims_to_remove)
    rem = (i for i in 1:N if !(i in tuple_dims_to_remove))
    color = 0
    mul = 1
    for i in rem
        color += coords[i] * mul
        mul *= dims[i]
    end
    return color
end
