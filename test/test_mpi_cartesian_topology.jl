using Test
using MPI
using HaloArrays

function _expected_neighbor(topology, dim, side)
    coord = topology.cart_coords[dim]
    width = topology.dims[dim]
    periodic = topology.periodic_boundary_condition[dim]

    shifted = side == 1 ? coord - 1 : coord + 1
    if shifted < 0 || shifted >= width
        periodic || return MPI.PROC_NULL
        shifted = mod(shifted, width)
    end

    neighbor_coords = ntuple(i -> i == dim ? shifted : topology.cart_coords[i], Val(length(topology.dims)))
    return MPI.Cart_rank(topology.cart_comm, neighbor_coords)
end

@testset "MPI CartesianTopology across ranks" begin
    comm = MPI.COMM_WORLD
    nranks = MPI.Comm_size(comm)
    @test nranks > 1

    topology = CartesianTopology(comm, (0, 0); periodic=(true, false))

    @test isactive(topology)
    @test prod(topology.dims) == nranks
    @test topology.periodic_boundary_condition == (true, false)
    @test all(0 <= topology.cart_coords[i] < topology.dims[i] for i in 1:2)

    for dim in 1:2, side in 1:2
        @test topology.neighbors[dim][side] == _expected_neighbor(topology, dim, side)
    end

    sub_comm, coords, subrank = HaloArrays.subcomm_for_slices(topology, (1,))
    @test coords == topology.cart_coords
    @test MPI.Comm_size(sub_comm) == topology.dims[1]
    @test subrank == topology.cart_coords[1]

    local_value = topology.global_rank + 1
    reduced = MPI.Reduce(local_value, MPI.SUM, 0, sub_comm)
    if subrank == 0
        expected = sum(
            MPI.Cart_rank(topology.cart_comm, (i, topology.cart_coords[2])) + 1
            for i in 0:(topology.dims[1] - 1)
        )
        @test reduced == expected
    end
    MPI.free(sub_comm)

    root_topology = HaloArrays.root_topology_multi(topology, (1,))
    if topology.cart_coords[1] == 0
        @test isactive(root_topology)
        @test root_topology.dims == (topology.dims[2],)
        @test root_topology.cart_coords == (topology.cart_coords[2],)
        @test root_topology.periodic_boundary_condition == (false,)
    else
        @test !isactive(root_topology)
        @test root_topology.cart_comm == MPI.COMM_NULL
    end

    MPI.Barrier(comm)
end
