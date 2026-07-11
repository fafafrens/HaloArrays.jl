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

    @test is_active(topology)
    @test prod(topology.dims) == nranks
    @test topology.periodic_boundary_condition == (true, false)
    @test all(0 <= topology.cart_coords[i] < topology.dims[i] for i in 1:2)

    for dim in 1:2, side in 1:2
        @test topology.neighbors[dim][side] == _expected_neighbor(topology, dim, side)
    end

    # The slice/root sub-topologies live inside DimReductionPlan (via Cart_sub):
    # the reduce comm spans the removed dim with sub-rank order matching the
    # Cartesian coordinate (sub-rank 0 = coordinate 0 — the plan's root), and
    # the output's topology covers the kept dims on the coordinate-0 slice.
    bc = ((Periodic(), Periodic()), (Repeating(), Repeating()))
    u = HaloArray(Float64, (2, 3), 1, topology; boundary_condition=bc)
    plan = DimReductionPlan(u, 1)

    @test MPI.Comm_size(plan.reduce_comm) == topology.dims[1]
    @test MPI.Comm_rank(plan.reduce_comm) == topology.cart_coords[1]
    @test plan.is_slice_root == (topology.cart_coords[1] == 0)

    local_value = topology.global_rank + 1
    reduced = MPI.Reduce(local_value, MPI.SUM, 0, plan.reduce_comm)
    if plan.is_slice_root
        expected = sum(
            MPI.Cart_rank(topology.cart_comm, (i, topology.cart_coords[2])) + 1
            for i in 0:(topology.dims[1] - 1)
        )
        @test reduced == expected
    end

    out_topo = parent(plan.output).topology
    if topology.cart_coords[1] == 0
        @test is_active(out_topo)
        @test out_topo.dims == (topology.dims[2],)
        @test out_topo.cart_coords == (topology.cart_coords[2],)
        @test out_topo.periodic_boundary_condition == (false,)
    else
        @test !is_active(out_topo)
        @test out_topo.cart_comm == MPI.COMM_NULL
    end
    free!(plan)

    MPI.Barrier(comm)
end
