using Test
using MPI
using HaloArrays

function _fill_rank_pattern!(ha, rank)
    for i in eachindex(ha)
        ha[i] = 100 * rank + i
    end
    return ha
end

function _check_periodic_1d_halo_exchange!(exchange!)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    @test nranks > 1

    topology = CartesianTopology(comm, (0,); periodic=(true,))
    halo_width_cells = 2
    local_cells = 5
    ha = HaloArray(
        Int,
        (local_cells,),
        halo_width_cells,
        topology;
        boundary_condition=((Periodic(), Periodic()),),
    )

    _fill_rank_pattern!(ha, rank)
    MPI.Barrier(comm)
    exchange!(ha)
    MPI.Barrier(comm)

    left_rank = topology.neighbors[1][1]
    right_rank = topology.neighbors[1][2]
    @test left_rank != MPI.PROC_NULL
    @test right_rank != MPI.PROC_NULL

    left_halo = collect(parent(ha)[1:halo_width_cells])
    right_halo = collect(parent(ha)[(halo_width_cells + local_cells + 1):(2 * halo_width_cells + local_cells)])
    interior = collect(interior_view(ha))

    expected_left = [100 * left_rank + local_cells - halo_width_cells + i for i in 1:halo_width_cells]
    expected_right = [100 * right_rank + i for i in 1:halo_width_cells]
    expected_interior = [100 * rank + i for i in 1:local_cells]

    @test left_halo == expected_left
    @test right_halo == expected_right
    @test interior == expected_interior

    return nothing
end

@testset "MPI halo exchange across ranks" begin
    _check_periodic_1d_halo_exchange!(halo_exchange_waitall!)
    _check_periodic_1d_halo_exchange!(halo_exchange_async!)
end
