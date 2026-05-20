using Test
using MPI
using HaloArrays

@testset "MPI gather_haloarray" begin
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    @test nranks > 1

    topology = CartesianTopology(comm, (0, 0); periodic=(true, true))
    owned_dims = (2, 3)
    halo = 1
    ha = HaloArray(
        Int,
        owned_dims,
        halo,
        topology;
        boundary_condition=((Periodic(), Periodic()), (Periodic(), Periodic())),
    )

    fill!(parent(ha), -1)
    interior = interior_view(ha)
    for i in 1:owned_dims[1], j in 1:owned_dims[2]
        interior[i, j] = 1000 * rank + 10 * i + j
    end

    gathered = gather_haloarray(ha; root=0)

    if MPI.Comm_rank(topology.cart_comm) == 0
        expected = zeros(Int, owned_dims .* topology.dims)
        for source_rank in 0:(nranks - 1)
            coords = MPI.Cart_coords(topology.cart_comm, source_rank)
            rows = (coords[1] * owned_dims[1] + 1):((coords[1] + 1) * owned_dims[1])
            cols = (coords[2] * owned_dims[2] + 1):((coords[2] + 1) * owned_dims[2])
            expected[rows, cols] .= [1000 * source_rank + 10 * i + j for i in 1:owned_dims[1], j in 1:owned_dims[2]]
        end

        @test gathered == expected
    else
        @test gathered === nothing
    end

    MPI.Barrier(comm)
end
