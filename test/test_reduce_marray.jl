using Test
using MPI
using LinearAlgebra
using HaloArrays

@testset "MultiHaloArray Reduction Tests" begin
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    owned_dims = (4, 4)
    halo_size = 1

    ha1 = HaloArray(owned_dims, halo_size)
    ha2 = HaloArray(owned_dims, halo_size)
    ha1.data .= rank + 1.0
    ha2.data .= 2.0 * (rank + 1.0)

    @test mapreduce(x -> x^2, +, ha1) == MPI.Allreduce(sum(x^2 for x in interior_view(ha1)), +, comm)
    @test mapreduce((x, y) -> x * y, +, ha1, ha2) ≈
          MPI.Allreduce(sum(x * y for (x, y) in zip(interior_view(ha1), interior_view(ha2))), +, comm)
    @test all(x -> x > 0, ha1)
    @test !any(x -> x < 0, ha1)

    mha = MultiHaloArray((; a=ha1, b=ha2))

    result = mapreduce(x -> x^2, +, mha)
    expected = mapreduce(x -> x^2, +, ha1) + mapreduce(x -> x^2, +, ha2)
    @test result == expected

    @test mapfoldl(abs2, +, mha) == result
    @test mapfoldr(abs2, +, mha) == result

    @test all(x -> x > 0, mha)
    @test !any(x -> x < 0, mha)

    if rank == 0
        mha.arrays[:a].data[2, 2] = -1
    end

    MPI.Barrier(comm)

    @test !all(x -> x > 0, mha)
    @test any(x -> x < 0, mha)

    MPI.Barrier(comm)
    MPI.Barrier(comm)
end
