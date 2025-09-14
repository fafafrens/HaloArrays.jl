using Test
using MPI
using LinearAlgebra
using HaloArrays

@testset "MultiHaloArray Reduction Tests" begin
# Setup
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

local_size = (4, 4)
halo_size = 1

# Construct HaloArrays with known content
ha1 = HaloArray(local_size, halo_size)
ha2 = HaloArray(local_size, halo_size)
ha1.data .= rank + 1.
ha2.data .= 2. * (rank + 1.)

# --- Test mapreduce on HaloArray
@test mapreduce(x -> x^2, +, ha1) == MPI.Allreduce(sum(x^2 for x in interior_view(ha1)), +, comm)
@test mapreduce((x, y) -> x * y, +, ha1, ha2) ≈ MPI.Allreduce(sum(x * y for (x, y) in zip(interior_view(ha1), interior_view(ha2))), +, comm)


# --- Test all and any on HaloArray
@test all(x -> x > 0, ha1)
@test !any(x -> x < 0, ha1)

# Now create a MultiHaloArray
mha = MultiHaloArray((; a=ha1, b=ha2))

# --- Test mapreduce on MultiHaloArray
# function: sum of squares, across all fields
result = mapreduce(x -> x^2, +, mha)
expected = mapreduce(x -> x^2, +, ha1) + mapreduce(x -> x^2, +, ha2)
@test result == expected

# --- Test mapfoldl/r on MultiHaloArray
@test mapfoldl(abs2, +, mha) == result
@test mapfoldr(abs2, +, mha) == result

# --- Test all / any on MultiHaloArray
@test all(x -> x > 0, mha)
@test !any(x -> x < 0, mha)

# Custom test to check false condition
if rank == 0
    mha.arrays[:a].data[2, 2] = -1
end


MPI.Barrier(comm)

@test !all(x -> x > 0, mha)
@test any(x -> x < 0, mha)

MPI.Barrier(comm)
MPI.Finalize()
end # End of testset