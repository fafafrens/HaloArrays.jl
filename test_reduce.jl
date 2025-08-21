using Test
using MPI
using LinearAlgebra
MPI.Init()
include("cartesian_topology.jl")
include("haloarray.jl")         # Your HaloArray implementation
include("halo_exchange.jl")     # Your halo exchange routines
include("boundary.jl")    
include("reduction.jl")      # Boundary conditions


function test_reduction()
    h = 1
    local_shape = (4, 4)
    halo = HaloArray(local_shape, h)
    
    # Fill interior with rank-specific values
    rank = MPI.Comm_rank(get_comm(halo))
    interior = interior_view(halo)
    fill!(halo.data, -1)  # Reset full buffer including halos
    interior .= rank + 1  # Interior elements = rank + 1
    
    # Perform MPI halo exchange if needed (optional, depending on your use)
    # halo_exchange!(halo)

    # Test sum reduction over the HaloArray interior
    expected_sum = (rank + 1.) * prod(local_shape)
    expected_prod = (rank + 1.) ^ prod(local_shape)
    global_sum = reduce(+, halo)
    global_prod = reduce(*, halo)


    global_sum_mpi =MPI.Allreduce(expected_sum, MPI.SUM, get_comm(halo))

    global_prod_mpi =  MPI.Allreduce(expected_prod, MPI.PROD, get_comm(halo))
    
    @test isapprox(global_sum, global_sum_mpi, atol=1e-14)
    @test isapprox(global_prod, global_prod_mpi, atol=1e-14)

    
    

    println("✅ Reduction tests passed on rank $rank")
end

test_reduction()

MPI.Finalize()