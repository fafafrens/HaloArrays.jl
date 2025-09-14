#@mpi_do manager begin
using MPI
using Test
using HaloArrays
# -- MAIN SCRIPT --


function test_multihaloarray_broadcast()

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

#dims = (2, 2) # 2x2 process grid for example
#topo = CartesianTopology(comm, dims)



local_size = (4, 4)
halo_size = 1
#bd=((Periodic(),Periodic()),(Periodic(),Periodic()))
halo_arr = HaloArray( local_size, halo_size)

    A = similar(halo_arr)
    A.data .=π
    B = similar(halo_arr)
    B.data .=20
    C = similar(halo_arr)
    C.data .=3

   
    mha1 = MultiHaloArray((; a=A, b=B));
    mha2 = MultiHaloArray((; a=A, b=C));


    # Test field-wise addition
    res=similar(mha1)
    res .= mha1 .+ mha2
    @test res isa MultiHaloArray
    res.arrays[:b].data
    @test all(interior_view(res.arrays[:a]) .== 2π)
    @test all(interior_view(res.arrays[:b]) .== 23.0)

    # Test broadcasting with scalar
    res2=similar(mha1)
    res2 = mha1 .+ 1
   
  
    @test all(interior_view(res2.arrays[:a]) .== 1+π)
    @test all(interior_view(res2.arrays[:b]).== 21.0)

    local_size = (7, 4)
    halo_size = 1
#bd=((Periodic(),Periodic()),(Periodic(),Periodic()))
    halo_arr2 = HaloArray( local_size, halo_size)

    # Mismatched dimensions should error
    ntupbad=(; a=A, b=halo_arr2)
    @test_throws DimensionMismatch mha_bad = MultiHaloArray(ntupbad)
    

    A = similar(halo_arr)
    A.data .=π+1
    B = similar(halo_arr)
    B.data .=π/2
 
   
    mha1 = MultiHaloArray((; a=A, b=B));


    res3=similar( mha1)

    res2 = sin.(mha1)

    #res2.=res2 .+ A
    @test res2 isa MultiHaloArray
    @test res2.arrays[:a] isa HaloArray
    @test res2.arrays[:b] isa HaloArray 
    @test all(interior_view(res2.arrays[:a]) .≈ sin(π+1))
    @test all(interior_view(res2.arrays[:b]).==sin(π/2))

    res2323= sin.(mha1 .* 2)
    @test all(interior_view(res2323.arrays[:a]) .≈ sin(2*(π+1)))



end 

function test_haloarray_broadcast()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    local_size = (4, 4)
    halo_size = 1

    A = HaloArray(local_size, halo_size)
    B = similar(A)
    C = similar(A)

    A.data .= 2.0
    B.data .= 3.0
    C.data .= 0.0

    # Binary operation test
    res1 = similar(A)
    res1 .= A .+ B
    @test all(interior_view(res1) .== 5.0)

    # Scalar broadcast
    res2 = similar(A)
    res2 .= A .+ 10
    @test all(interior_view(res2) .== 12.0)

    # Unary broadcast
    res3 = similar(A)
    res3 .= sin.(A)
    @test all(interior_view(res3) .≈ sin.(2.0))

    # Nested broadcast
    res4 = similar(A)
    res4 .= A .+ sin.(B)
    @test all(interior_view(res4) .≈ 2.0 .+ sin.(3.0))

    # Test copy
    bc_expr = Base.Broadcast.broadcasted(+, A, B)
    res5 = copy(bc_expr)
    @test all(interior_view(res5) .== 5.0)

    # Test materialize!
    res6 = similar(A)
    bc_expr = Base.Broadcast.broadcasted(*, A, B)
    Base.Broadcast.materialize!(res6, bc_expr)
    @test all(interior_view(res6) .== 6.0)

    # Dimension mismatch
    A2 = HaloArray((5, 5), halo_size)
    @test_throws DimensionMismatch A .+ A2

    # Type promotion
    Aint = HaloArray(local_size, halo_size)
    Aint.data .= 2
    res7 = A .+ Aint
    @test eltype(res7) == Float64
    @test all(interior_view(res7) .== 4.0)

 
end


@testset "HaloArray Broadcast Tests" begin
    test_haloarray_broadcast()
end

@testset "MultiHaloArray Broadcast Tests" begin
    test_multihaloarray_broadcast()
end

  

