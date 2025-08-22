using MPI, BenchmarkTools, Test
include("cartesian_topology.jl") 
include("haloarray.jl")
include("haloarrays.jl")
include("interior_broadcast.jl")
include("halo_exchange.jl") 

function test_halo_exchange_correctness()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    dims = (0, 0)  # Let MPI decide
    topo = CartesianTopology(comm, dims)


    halo = 1
    local_shape = (100, 100)  # Inner shape (without halos)
    bd=((Periodic(),Periodic()),(Periodic(),Periodic()))  # periodic boundary conditions
    # Create and fill three identical halo arrays
    halo1 = HaloArray(Float64, local_shape, halo, topo, bd)
    halo2 = copy(halo1)
    halo3 = copy(halo1)
    halo4 = copy(halo1)
    halo5 = copy(halo1)

    fill!(halo1.data, rank + 1.0)
    fill!(halo2.data, rank + 1.0)
    fill!(halo3.data, rank + 1.0)
    fill!(halo4.data, rank + 1.0)
    fill!(halo5.data, rank + 1.0)

    # Ensure all halos are initialized with the same data
    is_equal12 = all(halo1.data .== halo2.data)
    is_equal13 = all(halo1.data .== halo3.data)
    is_equal23 = all(halo2.data .== halo3.data)
    is_equal14 = all(halo1.data .== halo4.data)
    is_equal24 = all(halo2.data .== halo4.data)
    is_equal34 = all(halo3.data .== halo4.data)
    is_equal45 = all(halo4.data .== halo5.data)
    is_equal25 = all(halo2.data .== halo5.data)
    is_equal35 = all(halo3.data .== halo5.data)
    is_equal15 = all(halo1.data .== halo5.data)
    

    for nrank in 0:topo.nprocs-1
        if nrank == topo.global_rank
        println("Halo exchange correctness:")
        println("halo1 vs halo2: $is_equal12 at $nrank")
        println("halo1 vs halo3: $is_equal13 at $nrank")
        println("halo2 vs halo3: $is_equal23 at $nrank")
        println("halo1 vs halo4: $is_equal14 at $nrank")
        println("halo2 vs halo4: $is_equal24 at $nrank")
        println("halo3 vs halo4: $is_equal34 at $nrank")
        println("halo4 vs halo5: $is_equal45 at $nrank")
        println("halo2 vs halo5: $is_equal25 at $nrank")
        println("halo3 vs halo5: $is_equal35 at $nrank")
        println("halo1 vs halo5: $is_equal15 at $nrank")
    end
    MPI.Barrier(topo.comm)
    end



    # Perform halo exchanges
    halo_exchange_async_unsafe!(halo1)
    halo_exchange_async!(halo2)
    halo_exchange_waitall!(halo3)
    halo_exchange_waitall_unsafe!(halo4)
    halo_exchange_async_unsafe!(halo5)

    # Compare halo regions and interiors
    is_equal12 = all(halo1.data .== halo2.data)
    is_equal13 = all(halo1.data .== halo3.data)
    is_equal23 = all(halo2.data .== halo3.data)
    is_equal14 = all(halo1.data .== halo4.data)
    is_equal24 = all(halo2.data .== halo4.data)
    is_equal34 = all(halo3.data .== halo4.data)
    is_equal45 = all(halo4.data .== halo5.data)
    is_equal25 = all(halo2.data .== halo5.data)
    is_equal35 = all(halo3.data .== halo5.data)
    is_equal15 = all(halo1.data .== halo5.data)

    for nrank in 0:topo.nprocs-1
        if nrank == topo.global_rank
        println("Halo exchange correctness:")
        println("halo1 vs halo2: $is_equal12 at $nrank")
        println("halo1 vs halo3: $is_equal13 at $nrank")
        println("halo2 vs halo3: $is_equal23 at $nrank")
        println("halo1 vs halo4: $is_equal14 at $nrank")
        println("halo2 vs halo4: $is_equal24 at $nrank")
        println("halo3 vs halo4: $is_equal34 at $nrank")
        println("halo4 vs halo5: $is_equal45 at $nrank")
        println("halo2 vs halo5: $is_equal25 at $nrank")
        println("halo3 vs halo5: $is_equal35 at $nrank")
        println("halo1 vs halo5: $is_equal15 at $nrank")
        println("%%%%%%%%%%%%%%%%%%%%%%")
        println("Rank $rank after halo exchange:")
        #@show(halo1.data)
        #@show(halo2.data)
        #@show(halo3.data)
        #@show(halo4.data)
    end
    MPI.Barrier(topo.comm)
    end
    # Use @test to check correctness
    @test is_equal12
    @test is_equal13
    @test is_equal23
    @test is_equal14
    @test is_equal24
    @test is_equal34
    @test is_equal45
    @test is_equal25
    @test is_equal35
    @test is_equal15
    # Print final results

    if rank == 0
        println("✅ All halo exchange implementations produce the same result.")
    end

    MPI.Barrier(comm)
    MPI.Finalize()
end

test_halo_exchange_correctness()



