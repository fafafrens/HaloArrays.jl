using MPI
using Test
using HaloArrays

MPI.Init()

function test_gather_haloarray_1D()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    # 1D topology for simplicity in this test
    #topo = CartesianTopology((size,), comm)

    # Local domain size (interior)
    local_shape = (4,)
    h = 1  # halo width

    # Global logical coordinates
    #coords = topo.cart_coords[1]

    # Create a local array including halo
    full_shape = (local_shape[1] + 2h,)
    data = fill(rank + 1, full_shape)  # Use rank+1 for easy testing

    # Construct the HaloArray manually
    halo = HaloArray(Float64, local_shape, h; boundary_condition = ((Periodic(),Periodic()),))
    parent(halo).=data
    gathered = gather_haloarray(halo; root=0)

    if rank == 0
        @test size(gathered) == (local_shape[1] * nprocs,)
        expected = vcat([fill(r + 1, local_shape[1]) for r in 0:nprocs-1]...)
        @test gathered == expected
        println("Gather test passed: ", gathered)
    end
end


function test_gather_haloarray_2D()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    # 2D domain locale: 4x3 (interno)
    local_shape = (4, 3)
    dims=(2,2)
    h = 1
    
    # Crea l'array con halo
    full_shape = (local_shape[1] + 2h, local_shape[2] + 2h)
    data = fill(rank + 1, full_shape)

    # Crea HaloArray e inizializza
    topo=CartesianTopology(comm, dims; periodic = (true,true))
    bc = ((Periodic(),Periodic()), (Periodic(),Periodic()))
    halo = HaloArray(Float64, local_shape, h,topo; boundary_condition = bc)
    interior_view(halo) .= rank + 1  # Inizializza solo l'interno

    # Raccogli sul root
    gathered = gather_haloarray(halo; root=0)

    if rank == 0
        # Verifica dimensioni
        @test size(gathered) == (local_shape[1] * dims[1], local_shape[2] * dims[2])
        
        # Crea array atteso per griglia 2x2 di processi
        expected = zeros(local_shape[1] * dims[1], local_shape[2] * dims[2])
        for r in 0:nprocs-1
            coords_r = MPI.Cart_coords(topo.cart_comm, r)
            
            # Calcola gli indici usando le coordinate
            i_start = coords_r[1] * local_shape[1] + 1
            i_end = (coords_r[1] + 1) * local_shape[1]
            j_start = coords_r[2] * local_shape[2] + 1
            j_end = (coords_r[2] + 1) * local_shape[2]
            
            expected[i_start:i_end, j_start:j_end] .= r + 1
        end

        # Debug: mostra la mappatura rank -> coordinate
        println("Rank to coordinates mapping:")
        for r in 0:nprocs-1
            coords = MPI.Cart_coords(topo.cart_comm, r)
            println("Rank $r -> coords $coords")
        end

        @test gathered == expected
        println("2D Gather test passed")
        println("Gathered array: ", gathered)
        println("Expected array: ", expected)
        println("Gathered array size: ", size(gathered))
        println("Expected array size: ", size(expected))
    end 
end

function test_gather_haloarray_3D()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    # 3D domain locale: 4x3x2 (interno)
    local_shape = (4, 3, 2)
    dims=(2,2,1)
    h = 1
    
    # Crea l'array con halo
    full_shape = (local_shape[1] + 2h, local_shape[2] + 2h, local_shape[3] + 2h)
    data = fill(rank + 1, full_shape)

    # Crea HaloArray e inizializza
    topo=CartesianTopology(comm, dims; periodic = (true,true,true))
    bc = ((Periodic(),Periodic()), (Periodic(),Periodic()), (Periodic(),Periodic()))
    halo = HaloArray(Float64, local_shape, h,topo; boundary_condition = bc)
    interior_view(halo) .= rank + 1  # Inizializza solo l'interno

    # Raccogli sul root
    gathered = gather_haloarray(halo; root=0)

    if rank == 0
        # Verifica dimensioni
        @test size(gathered) == (local_shape[1] * dims[1], local_shape[2]* dims[2], local_shape[3]* dims[3])
        
        # Crea array atteso: ogni processo contribuisce con un blocco
        expected = zeros(local_shape[1] * dims[1], local_shape[2] * dims[2], local_shape[3] * dims[3])
        for r in 0:nprocs-1
            coords_r = MPI.Cart_coords(topo.cart_comm, r)
            
            i_start = coords_r[1] * local_shape[1] + 1
            i_end = (coords_r[1] + 1) * local_shape[1]
            j_start = coords_r[2] * local_shape[2] + 1
            j_end = (coords_r[2] + 1) * local_shape[2]
            k_start = coords_r[3] * local_shape[3] + 1
            k_end = (coords_r[3] + 1) * local_shape[3]
            
            expected[i_start:i_end, j_start:j_end, k_start:k_end] .= r + 1
        end

        # Debug: mostra la mappatura rank -> coordinate
        println("Rank to coordinates mapping:")
        for r in 0:nprocs-1
            coords = MPI.Cart_coords(topo.cart_comm, r)
            println("Rank $r -> coords $coords")
        end
        
        @test gathered == expected
        println("3D Gather test passed")
        println("Gathered array: ", gathered)
        println("Expected array: ", expected)
        println("Gathered array size: ", size(gathered))
        println("Expected array size: ", size(expected))
    end
end

# Esegui i test
@testset "HaloArray Gather Tests" begin
    test_gather_haloarray_1D()
    test_gather_haloarray_2D()
    test_gather_haloarray_3D()
end

test_gather_haloarray_1D()
test_gather_haloarray_2D()
test_gather_haloarray_3D()

MPI.Barrier(MPI.COMM_WORLD)
MPI.Finalize()