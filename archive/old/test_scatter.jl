using MPI
using Test
include("cartesian_topology.jl")
include("haloarray.jl")
include("boundary.jl")
include("gather.jl")
include("scatter.jl")
MPI.Init()

function test_scatter_invalid_dimensions()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    dims = (2,2)
    
    # Crea una topologia cartesiana
    topo = CartesianTopology(comm, dims; periodic=(true,true))
    
    if rank == 0
        # Crea un array globale con dimensioni non divisibili
        global_array = zeros(5, 6)  # 5 non è divisibile per 2
        println("\nTest scatter con dimensioni non valide:")
        println("global_array size = $(size(global_array))")
        println("dims = $dims")
    else
        global_array = nothing
    end
    
    try
        halo = scatter_haloarray(global_array, comm, topo, 1)
        if rank == 0
            error("Il test dovrebbe fallire con dimensioni non valide")
        end
    catch e
        if rank == 0
            println("Test dimensioni non valide passato: errore catturato correttamente")
            println("Errore: ", e)
        end
    end
end


function test_scatter_haloarray()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    # Setup per test 2D
    dims = (2,2)
    local_shape = (4,3)
    h = 1
    
    # Crea topologia cartesiana
    topo = CartesianTopology(comm, dims; periodic=(true,true))
    
    if rank == 0
        # Crea array globale da distribuire
        global_shape = (local_shape[1] * dims[1], local_shape[2] * dims[2])
        global_array = zeros(global_shape)
        
        # Riempi con valori di test
        for i in 1:global_shape[1], j in 1:global_shape[2]
            global_array[i,j] = i + j
        end
        halo = scatter_haloarray(global_array, comm, topo, h)
    end 
    
   
    # Scatter l'array globale
    #halo = scatter_haloarray(global_array, comm, topo, h)
    
    # Verifica che ogni processo abbia ricevuto la parte corretta
    coords = MPI.Cart_coords(comm, rank)
    i_start = coords[1] * local_shape[1] + 1
    j_start = coords[2] * local_shape[2] + 1
    
    if rank == 0
        println("Scattering completed. Verifying data...")
    end
    
    # Raccogli tutto di nuovo per verifica
    gathered = gather_haloarray(halo; root=0)
    
    if rank == 0
        @test gathered == global_array
        println("Scatter-gather test passed!")
    end
end

#@testset "HaloArray Scatter Tests" begin
    test_scatter_haloarray()
    #test_scatter_invalid_dimensions()
    # ...altri test...
#end

MPI.Finalize()