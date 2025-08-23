using MPI
using Test
using BenchmarkTools

include("cartesian_topology.jl")
include("haloarray.jl")
include("haloarrays.jl")
include("boundary.jl")
include("halo_exchange.jl")

MPI.Init()

@testset "Boundary Condition Test" begin
    comm = MPI.COMM_WORLD
    N = 3
    dims = (10, 10,10)
    halo = 1
    topology = CartesianTopology(comm, N, periodic=(false, false,false))

    # --- Repeating ---
    bc = (:repeating, :repeating,:repeating)

    A = HaloArray(Float64, dims, halo, topology; boundary_condition=bc)
     
    fill_interior(A, 42.0)
    boundary_condition!(A, Side(1), Dim(1), Repeating())
    left_halo = get_recv_view(Side(1), Dim(1), parent(A), halo)
    if topology.neighbors[1][1] == MPI.PROC_NULL
        @test all(left_halo .== 42.0)
    end
    boundary_condition!(A, Side(2), Dim(2), Repeating())
    right_halo = get_recv_view(Side(2), Dim(2), parent(A), halo)
    if topology.neighbors[1][2] == MPI.PROC_NULL
        @test all(right_halo .== 42.0)
    end

    # --- Reflecting ---
    bc = (:reflecting, :reflecting,:reflecting)
    B = HaloArray(Float64, dims, halo, topology; boundary_condition=bc)
    fill_interior(B, 0.0)
    parent(B)[2,:, :] .= 99.0   # bordo sinistro interno
    parent(B)[end-1,:, :] .= 77.0  # bordo destro interno
    boundary_condition!(B,Side(1), Dim(1), Reflecting())
    left_halo_B = get_recv_view(Side(1), Dim(1), parent(B), halo)
     if topology.neighbors[1][1] == MPI.PROC_NULL
        @test all(left_halo_B .== 99.0)
    end
    boundary_condition!(B, Side(2), Dim(1), Reflecting())
    right_halo_B = get_recv_view(Side(2), Dim(1), parent(B), halo)
    if topology.neighbors[1][2] == MPI.PROC_NULL
        @test all(right_halo_B .== 77.0)
    end

    # --- Reflecting ---
    bc = (:antireflecting, :antireflecting,:antireflecting)
    C = HaloArray(Float64, dims, halo, topology; boundary_condition=bc)
    fill_interior(C, 0.0)
    parent(C)[2, :, :] .= 1000.0   # bordo sinistro interno
    parent(C)[end-1, :, :] .= 72.0  # bordo destro interno
    #parent(C)
     # bordo destro interno
     #using BenchmarkTools
    #@benchmark  boundary_condition!($C)
    #@code_warntype boundary_condition!(C)
    boundary_condition!(C)
    
    left_halo_C = get_recv_view(Side(1), Dim(1), C)
    @show left_halo_C
    if topology.neighbors[1][1] == MPI.PROC_NULL
    @test all(left_halo_C .== -1000.0)
    end
    #boundary_condition!(C, Side(2), Dim(1), Reflecting())
    right_halo_C = get_recv_view(Side(2), Dim(1), C)
    @show right_halo_C
    if topology.neighbors[1][2] == MPI.PROC_NULL
        @test all(right_halo_C .== -72.0)
    end

    # --- Repeat tests with halo = 2 ---
    halow = 2

# ...existing code...
    # Repeating
    bc2 = (:repeating, :repeating, :repeating)
    A2 = HaloArray(Float64, dims, halow, topology; boundary_condition=bc2)
    fill_interior(A2, 42.0)
    boundary_condition!(A2, Side(1), Dim(1), Repeating())
    left_halo_A2 = get_recv_view(Side(1), Dim(1), parent(A2), halow)
    if topology.neighbors[1][1] == MPI.PROC_NULL
        @test all(left_halo_A2 .== 42.0)
    end
    boundary_condition!(A2, Side(2), Dim(2), Repeating())
    right_halo_A2 = get_recv_view(Side(2), Dim(2), parent(A2), halow)
    if topology.neighbors[2][2] == MPI.PROC_NULL
        @test all(right_halo_A2 .== 42.0)
    end

    # Reflecting
    bc2 = (:reflecting, :reflecting, :reflecting)
    B2 = HaloArray(Float64, dims, halow, topology; boundary_condition=bc2)
    fill_interior(B2, 0.0)
    # set the two nearest interior slices to the same value so reflection is easy to test
    parent(B2)[halow+1, :, :] .= 11.0
    parent(B2)[halow+2, :, :] .= 10.0
    parent(B2)[end-halow, :, :]   .= 22.0
    parent(B2)[end-halow-1, :, :] .= 20.0
    boundary_condition!(B2, Side(1), Dim(1), Reflecting())
    left_halo_B2 = get_recv_view(Side(1), Dim(1), parent(B2), halow)
    if topology.neighbors[1][1] == MPI.PROC_NULL
        @test all(left_halo_B2[1, :, :] .== 10.0)
        @test all(left_halo_B2[2, :, :] .== 11.0)
    end

    boundary_condition!(B2, Side(2), Dim(1), Reflecting())
    right_halo_B2 = get_recv_view(Side(2), Dim(1), parent(B2), halow)
    if topology.neighbors[1][2] == MPI.PROC_NULL
        @test all(right_halo_B2[1, :, :] .== 22.0)
        @test all(right_halo_B2[2, :, :] .== 20.0)
    end

    # Antireflecting
    bc2 = (:antireflecting, :antireflecting, :antireflecting)
    C2 = HaloArray(Float64, dims, halow, topology; boundary_condition=bc2)
    fill_interior(C2, 0.0)
    parent(C2)[halow+1, :, :] .= 1000.0
    parent(C2)[halow+2, :, :] .= 1000.0
    parent(C2)[end-halow, :, :]   .= 72.0
    parent(C2)[end-halow-1, :, :] .= 72.0
    # apply all per-dim BCs
    boundary_condition!(C2)
    left_halo_C2 = get_recv_view(Side(1), Dim(1), parent(C2), halow)
    if topology.neighbors[1][1] == MPI.PROC_NULL
        @test all(left_halo_C2 .== -1000.0)
    end
    right_halo_C2 = get_recv_view(Side(2), Dim(1), parent(C2), halow)
    if topology.neighbors[1][2] == MPI.PROC_NULL
        @test all(right_halo_C2 .== -72.0)
    end
# ...existing code...


end


MPI.Finalize()