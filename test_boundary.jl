using MPI
using Test
using BenchmarkTools

include("cartesian_topology.jl")
include("haloarray.jl")
include("boundary.jl")
include("halo_exchange.jl")

MPI.Init()

#@testset "Boundary Condition Test" begin
    comm = MPI.COMM_WORLD
    N = 3
    dims = (10, 10,10)
    halo = 1
    topology = CartesianTopology(comm, (1,2,2), periodic=(false, false,false))

    # --- Repeating ---
    bc = (:repeating, :repeating,:repeating)

    A = HaloArray(Float64, dims, halo; boundary_condition=bc)
     
    fill_interior(A, 42.0)
    boundary_condition!(A, Side(1), Dim(1), Repeating())
    left_halo = get_recv_view(Side(1), Dim(1), parent(A), halo)
    @test all(left_halo .== 42.0)
    boundary_condition!(A, Side(2), Dim(2), Repeating())
    right_halo = get_recv_view(Side(2), Dim(2), parent(A), halo)
    @test all(right_halo .== 42.0)

    # --- Reflecting ---
    bc = (:reflecting, :reflecting,:reflecting)
    B = HaloArray(Float64, dims, halo, topology; boundary_condition=bc)
    fill_interior(B, 0.0)
    parent(B)[2,:, :] .= 99.0   # bordo sinistro interno
    parent(B)[end-1,:, :] .= 77.0  # bordo destro interno
    boundary_condition!(B,Side(1), Dim(1), Reflecting())
    left_halo_B = get_recv_view(Side(1), Dim(1), parent(B), halo)
    @test all(left_halo_B .== 99.0)
    boundary_condition!(B, Side(2), Dim(1), Reflecting())
    right_halo_B = get_recv_view(Side(2), Dim(1), parent(B), halo)
    @test all(right_halo_B .== 77.0)

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
    @test all(left_halo_C .== .-1000.0)
    #boundary_condition!(C, Side(2), Dim(1), Reflecting())
    right_halo_C = get_recv_view(Side(2), Dim(1), C)
    @test all(right_halo_C .== .-72.0)
#end


MPI.Finalize()