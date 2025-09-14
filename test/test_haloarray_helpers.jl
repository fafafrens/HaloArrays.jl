
using Test
using MPI

# include le definizioni usate nei test
using HaloArrays

# non richiediamo mpiexec per questi test; MPI è usato solo per costanti come COMM_NULL
@testset "HaloArray helpers" begin

    @testset "normalize_boundary_condition" begin
        # single symbol -> same BC in all dims
        bc1 = HaloArrays.normalize_boundary_condition(:repeating, 2)
        @test length(bc1) == 2
@test bc1[1][1] isa HaloArrays.Repeating && bc1[1][2] isa HaloArrays.Repeating
@test bc1[2][1] isa HaloArrays.Repeating && bc1[2][2] isa HaloArrays.Repeating

        # tuple of per-dimension specs (Symbols)
        bc2 = HaloArrays.normalize_boundary_condition((:reflecting, :periodic), 2)
@test bc2[1][1] isa HaloArrays.Reflecting && bc2[1][2] isa HaloArrays.Reflecting
@test bc2[2][1] isa HaloArrays.Periodic && bc2[2][2] isa HaloArrays.Periodic

        # tuple of per-dimension pairs
        bc3 = HaloArrays.normalize_boundary_condition(((:reflecting, :repeating), (:periodic, :periodic)), 2)
@test bc3[1][1] isa HaloArrays.Reflecting && bc3[1][2] isa HaloArrays.Repeating
@test bc3[2][1] isa HaloArrays.Periodic && bc3[2][2] isa HaloArrays.Periodic

        # bad length -> error
        @test_throws ArgumentError HaloArrays.normalize_boundary_condition((:repeating,), 2)
    end

    @testset "uninitialized HaloArray constructor (undef)" begin
        # costruttore parametric: HaloArray{T,N,A,Halo}(undef, Array, bc)
        # crea HaloArray non-inizializzata per Array{T,N} con halo=1

        bc=HaloArrays.normalize_boundary_condition((:repeating, :repeating), 2)
h = HaloArrays.HaloArray{Float64,2,Array{Float64,2},1}(undef, bc)

        @test eltype(h) === Float64
        @test ndims(h) == 2
@test HaloArrays.halo_width(h) == 1

        # topology inattiva: cart_comm dovrebbe essere MPI.COMM_NULL
        @test isdefined(h, :topology)
        @test h.topology.cart_comm == MPI.COMM_NULL

        # boundary_condition normalizzata
        @test length(h.boundary_condition) == 2
@test h.boundary_condition[1][1] isa HaloArrays.Repeating && h.boundary_condition[1][2] isa HaloArrays.Repeating

        # receive/send buffers struttura: NTuple{N,NTuple{2,_}}
        @test length(h.receive_bufs) == ndims(h)
        @test all(length(pair) == 2 for pair in h.receive_bufs)


    end

end


