
using Test
using MPI

# include le definizioni usate nei test
using HaloArrays

struct CustomBoundaryForTest <: HaloArrays.AbstractBoundaryCondition end

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
        @test_throws ArgumentError HaloArrays.normalize_boundary_condition(:custom, 1)
        @test HaloArrays.to_bc(CustomBoundaryForTest) isa CustomBoundaryForTest
        @test HaloArrays.to_bc(CustomBoundaryForTest()) isa CustomBoundaryForTest
        @test !isdefined(HaloArrays, :register_bc)
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

    @testset "owned face ranges" begin
        ha = LocalHaloArray(Int, (4, 5), 1; boundary_condition=:repeating)

        @test lower_owned_face_range(ha, 1) == (2:2, 2:6)
        @test internal_owned_face_left_range(ha, 1) == (2:4, 2:6)
        @test upper_owned_face_range(ha, 1) == (5:5, 2:6)
        @test face_offset(ha, 1) == CartesianIndex(1, 0)

        dim2_ranges = owned_face_ranges(ha, Dim(2))
        @test dim2_ranges.lower_owned == (2:5, 2:2)
        @test dim2_ranges.internal_left == (2:5, 2:5)
        @test dim2_ranges.upper_owned == (2:5, 6:6)
        @test face_offset(ha, Dim(2)) == CartesianIndex(0, 1)

        one_cell = LocalHaloArray(Int, (1,), 1; boundary_condition=:repeating)
        one_cell_ranges = owned_face_ranges(one_cell, 1)
        @test collect(CartesianIndices(one_cell_ranges.lower_owned)) == [CartesianIndex(2)]
        @test isempty(CartesianIndices(one_cell_ranges.internal_left))
        @test collect(CartesianIndices(one_cell_ranges.upper_owned)) == [CartesianIndex(2)]
    end

    @testset "owned face update helper" begin
        u = LocalHaloArray(Int, (4,), 1; boundary_condition=:repeating)
        du = similar(u)

        parent(u) .= [100, 1, 2, 3, 4, 200]
        fill!(parent(du), 0)

        foreach_owned_face!((ul, ur) -> ur - ul, du, u, 1)

        @test collect(interior_view(du)) == [-100, 0, 0, -195]
        @test parent(du)[1] == 0
        @test parent(du)[end] == 0

        fill!(parent(du), 0)
        foreach_owned_face!((ul, ur) -> ur - ul, du, u, Dim(1))
        @test collect(interior_view(du)) == [-100, 0, 0, -195]

        bad_size = LocalHaloArray(Int, (5,), 1; boundary_condition=:repeating)
        bad_halo = LocalHaloArray(Int, (4,), 2; boundary_condition=:repeating)
        @test_throws DimensionMismatch foreach_owned_face!((ul, ur) -> ur - ul, bad_size, u, 1)
        @test_throws DimensionMismatch foreach_owned_face!((ul, ur) -> ur - ul, bad_halo, u, 1)
        @test_throws ArgumentError foreach_owned_face!((ul, ur) -> ur - ul, du, u, 2)
    end

end
