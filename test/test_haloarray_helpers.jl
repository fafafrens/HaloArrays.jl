
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

        @test left_face_range(ha, 1) == (1:1, 2:6)
        @test internal_face_range(ha) == (2:4, 2:5)
        @test right_face_range(ha, 1) == (5:5, 2:6)
        @test face_offset(ha, 1) == CartesianIndex(1, 0)

        dim2_ranges = FaceRanges(ha, Dim(2))
        @test collect(get_left_face(dim2_ranges)) == collect(CartesianIndices((2:5, 1:1)))
        @test collect(get_internal_face(dim2_ranges)) == collect(CartesianIndices((2:4, 2:5)))
        @test collect(get_right_face(dim2_ranges)) == collect(CartesianIndices((2:5, 6:6)))
        @test face_offset(ha, Dim(2)) == CartesianIndex(0, 1)

        one_cell = LocalHaloArray(Int, (1,), 1; boundary_condition=:repeating)
        one_cell_ranges = FaceRanges(one_cell, 1)
        @test collect(get_left_face(one_cell_ranges)) == [CartesianIndex(1)]
        @test isempty(get_internal_face(one_cell_ranges))
        @test collect(get_right_face(one_cell_ranges)) == [CartesianIndex(2)]

        range_struct = HaloArrays.FaceRanges(ha, 1)
        @test collect(HaloArrays.get_left_face(range_struct)) == collect(CartesianIndices((1:1, 2:6)))
        @test collect(HaloArrays.get_internal_face(range_struct)) == collect(CartesianIndices((2:4, 2:5)))
        @test collect(HaloArrays.get_right_face(range_struct)) == collect(CartesianIndices((5:5, 2:6)))
        @test HaloArrays.get_unit_vector(range_struct) == CartesianIndex(1, 0)
    end

    @testset "face ranges support owned-cell update" begin
        u = LocalHaloArray(Int, (4,), 1; boundary_condition=:repeating)
        du = similar(u)

        parent(u) .= [100, 1, 2, 3, 4, 200]
        fill!(parent(du), 0)

        ranges = FaceRanges(u, 1)
        offset = get_unit_vector(ranges)

        for IL in get_left_face(ranges)
            IR = IL + offset
            parent(du)[IR] += parent(u)[IR] - parent(u)[IL]
        end

        for IL in get_internal_face(ranges)
            IR = IL + offset
            flux = parent(u)[IR] - parent(u)[IL]
            parent(du)[IL] -= flux
            parent(du)[IR] += flux
        end

        for IL in get_right_face(ranges)
            IR = IL + offset
            parent(du)[IL] -= parent(u)[IR] - parent(u)[IL]
        end

        @test collect(interior_view(du)) == [-100, 0, 0, -195]
        @test parent(du)[1] == 0
        @test parent(du)[end] == 0
    end

end
