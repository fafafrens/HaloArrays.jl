using Test
using MPI
using HaloArrays

if !MPI.Initialized()
    MPI.Init()
end

struct CustomBoundaryForTest <: HaloArrays.AbstractBoundaryCondition end

# These tests run under one process; MPI is only needed for constants and
# `COMM_SELF` topologies.
@testset "HaloArray helpers" begin

    @testset "normalize_boundary_condition" begin
        bc1 = HaloArrays.normalize_boundary_condition(:repeating, 2)
        @test length(bc1) == 2
        @test bc1[1][1] isa HaloArrays.Repeating && bc1[1][2] isa HaloArrays.Repeating
        @test bc1[2][1] isa HaloArrays.Repeating && bc1[2][2] isa HaloArrays.Repeating

        bc2 = HaloArrays.normalize_boundary_condition((:reflecting, :periodic), 2)
        @test bc2[1][1] isa HaloArrays.Reflecting && bc2[1][2] isa HaloArrays.Reflecting
        @test bc2[2][1] isa HaloArrays.Periodic && bc2[2][2] isa HaloArrays.Periodic

        bc3 = HaloArrays.normalize_boundary_condition(((:reflecting, :repeating), (:periodic, :periodic)), 2)
        @test bc3[1][1] isa HaloArrays.Reflecting && bc3[1][2] isa HaloArrays.Repeating
        @test bc3[2][1] isa HaloArrays.Periodic && bc3[2][2] isa HaloArrays.Periodic

        @test_throws ArgumentError HaloArrays.normalize_boundary_condition((:repeating,), 2)
        @test_throws ArgumentError HaloArrays.normalize_boundary_condition(:custom, 1)
        @test HaloArrays.to_bc(CustomBoundaryForTest) isa CustomBoundaryForTest
        @test HaloArrays.to_bc(CustomBoundaryForTest()) isa CustomBoundaryForTest
        @test !isdefined(HaloArrays, :register_bc)
    end

    @testset "uninitialized HaloArray constructor (undef)" begin
        bc = HaloArrays.normalize_boundary_condition((:repeating, :repeating), 2)
        h = HaloArrays.HaloArray{Float64,2,Array{Float64,2},1}(undef, bc)

        @test eltype(h) === Float64
        @test ndims(h) == 2
        @test HaloArrays.halo_width(h) == 1

        @test isdefined(h, :topology)
        @test h.topology.cart_comm == MPI.COMM_NULL

        @test length(h.boundary_condition) == 2
        @test h.boundary_condition[1][1] isa HaloArrays.Repeating &&
              h.boundary_condition[1][2] isa HaloArrays.Repeating

        @test length(h.receive_bufs) == ndims(h)
        @test all(length(pair) == 2 for pair in h.receive_bufs)
    end

    @testset "owned face ranges" begin
        ha = LocalHaloArray(Int, (4, 5), 1; boundary_condition=:repeating)

        @test left_face_range(ha, 1) == (1:1, 2:6)
        @test internal_face_range(ha) == (2:4, 2:5)
        @test right_face_range(ha, 1) == (5:5, 2:6)
        @test face_offset(ha, 1) == CartesianIndex(1, 0)

        dim2_ranges = FaceRanges(ha)
        @test collect(get_left_face(dim2_ranges, Dim(2))) == collect(CartesianIndices((2:5, 1:1)))
        @test collect(get_internal_face(dim2_ranges)) == collect(CartesianIndices((2:4, 2:5)))
        @test collect(get_right_face(dim2_ranges, Dim(2))) == collect(CartesianIndices((2:5, 6:6)))
        @test face_offset(ha, Dim(2)) == CartesianIndex(0, 1)
        @test get_unit_vector(dim2_ranges, Dim(2)) == CartesianIndex(0, 1)

        one_cell = LocalHaloArray(Int, (1,), 1; boundary_condition=:repeating)
        one_cell_ranges = FaceRanges(one_cell)
        @test collect(get_left_face(one_cell_ranges, 1)) == [CartesianIndex(1)]
        @test isempty(get_internal_face(one_cell_ranges))
        @test collect(get_right_face(one_cell_ranges, 1)) == [CartesianIndex(2)]

        range_struct = HaloArrays.FaceRanges(ha)
        @test collect(HaloArrays.get_left_face(range_struct, 1)) == collect(CartesianIndices((1:1, 2:6)))
        @test collect(HaloArrays.get_internal_face(range_struct)) == collect(CartesianIndices((2:4, 2:5)))
        @test collect(HaloArrays.get_right_face(range_struct, 1)) == collect(CartesianIndices((5:5, 2:6)))
        @test HaloArrays.get_unit_vector(range_struct, 1) == CartesianIndex(1, 0)

        left_pairs = @inferred get_left_face_pairs(range_struct, Dim(1))
        internal_pairs = @inferred get_internal_face_pairs(range_struct, Dim(1))
        right_pairs = @inferred get_right_face_pairs(range_struct, Dim(1))
        @test first(left_pairs) == (CartesianIndex(1, 2), CartesianIndex(2, 2))
        @test collect(left_pairs) ==
              [(I, I + CartesianIndex(1, 0)) for I in get_left_face(range_struct, 1)]
        @test collect(internal_pairs) ==
              [(I, I + CartesianIndex(1, 0)) for I in get_internal_face(range_struct)]
        @test collect(right_pairs) ==
              [(I, I + CartesianIndex(1, 0)) for I in get_right_face(range_struct, 1)]

        dim2_left_pairs = @inferred get_left_face_pairs(range_struct, Dim(2))
        @test first(dim2_left_pairs) == (CartesianIndex(2, 1), CartesianIndex(2, 2))
        @test collect(dim2_left_pairs) ==
              [(I, I + CartesianIndex(0, 1)) for I in get_left_face(range_struct, Dim(2))]

        left_region = @inferred get_left_face_region(range_struct, Dim(1))
        internal_region = @inferred get_internal_face_region(range_struct, Dim(1))
        right_region_dim2 = @inferred get_right_face_region(range_struct, Dim(2))
        @test left_region == FaceKernelRegion(CartesianIndex(1, 2), (1, 5), CartesianIndex(1, 0), false, true)
        @test internal_region == FaceKernelRegion(CartesianIndex(2, 2), (3, 4), CartesianIndex(1, 0), true, true)
        @test right_region_dim2 == FaceKernelRegion(CartesianIndex(2, 6), (4, 1), CartesianIndex(0, 1), true, false)

        topology = CartesianTopology(MPI.COMM_SELF, (1, 1); periodic=(false, false))
        mpi_ha = HaloArray(Int, (4, 5), 1, topology; boundary_condition=:repeating)
        mpi_ranges = FaceRanges(mpi_ha)
        @test collect(get_left_face(mpi_ranges, 1)) == collect(get_left_face(range_struct, 1))
        @test collect(get_internal_face(mpi_ranges)) == collect(get_internal_face(range_struct))
        @test collect(get_right_face(mpi_ranges, 1)) == collect(get_right_face(range_struct, 1))
        @test get_unit_vector(mpi_ranges, 1) == CartesianIndex(1, 0)
        @test collect(get_left_face_pairs(mpi_ranges, 1)) == collect(get_left_face_pairs(range_struct, 1))
        @test get_internal_face_region(mpi_ranges, 1) == get_internal_face_region(range_struct, 1)

        threaded_ha = ThreadedHaloArray(Int, (4, 5), 1; dims=(1, 1), boundary_condition=:repeating)
        threaded_ranges = FaceRanges(threaded_ha)
        @test collect(get_left_face(threaded_ranges, 1)) == collect(get_left_face(range_struct, 1))
        @test collect(get_internal_face(threaded_ranges)) == collect(get_internal_face(range_struct))
        @test collect(get_right_face(threaded_ranges, 1)) == collect(get_right_face(range_struct, 1))
        @test get_unit_vector(threaded_ranges, 1) == CartesianIndex(1, 0)
        @test collect(get_internal_face_pairs(threaded_ranges, 1)) == collect(get_internal_face_pairs(range_struct, 1))
        @test get_right_face_region(threaded_ranges, 1) == get_right_face_region(range_struct, 1)

        fields = MultiHaloArray((;
            rho=LocalHaloArray(Int, (4, 5), 1; boundary_condition=:repeating),
            mom=LocalHaloArray(Int, (4, 5), 1; boundary_condition=:repeating),
        ))
        field_ranges = FaceRanges(fields)
        @test collect(get_left_face(field_ranges, 1)) == collect(get_left_face(range_struct, 1))
        @test collect(get_internal_face(field_ranges)) == collect(get_internal_face(range_struct))
        @test collect(get_right_face(field_ranges, 1)) == collect(get_right_face(range_struct, 1))
        @test get_unit_vector(field_ranges, 1) == CartesianIndex(1, 0)
        @test_throws BoundsError get_left_face(field_ranges, 3)
        @test collect(get_right_face_pairs(field_ranges, 1)) == collect(get_right_face_pairs(range_struct, 1))
        @test get_left_face_region(field_ranges, 1) == get_left_face_region(range_struct, 1)
        @test_throws BoundsError get_left_face_pairs(field_ranges, 3)

        array_fields = ArrayOfHaloArray([
            LocalHaloArray(Int, (4, 5), 1; boundary_condition=:repeating) for _ in 1:2, _ in 1:2
        ])
        array_field_ranges = FaceRanges(array_fields)
        @test collect(get_left_face(array_field_ranges, 1)) == collect(get_left_face(range_struct, 1))
        @test collect(get_internal_face(array_field_ranges)) == collect(get_internal_face(range_struct))
        @test collect(get_right_face(array_field_ranges, 1)) == collect(get_right_face(range_struct, 1))
        @test get_unit_vector(array_field_ranges, 1) == CartesianIndex(1, 0)
        @test_throws BoundsError get_right_face(array_field_ranges, 3)
        @test collect(get_left_face_pairs(array_field_ranges, 1)) == collect(get_left_face_pairs(range_struct, 1))
        @test get_right_face_region(array_field_ranges, 1) == get_right_face_region(range_struct, 1)

        @test collect(get_left_face_pairs(one_cell_ranges, 1)) ==
              [(CartesianIndex(1), CartesianIndex(2))]
        @test isempty(get_internal_face_pairs(one_cell_ranges, 1))
        @test collect(get_right_face_pairs(one_cell_ranges, 1)) ==
              [(CartesianIndex(2), CartesianIndex(3))]
        one_cell_internal_region = @inferred get_internal_face_region(one_cell_ranges, Dim(1))
        @test one_cell_internal_region ==
              FaceKernelRegion(CartesianIndex(2), (0,), CartesianIndex(1), true, true)
    end

    @testset "face ranges support owned-cell update" begin
        u = LocalHaloArray(Int, (4,), 1; boundary_condition=:repeating)
        du = similar(u)

        parent(u) .= [100, 1, 2, 3, 4, 200]
        fill!(parent(du), 0)

        ranges = FaceRanges(u)

        for (IL, IR) in get_left_face_pairs(ranges, 1)
            parent(du)[IR] += parent(u)[IR] - parent(u)[IL]
        end

        for (IL, IR) in get_internal_face_pairs(ranges, 1)
            flux = parent(u)[IR] - parent(u)[IL]
            parent(du)[IL] -= flux
            parent(du)[IR] += flux
        end

        for (IL, IR) in get_right_face_pairs(ranges, 1)
            parent(du)[IL] -= parent(u)[IR] - parent(u)[IL]
        end

        @test collect(interior_view(du)) == [-100, 0, 0, -195]
        @test parent(du)[1] == 0
        @test parent(du)[end] == 0
    end

    @testset "is_root array fallbacks" begin
        topology = CartesianTopology(MPI.COMM_SELF, (1,); periodic=(false,))
        distributed = HaloArray(Float64, (4,), 1, topology; boundary_condition=:repeating)
        local_field = LocalHaloArray(Float64, (4,), 1; boundary_condition=:repeating)
        threaded = ThreadedHaloArray(Float64, (4,), 1; dims=(2,), boundary_condition=:repeating)
        fields = MultiHaloArray((; rho=local_field, mom=similar(local_field)))
        array_fields = ArrayOfHaloArray([local_field, similar(local_field)])

        @test is_root(distributed)
        @test is_root(local_field)
        @test is_root(threaded)
        @test is_root(fields)
        @test is_root(array_fields)
        @test is_root(MaybeHaloArray(local_field))
        @test !is_root(distributed; root=1)
    end

    @testset "halo_backend traits" begin
        topology = CartesianTopology(MPI.COMM_SELF, (1,); periodic=(false,))
        distributed = HaloArray(Float64, (4,), 1, topology; boundary_condition=:repeating)
        local_field = LocalHaloArray(Float64, (4,), 1; boundary_condition=:repeating)
        threaded = ThreadedHaloArray(Float64, (2,), 1; dims=(2,), boundary_condition=:repeating)

        @test halo_backend(distributed) isa MPIHaloBackend
        @test halo_backend(local_field) isa LocalHaloBackend
        @test halo_backend(threaded) isa ThreadedHaloBackend
        @test halo_backend(MultiHaloArray((; rho=distributed, mom=similar(distributed)))) isa MPIHaloBackend
        @test halo_backend(MultiHaloArray((; rho=local_field, mom=similar(local_field)))) isa LocalHaloBackend
        @test halo_backend(MultiHaloArray((; rho=threaded, mom=similar(threaded)))) isa ThreadedHaloBackend
        @test halo_backend(ArrayOfHaloArray([local_field, similar(local_field)])) isa LocalHaloBackend
        @test halo_backend(ArrayOfHaloArray([threaded, similar(threaded)])) isa ThreadedHaloBackend
        @test halo_backend(MaybeHaloArray(threaded)) isa ThreadedHaloBackend

        @test_throws ArgumentError MultiHaloArray((; rho=local_field, mom=threaded))
        @test_throws ArgumentError ArrayOfHaloArray(Any[local_field, threaded])
    end

end
