using Test
using MPI
using HaloArrays
using LinearAlgebra

if !MPI.Initialized()
    MPI.Init()
end

struct CustomBoundaryForTest <: HaloArrays.AbstractBoundaryCondition end

# exercises the public launch-index mapping for colored face regions
_colored_region_index(region::ColoredFaceKernelRegion{N}, J::CartesianIndex{N}) where {N} =
    cell_index(region, J)

function _colored_region_indices(region::ColoredFaceKernelRegion)
    return vec([_colored_region_index(region, J) for J in CartesianIndices(region.size)])
end

_colored_face_indices(indices::CartesianIndices) = vec(collect(indices))

function _cell_subrange_indices(ranges::Tuple)
    return reduce(vcat, map(r -> vec(collect(r)), ranges))
end

function _colored_cell_region_indices(region::ColoredCellKernelRegion{N}) where {N}
    indices = CartesianIndex{N}[]

    for J in CartesianIndices(region.size)
        I = cell_index(region, J)
        is_cell_index_inbounds(region, I) && push!(indices, I)
    end

    return indices
end

function _has_nearest_neighbor_conflict(indices)
    isempty(indices) && return false
    index_set = Set(indices)
    N = length(Tuple(first(indices)))

    for I in indices
        for d in 1:N
            e = CartesianIndex(ntuple(j -> j == d ? 1 : 0, Val(N)))
            (I + e in index_set) && return true
        end
    end

    return false
end

function _apply_colored_face_update!(du_data, u_data, indices, offset, lower_owned, upper_owned)
    for IL in indices
        IR = IL + offset
        flux = u_data[IR] - u_data[IL]
        lower_owned && (du_data[IL] -= flux)
        upper_owned && (du_data[IR] += flux)
    end
    return du_data
end

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

        @test HaloArrays.left_face_range(ha, 1) == (1:1, 2:6)
        @test HaloArrays.right_face_range(ha, 1) == (5:5, 2:6)
        @test HaloArrays.face_offset(ha, 1) == CartesianIndex(1, 0)

        dim2_ranges = FaceRanges(ha)
        @test collect(get_left_face(dim2_ranges, Dim(2))) == collect(CartesianIndices((2:5, 1:1)))
        @test collect(get_right_face(dim2_ranges, Dim(2))) == collect(CartesianIndices((2:5, 6:6)))
        @test HaloArrays.face_offset(ha, Dim(2)) == CartesianIndex(0, 1)
        @test get_unit_vector(dim2_ranges, Dim(2)) == CartesianIndex(0, 1)

        # direction-aware internal faces keep the transverse dimension full
        @test HaloArrays.internal_face_range(ha, 1) == (2:4, 2:6)
        @test HaloArrays.internal_face_range(ha, 2) == (2:5, 2:5)
        @test collect(get_internal_face(dim2_ranges, 1)) == collect(CartesianIndices((2:4, 2:6)))
        @test collect(get_internal_face(dim2_ranges, Dim(2))) == collect(CartesianIndices((2:5, 2:5)))

        one_cell = LocalHaloArray(Int, (1,), 1; boundary_condition=:repeating)
        one_cell_ranges = FaceRanges(one_cell)
        @test collect(get_left_face(one_cell_ranges, 1)) == [CartesianIndex(1)]
        @test isempty(get_internal_face(one_cell_ranges, 1))
        @test collect(get_right_face(one_cell_ranges, 1)) == [CartesianIndex(2)]

        range_struct = HaloArrays.FaceRanges(ha)
        @test collect(HaloArrays.get_left_face(range_struct, 1)) == collect(CartesianIndices((1:1, 2:6)))
        @test collect(HaloArrays.get_right_face(range_struct, 1)) == collect(CartesianIndices((5:5, 2:6)))

        # accumulate_flux_divergence! is conservative on a 2-D uniform field:
        # a per-direction sweep must give zero update in every row/column,
        # including the last transverse one (regression for the dim-aware fix).
        u = LocalHaloArray(Float64, (6, 6), 1; boundary_condition=:repeating)
        fill!(interior_view(u), 2.0)
        synchronize_halo!(u)
        du = similar(u); fill!(parent(du), 0.0)
        fr = FaceRanges(u)
        accumulate_flux_divergence!(parent(du), parent(u), fr, 1, 1.0, (a, b) -> 0.5 * (a + b))
        accumulate_flux_divergence!(parent(du), parent(u), fr, 2, 1.0, (a, b) -> 0.5 * (a + b))
        @test maximum(abs, interior_view(du)) == 0.0
        @test HaloArrays.get_unit_vector(range_struct, 1) == CartesianIndex(1, 0)

        left_region = @inferred get_left_face_region(range_struct, Dim(1))
        internal_region = @inferred get_internal_face_region(range_struct, Dim(1))
        right_region_dim2 = @inferred get_right_face_region(range_struct, Dim(2))
        @test left_region == FaceKernelRegion(CartesianIndex(1, 2), (1, 5), CartesianIndex(1, 0), false, true)
        @test internal_region == FaceKernelRegion(CartesianIndex(2, 2), (3, 5), CartesianIndex(1, 0), true, true)
        @test right_region_dim2 == FaceKernelRegion(CartesianIndex(2, 6), (4, 1), CartesianIndex(0, 1), true, false)

        left_face_color0 = @inferred get_colored_left_face(range_struct, Dim(1), 0)
        left_face_color1 = @inferred get_colored_left_face(range_struct, Dim(1), 1)
        internal_face_color0 = @inferred get_colored_internal_face(range_struct, Dim(1), 0)
        internal_face_color1 = @inferred get_colored_internal_face(range_struct, Dim(1), 1)
        right_face_dim2_color0 = @inferred get_colored_right_face(range_struct, Dim(2), 0)
        right_face_dim2_color1 = @inferred get_colored_right_face(range_struct, Dim(2), 1)

        @test _colored_face_indices(left_face_color0) == CartesianIndex{2}[]
        @test _colored_face_indices(left_face_color1) == vec(collect(CartesianIndices((1:2:1, 2:6))))
        @test _colored_face_indices(internal_face_color0) == vec(collect(CartesianIndices((2:2:4, 2:6))))
        @test _colored_face_indices(internal_face_color1) == vec(collect(CartesianIndices((3:2:3, 2:6))))
        @test _colored_face_indices(right_face_dim2_color0) == vec(collect(CartesianIndices((2:5, 6:2:6))))
        @test _colored_face_indices(right_face_dim2_color1) == CartesianIndex{2}[]
        @test Set(vcat(_colored_face_indices(internal_face_color0), _colored_face_indices(internal_face_color1))) ==
              Set(collect(get_internal_face(range_struct, 1)))
        @test Set(vcat(_colored_face_indices(right_face_dim2_color0), _colored_face_indices(right_face_dim2_color1))) ==
              Set(collect(get_right_face(range_struct, Dim(2))))
        @test_throws ArgumentError get_colored_internal_face(range_struct, 1, -1)
        @test_throws ArgumentError get_colored_internal_face(range_struct, 1, 2)

        left_color0 = @inferred get_colored_left_face_region(range_struct, Dim(1), 0)
        left_color1 = @inferred get_colored_left_face_region(range_struct, Dim(1), 1)
        internal_color0 = @inferred get_colored_internal_face_region(range_struct, Dim(1), 0)
        internal_color1 = @inferred get_colored_internal_face_region(range_struct, Dim(1), 1)
        right_dim2_color0 = @inferred get_colored_right_face_region(range_struct, Dim(2), 0)
        right_dim2_color1 = @inferred get_colored_right_face_region(range_struct, Dim(2), 1)

        @test left_color0 ==
              ColoredFaceKernelRegion(CartesianIndex(2, 2), (0, 5), CartesianIndex(2, 1), CartesianIndex(1, 0), false, true)
        @test left_color1 ==
              ColoredFaceKernelRegion(CartesianIndex(1, 2), (1, 5), CartesianIndex(2, 1), CartesianIndex(1, 0), false, true)
        @test internal_color0 ==
              ColoredFaceKernelRegion(CartesianIndex(2, 2), (2, 5), CartesianIndex(2, 1), CartesianIndex(1, 0), true, true)
        @test internal_color1 ==
              ColoredFaceKernelRegion(CartesianIndex(3, 2), (1, 5), CartesianIndex(2, 1), CartesianIndex(1, 0), true, true)
        @test right_dim2_color0 ==
              ColoredFaceKernelRegion(CartesianIndex(2, 6), (4, 1), CartesianIndex(1, 2), CartesianIndex(0, 1), true, false)
        @test right_dim2_color1 ==
              ColoredFaceKernelRegion(CartesianIndex(2, 7), (4, 0), CartesianIndex(1, 2), CartesianIndex(0, 1), true, false)
        @test _colored_region_indices(internal_color0) == _colored_face_indices(internal_face_color0)
        @test _colored_region_indices(internal_color1) == _colored_face_indices(internal_face_color1)
        @test _colored_region_indices(right_dim2_color0) == _colored_face_indices(right_face_dim2_color0)
        @test _colored_region_indices(right_dim2_color1) == _colored_face_indices(right_face_dim2_color1)
        @test_throws ArgumentError get_colored_internal_face_region(range_struct, 1, -1)
        @test_throws ArgumentError get_colored_internal_face_region(range_struct, 1, 2)

        topology = CartesianTopology(MPI.COMM_SELF, (1, 1); periodic=(false, false))
        mpi_ha = HaloArray(Int, (4, 5), 1, topology; boundary_condition=:repeating)
        mpi_ranges = FaceRanges(mpi_ha)
        @test collect(get_left_face(mpi_ranges, 1)) == collect(get_left_face(range_struct, 1))
        @test collect(get_internal_face(mpi_ranges, 1)) == collect(get_internal_face(range_struct, 1))
        @test collect(get_right_face(mpi_ranges, 1)) == collect(get_right_face(range_struct, 1))
        @test get_unit_vector(mpi_ranges, 1) == CartesianIndex(1, 0)
        @test get_left_face_region(mpi_ranges, 1) == get_left_face_region(range_struct, 1)
        @test get_internal_face_region(mpi_ranges, 1) == get_internal_face_region(range_struct, 1)
        @test _colored_face_indices(get_colored_left_face(mpi_ranges, 1, 1)) ==
              _colored_face_indices(get_colored_left_face(range_struct, 1, 1))
        @test _colored_face_indices(get_colored_internal_face(mpi_ranges, 1, 0)) ==
              _colored_face_indices(get_colored_internal_face(range_struct, 1, 0))
        @test get_colored_left_face_region(mpi_ranges, 1, 1) ==
              get_colored_left_face_region(range_struct, 1, 1)
        @test get_colored_internal_face_region(mpi_ranges, 1, 0) ==
              get_colored_internal_face_region(range_struct, 1, 0)

        threaded_ha = ThreadedHaloArray(Int, (4, 5), 1; dims=(1, 1), boundary_condition=:repeating)
        threaded_ranges = FaceRanges(threaded_ha)
        @test collect(get_left_face(threaded_ranges, 1)) == collect(get_left_face(range_struct, 1))
        @test collect(get_internal_face(threaded_ranges, 1)) == collect(get_internal_face(range_struct, 1))
        @test collect(get_right_face(threaded_ranges, 1)) == collect(get_right_face(range_struct, 1))
        @test get_unit_vector(threaded_ranges, 1) == CartesianIndex(1, 0)
        @test get_internal_face_region(threaded_ranges, 1) == get_internal_face_region(range_struct, 1)
        @test get_right_face_region(threaded_ranges, 1) == get_right_face_region(range_struct, 1)
        @test _colored_face_indices(get_colored_internal_face(threaded_ranges, 1, 1)) ==
              _colored_face_indices(get_colored_internal_face(range_struct, 1, 1))
        @test _colored_face_indices(get_colored_right_face(threaded_ranges, 1, 1)) ==
              _colored_face_indices(get_colored_right_face(range_struct, 1, 1))
        @test get_colored_internal_face_region(threaded_ranges, 1, 1) ==
              get_colored_internal_face_region(range_struct, 1, 1)
        @test get_colored_right_face_region(threaded_ranges, 1, 1) ==
              get_colored_right_face_region(range_struct, 1, 1)

        fields = MultiHaloArray((;
            rho=LocalHaloArray(Int, (4, 5), 1; boundary_condition=:repeating),
            mom=LocalHaloArray(Int, (4, 5), 1; boundary_condition=:repeating),
        ))
        field_ranges = FaceRanges(fields)
        @test collect(get_left_face(field_ranges, 1)) == collect(get_left_face(range_struct, 1))
        @test collect(get_internal_face(field_ranges, 1)) == collect(get_internal_face(range_struct, 1))
        @test collect(get_right_face(field_ranges, 1)) == collect(get_right_face(range_struct, 1))
        @test get_unit_vector(field_ranges, 1) == CartesianIndex(1, 0)
        @test_throws BoundsError get_left_face(field_ranges, 3)
        @test get_left_face_region(field_ranges, 1) == get_left_face_region(range_struct, 1)
        @test get_right_face_region(field_ranges, 1) == get_right_face_region(range_struct, 1)
        @test_throws BoundsError get_left_face_region(field_ranges, 3)
        @test _colored_face_indices(get_colored_left_face(field_ranges, 1, 1)) ==
              _colored_face_indices(get_colored_left_face(range_struct, 1, 1))
        @test _colored_face_indices(get_colored_right_face(field_ranges, 1, 1)) ==
              _colored_face_indices(get_colored_right_face(range_struct, 1, 1))
        @test get_colored_left_face_region(field_ranges, 1, 1) ==
              get_colored_left_face_region(range_struct, 1, 1)
        @test get_colored_right_face_region(field_ranges, 1, 1) ==
              get_colored_right_face_region(range_struct, 1, 1)
        @test_throws BoundsError get_colored_left_face(field_ranges, 3, 1)
        @test_throws BoundsError get_colored_left_face_region(field_ranges, 3, 1)

        array_fields = ArrayOfHaloArray([
            LocalHaloArray(Int, (4, 5), 1; boundary_condition=:repeating) for _ in 1:2, _ in 1:2
        ])
        array_field_ranges = FaceRanges(array_fields)
        @test collect(get_left_face(array_field_ranges, 1)) == collect(get_left_face(range_struct, 1))
        @test collect(get_internal_face(array_field_ranges, 1)) == collect(get_internal_face(range_struct, 1))
        @test collect(get_right_face(array_field_ranges, 1)) == collect(get_right_face(range_struct, 1))
        @test get_unit_vector(array_field_ranges, 1) == CartesianIndex(1, 0)
        @test_throws BoundsError get_right_face(array_field_ranges, 3)
        @test get_left_face_region(array_field_ranges, 1) == get_left_face_region(range_struct, 1)
        @test get_right_face_region(array_field_ranges, 1) == get_right_face_region(range_struct, 1)
        @test _colored_face_indices(get_colored_left_face(array_field_ranges, 1, 1)) ==
              _colored_face_indices(get_colored_left_face(range_struct, 1, 1))
        @test _colored_face_indices(get_colored_right_face(array_field_ranges, 1, 1)) ==
              _colored_face_indices(get_colored_right_face(range_struct, 1, 1))
        @test get_colored_left_face_region(array_field_ranges, 1, 1) ==
              get_colored_left_face_region(range_struct, 1, 1)
        @test get_colored_right_face_region(array_field_ranges, 1, 1) ==
              get_colored_right_face_region(range_struct, 1, 1)

        one_cell_left_region = @inferred get_left_face_region(one_cell_ranges, Dim(1))
        one_cell_internal_region = @inferred get_internal_face_region(one_cell_ranges, Dim(1))
        one_cell_right_region = @inferred get_right_face_region(one_cell_ranges, Dim(1))
        @test one_cell_left_region == FaceKernelRegion(CartesianIndex(1), (1,), CartesianIndex(1), false, true)
        @test one_cell_internal_region == FaceKernelRegion(CartesianIndex(2), (0,), CartesianIndex(1), true, true)
        @test one_cell_right_region == FaceKernelRegion(CartesianIndex(2), (1,), CartesianIndex(1), true, false)

        one_cell_internal_color0 = @inferred get_colored_internal_face_region(one_cell_ranges, Dim(1), 0)
        one_cell_internal_color1 = @inferred get_colored_internal_face_region(one_cell_ranges, Dim(1), 1)
        @test one_cell_internal_color0 ==
              ColoredFaceKernelRegion(CartesianIndex(2), (0,), CartesianIndex(2), CartesianIndex(1), true, true)
        @test one_cell_internal_color1 ==
              ColoredFaceKernelRegion(CartesianIndex(3), (0,), CartesianIndex(2), CartesianIndex(1), true, true)
    end

    @testset "accumulate_flux_divergence!" begin
        nx = 5
        u = LocalHaloArray(Float64, (nx,), 1; boundary_condition=:repeating)
        interior_view(u) .= [1.0, 2.0, 4.0, 7.0, 11.0]
        synchronize_halo!(u)
        ranges = FaceRanges(u)
        flux(uL, uR) = 0.5 * (uL + uR)

        # Reference: explicit left / internal / right conservative scatter.
        pu = parent(u)
        ref = zeros(nx + 2)
        ref[2] += flux(pu[1], pu[2])                       # left ghost|owned
        for i in 2:5                                       # internal owned faces
            F = flux(pu[i], pu[i + 1])
            ref[i] -= F
            ref[i + 1] += F
        end
        ref[6] -= flux(pu[6], pu[7])                       # right owned|ghost

        # Scalar default read/scatter.
        du = LocalHaloArray(Float64, (nx,), 1; boundary_condition=:repeating)
        fill!(parent(du), 0.0)
        @test accumulate_flux_divergence!(parent(du), parent(u), ranges, 1, 1.0, flux) ===
              parent(du)
        @test parent(du) ≈ ref

        # Dim{D} accepted as well, scale applied.
        du2 = LocalHaloArray(Float64, (nx,), 1; boundary_condition=:repeating)
        fill!(parent(du2), 0.0)
        accumulate_flux_divergence!(parent(du2), parent(u), ranges, Dim(1), 2.0, flux)
        @test parent(du2) ≈ 2.0 .* ref

        # Custom read/scatter (two-field state via a tuple of arrays).
        v = LocalHaloArray(Float64, (nx,), 1; boundary_condition=:repeating)
        interior_view(v) .= [10.0, 20.0, 30.0, 40.0, 50.0]
        synchronize_halo!(v)
        dua = LocalHaloArray(Float64, (nx,), 1; boundary_condition=:repeating)
        dub = LocalHaloArray(Float64, (nx,), 1; boundary_condition=:repeating)
        fill!(parent(dua), 0.0); fill!(parent(dub), 0.0)

        read2(d, I) = (d[1][I], d[2][I])
        flux2(L, R) = (0.5 * (L[1] + R[1]), 0.5 * (L[2] + R[2]))
        function scatter2!(d, I, s, F)
            d[1][I] += s * F[1]
            d[2][I] += s * F[2]
            return d
        end
        accumulate_flux_divergence!((parent(dua), parent(dub)), (parent(u), parent(v)),
            ranges, 1, 1.0, flux2, read2, scatter2!)

        ref_v = zeros(nx + 2)
        pv = parent(v)
        ref_v[2] += flux(pv[1], pv[2])
        for i in 2:5
            F = flux(pv[i], pv[i + 1])
            ref_v[i] -= F
            ref_v[i + 1] += F
        end
        ref_v[6] -= flux(pv[6], pv[7])
        @test parent(dua) ≈ ref      # first component matches the scalar field u
        @test parent(dub) ≈ ref_v    # second component matches the scalar field v
    end

    @testset "owned cell ranges" begin
        ha = LocalHaloArray(Int, (4, 5), 1; boundary_condition=:repeating)
        ranges = CellRanges(ha)
        owned_cells = @inferred get_interior_cells(ranges)

        @test collect(owned_cells) == collect(CartesianIndices((2:5, 2:6)))

        color0_ranges = @inferred get_colored_interior_cell_ranges(ranges, 0)
        color1_ranges = @inferred get_colored_interior_cell_ranges(ranges, 1)

        @test length(color0_ranges) == 2
        @test length(color1_ranges) == 2
        @test collect(color0_ranges[1]) == collect(CartesianIndices((2:2:4, 2:2:6)))
        @test collect(color0_ranges[2]) == collect(CartesianIndices((3:2:5, 3:2:5)))
        @test collect(color1_ranges[1]) == collect(CartesianIndices((2:2:4, 3:2:5)))
        @test collect(color1_ranges[2]) == collect(CartesianIndices((3:2:5, 2:2:6)))

        color0_cells = _cell_subrange_indices(color0_ranges)
        color1_cells = _cell_subrange_indices(color1_ranges)
        all_colored_cells = vcat(color0_cells, color1_cells)

        @test Set(all_colored_cells) == Set(collect(owned_cells))
        @test length(all_colored_cells) == length(owned_cells)
        @test all(I -> mod(sum(Tuple(I)), 2) == 0, color0_cells)
        @test all(I -> mod(sum(Tuple(I)), 2) == 1, color1_cells)
        @test !_has_nearest_neighbor_conflict(color0_cells)
        @test !_has_nearest_neighbor_conflict(color1_cells)
        @test_throws ArgumentError get_colored_interior_cell_ranges(ranges, -1)
        @test_throws ArgumentError get_colored_interior_cell_ranges(ranges, 2)

        cell_region = @inferred get_interior_cell_region(ranges)
        color0_region = @inferred get_colored_interior_cell_region(ranges, 0)
        color1_region = @inferred get_colored_interior_cell_region(ranges, 1, Dim(1))
        color0_dim2_region = @inferred get_colored_interior_cell_region(ranges, 0, Dim(2))

        @test cell_region == CellKernelRegion(CartesianIndex(2, 2), (4, 5))
        @test color0_region == ColoredCellKernelRegion(CartesianIndex(2, 2), (2, 5), (4, 5), 0, 1)
        @test color1_region == ColoredCellKernelRegion(CartesianIndex(2, 2), (2, 5), (4, 5), 1, 1)
        @test color0_dim2_region == ColoredCellKernelRegion(CartesianIndex(2, 2), (4, 3), (4, 5), 0, 2)
        @test @inferred(cell_index(cell_region, CartesianIndex(1, 1))) == CartesianIndex(2, 2)
        @test @inferred(cell_index(color0_region, CartesianIndex(1, 1))) == CartesianIndex(2, 2)
        @test @inferred(cell_index(color1_region, (1, 1))) == (3, 2)
        @test @inferred(is_cell_index_inbounds(color0_region, CartesianIndex(5, 6)))
        @test !@inferred(is_cell_index_inbounds(color0_region, CartesianIndex(7, 6)))
        @test Set(_colored_cell_region_indices(color0_region)) == Set(color0_cells)
        @test Set(_colored_cell_region_indices(color1_region)) == Set(color1_cells)
        @test Set(_colored_cell_region_indices(color0_dim2_region)) == Set(color0_cells)
        @test all(I -> mod(sum(Tuple(I)), 2) == 0, _colored_cell_region_indices(color0_region))
        @test all(I -> mod(sum(Tuple(I)), 2) == 1, _colored_cell_region_indices(color1_region))
        @test_throws ArgumentError get_colored_interior_cell_region(ranges, -1)
        @test_throws ArgumentError get_colored_interior_cell_region(ranges, 2)
        @test_throws ArgumentError get_colored_interior_cell_region(ranges, 0, 0)
        @test_throws ArgumentError get_colored_interior_cell_region(ranges, 0, 3)

        one_d = LocalHaloArray(Int, (4,), 1; boundary_condition=:repeating)
        one_d_ranges = CellRanges(one_d)
        one_d_color0 = @inferred get_colored_interior_cell_ranges(one_d_ranges, 0)
        one_d_color1 = @inferred get_colored_interior_cell_ranges(one_d_ranges, 1)

        @test collect(get_interior_cells(one_d_ranges)) == collect(CartesianIndices((2:5,)))
        @test length(one_d_color0) == 1
        @test length(one_d_color1) == 1
        @test _cell_subrange_indices(one_d_color0) == collect(CartesianIndices((2:2:4,)))
        @test _cell_subrange_indices(one_d_color1) == collect(CartesianIndices((3:2:5,)))

        one_cell = LocalHaloArray(Int, (1,), 1; boundary_condition=:repeating)
        one_cell_ranges = CellRanges(one_cell)
        one_cell_color0 = @inferred get_colored_interior_cell_ranges(one_cell_ranges, 0)
        one_cell_color1 = @inferred get_colored_interior_cell_ranges(one_cell_ranges, 1)
        one_cell_region_color0 = @inferred get_colored_interior_cell_region(one_cell_ranges, 0)
        one_cell_region_color1 = @inferred get_colored_interior_cell_region(one_cell_ranges, 1)

        @test collect(get_interior_cells(one_cell_ranges)) == [CartesianIndex(2)]
        @test _cell_subrange_indices(one_cell_color0) == [CartesianIndex(2)]
        @test isempty(_cell_subrange_indices(one_cell_color1))
        @test one_cell_region_color0 == ColoredCellKernelRegion(CartesianIndex(2), (1,), (1,), 0, 1)
        @test one_cell_region_color1 == ColoredCellKernelRegion(CartesianIndex(2), (1,), (1,), 1, 1)
        @test _colored_cell_region_indices(one_cell_region_color0) == [CartesianIndex(2)]
        @test isempty(_colored_cell_region_indices(one_cell_region_color1))

        three_d = LocalHaloArray(Int, (3, 4, 2), 1; boundary_condition=:repeating)
        three_d_ranges = CellRanges(three_d)
        three_d_color0 = @inferred get_colored_interior_cell_ranges(three_d_ranges, 0)
        three_d_color1 = @inferred get_colored_interior_cell_ranges(three_d_ranges, 1)
        three_d_color0_cells = _cell_subrange_indices(three_d_color0)
        three_d_color1_cells = _cell_subrange_indices(three_d_color1)
        three_d_color0_region = @inferred get_colored_interior_cell_region(three_d_ranges, 0)
        three_d_color1_region = @inferred get_colored_interior_cell_region(three_d_ranges, 1)

        @test length(three_d_color0) == 4
        @test length(three_d_color1) == 4
        @test Set(vcat(three_d_color0_cells, three_d_color1_cells)) ==
              Set(collect(get_interior_cells(three_d_ranges)))
        @test !_has_nearest_neighbor_conflict(three_d_color0_cells)
        @test !_has_nearest_neighbor_conflict(three_d_color1_cells)
        @test three_d_color0_region ==
              ColoredCellKernelRegion(CartesianIndex(2, 2, 2), (2, 4, 2), (3, 4, 2), 0, 1)
        @test three_d_color1_region ==
              ColoredCellKernelRegion(CartesianIndex(2, 2, 2), (2, 4, 2), (3, 4, 2), 1, 1)
        @test Set(_colored_cell_region_indices(three_d_color0_region)) == Set(three_d_color0_cells)
        @test Set(_colored_cell_region_indices(three_d_color1_region)) == Set(three_d_color1_cells)

        topology = CartesianTopology(MPI.COMM_SELF, (1, 1); periodic=(false, false))
        mpi_ha = HaloArray(Int, (4, 5), 1, topology; boundary_condition=:repeating)
        threaded_ha = ThreadedHaloArray(Int, (4, 5), 1; dims=(1, 1), boundary_condition=:repeating)
        fields = MultiHaloArray((;
            rho=LocalHaloArray(Int, (4, 5), 1; boundary_condition=:repeating),
            mom=LocalHaloArray(Int, (4, 5), 1; boundary_condition=:repeating),
        ))
        array_fields = ArrayOfHaloArray([
            LocalHaloArray(Int, (4, 5), 1; boundary_condition=:repeating) for _ in 1:2, _ in 1:2
        ])

        for other in (mpi_ha, threaded_ha, fields, array_fields)
            other_ranges = CellRanges(other)
            @test collect(get_interior_cells(other_ranges)) == collect(get_interior_cells(ranges))
            @test _cell_subrange_indices(get_colored_interior_cell_ranges(other_ranges, 0)) == color0_cells
            @test _cell_subrange_indices(get_colored_interior_cell_ranges(other_ranges, 1)) == color1_cells
            @test get_interior_cell_region(other_ranges) == cell_region
            @test get_colored_interior_cell_region(other_ranges, 0) == color0_region
            @test get_colored_interior_cell_region(other_ranges, 1) == color1_region
        end
    end

    @testset "face ranges support owned-cell update" begin
        u = LocalHaloArray(Int, (4,), 1; boundary_condition=:repeating)
        du = similar(u)

        parent(u) .= [100, 1, 2, 3, 4, 200]
        fill!(parent(du), 0)

        ranges = FaceRanges(u)
        offset = get_unit_vector(ranges, 1)

        for IL in get_left_face(ranges, 1)
            IR = IL + offset
            parent(du)[IR] += parent(u)[IR] - parent(u)[IL]
        end

        for IL in get_internal_face(ranges, 1)
            IR = IL + offset
            flux = parent(u)[IR] - parent(u)[IL]
            parent(du)[IL] -= flux
            parent(du)[IR] += flux
        end

        for IL in get_right_face(ranges, 1)
            IR = IL + offset
            parent(du)[IL] -= parent(u)[IR] - parent(u)[IL]
        end

        @test collect(interior_view(du)) == [-100, 0, 0, -195]
        @test parent(du)[1] == 0
        @test parent(du)[end] == 0
    end

    @testset "colored face ranges support checkerboard update" begin
        u = LocalHaloArray(Int, (4,), 1; boundary_condition=:repeating)
        du = similar(u)

        parent(u) .= [100, 1, 2, 3, 4, 200]
        fill!(parent(du), 0)

        ranges = FaceRanges(u)
        offset = get_unit_vector(ranges, 1)

        for color in 0:1
            color_touches = zeros(Int, length(parent(u)))
            colored_faces = (
                (get_colored_left_face(ranges, 1, color), false, true),
                (get_colored_internal_face(ranges, 1, color), true, true),
                (get_colored_right_face(ranges, 1, color), true, false),
            )

            for (indices, lower_owned, upper_owned) in colored_faces
                for IL in indices
                    IR = IL + offset
                    lower_owned && (color_touches[IL] += 1)
                    upper_owned && (color_touches[IR] += 1)
                end
                _apply_colored_face_update!(parent(du), parent(u), indices, offset, lower_owned, upper_owned)
            end

            @test maximum(color_touches) <= 1
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

    @testset "CellKernelRegion cell_index and is_cell_index_inbounds" begin
        u = LocalHaloArray(Float64, (4,), 1; boundary_condition=:repeating)
        cr = CellRanges(u)
        region = get_interior_cell_region(cr)

        # first owned cell is at storage index halo+1 = 2
        @test cell_index(region, (1,)) == (2,)
        @test cell_index(region, (4,)) == (5,)   # last owned cell
        @test cell_index(region, CartesianIndex(2)) == CartesianIndex(3)

        @test is_cell_index_inbounds(region, (2,))   # first owned
        @test is_cell_index_inbounds(region, (5,))   # last owned
        @test !is_cell_index_inbounds(region, (1,))  # left ghost
        @test !is_cell_index_inbounds(region, (6,))  # right ghost

        # 2-D: halo=1, owned (3,3), storage 2:4 in each dim
        u2 = LocalHaloArray(Float64, (3, 3), 1; boundary_condition=:repeating)
        cr2 = CellRanges(u2)
        r2 = get_interior_cell_region(cr2)

        @test cell_index(r2, (1, 1)) == (2, 2)
        @test cell_index(r2, (3, 3)) == (4, 4)
        @test is_cell_index_inbounds(r2, (2, 2))
        @test !is_cell_index_inbounds(r2, (1, 2))   # ghost in dim 1
        @test !is_cell_index_inbounds(r2, (2, 5))   # out of bounds in dim 2
    end

    @testset "ColoredCellKernelRegion cell_index and is_cell_index_inbounds" begin
        # 1-D: 4 owned cells, halo=1 → storage 2:5
        u = LocalHaloArray(Float64, (4,), 1; boundary_condition=:repeating)
        cr = CellRanges(u)

        r0 = get_colored_interior_cell_region(cr, 0)   # color 0: even storage index
        r1 = get_colored_interior_cell_region(cr, 1)   # color 1: odd storage index

        # color 0 maps J=(1,) → storage 2, J=(2,) → storage 4
        @test cell_index(r0, (1,)) == (2,)
        @test cell_index(r0, (2,)) == (4,)
        # color 1 maps J=(1,) → storage 3, J=(2,) → storage 5
        @test cell_index(r1, (1,)) == (3,)
        @test cell_index(r1, (2,)) == (5,)

        # verify each result has the correct color: mod(i, 2) == color
        for (region, color) in ((r0, 0), (r1, 1))
            for j in 1:region.size[1]
                I = cell_index(region, (j,))
                @test mod(I[1], 2) == color
                @test is_cell_index_inbounds(region, I)
            end
        end

        # launch size is compressed: ceil(4/2) = 2
        @test r0.size == (2,)
        @test r1.size == (2,)
        # full (uncompressed) size is 4
        @test r0.full_size == (4,)

        # 2-D: 4×4 owned cells, compressed_dim=2
        u2 = LocalHaloArray(Float64, (4, 4), 1; boundary_condition=:repeating)
        cr2 = CellRanges(u2)
        r2c0 = get_colored_interior_cell_region(cr2, 0; compressed_dim=2)

        # launch size: (4, ceil(4/2)) = (4, 2)
        @test r2c0.size == (4, 2)
        @test r2c0.full_size == (4, 4)

        # every reconstructed cell should have color 0: mod(i+j, 2) == 0
        for ji in 1:r2c0.size[1], jj in 1:r2c0.size[2]
            I = cell_index(r2c0, (ji, jj))
            if is_cell_index_inbounds(r2c0, I)
                @test mod(sum(I), 2) == 0
            end
        end
    end

    @testset "size/axes honor the trailing-dimension contract" begin
        # size(A, i) / axes(A, i) must return 1 / OneTo(1) for i > ndims, per the
        # AbstractArray contract — generic code (e.g. ArrayInterface.zeromatrix,
        # which does `u .* u'`) relies on it.
        u = LocalHaloArray(Float64, (4,), 1; boundary_condition=:periodic)
        @test size(u, 2) == 1
        @test size(u, 5) == 1
        @test axes(u, 2) == Base.OneTo(1)

        t = ThreadedHaloArray(Float64, (3,), 1; dims=(2,), boundary_condition=:periodic)
        @test size(t, 2) == 1
        @test axes(t, 2) == Base.OneTo(1)

        mha = MultiHaloArray((; a=u, b=similar(u)))   # ndims == 2 (field axis + 1 spatial)
        @test size(mha, 3) == 1
        @test axes(mha, 3) == Base.OneTo(1)

        m = MaybeHaloArray(u)
        @test size(m, 2) == 1
        @test axes(m, 2) == Base.OneTo(1)

        # getindex/setindex! must accept trailing singleton indices to match
        # axes (e.g. Diagonal's ldiv!, used as a preconditioner, does `A[i, 1]`).
        interior_view(u) .= [10.0, 20.0, 30.0, 40.0]
        @test u[2] == u[2, 1] == u[2, 1, 1] == 20.0
        @test_throws BoundsError u[2, 2]
        u[3, 1] = 99.0
        @test u[3] == 99.0
        diag = LocalHaloArray(Float64, (4,), 1; boundary_condition=:periodic)
        interior_view(diag) .= 2.0
        x = LocalHaloArray(Float64, (4,), 1; boundary_condition=:periodic)
        interior_view(x) .= [2.0, 4.0, 6.0, 8.0]
        LinearAlgebra.ldiv!(LinearAlgebra.Diagonal(diag), x)
        @test collect(interior_view(x)) == [1.0, 2.0, 3.0, 4.0]
    end

    @testset "non-tiled arrays are a one-tile decomposition" begin
        # tile_count/tile_parent degenerate so a per-tile kernel
        # (`for t in 1:tile_count(u); tile_parent(u, t)`) is backend-agnostic.
        u = LocalHaloArray(Float64, (5,), 1; boundary_condition=:periodic)
        @test tile_count(u) == 1
        @test tile_parent(u, 1) === parent(u)
    end

end
