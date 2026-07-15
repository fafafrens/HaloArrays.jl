using Test
using MPI
using HaloArrays
using LinearAlgebra

if !MPI.Initialized()
    MPI.Init()
end

struct CustomBoundaryForTest <: HaloArrays.AbstractBoundaryCondition end

# exercises the public launch-index mapping for colored face regions
_colored_region_index(region::FaceCheckerboard{N}, J::CartesianIndex{N}) where {N} =
    cell_index(region, J)

function _colored_region_indices(region::FaceCheckerboard)
    return vec([_colored_region_index(region, J) for J in CartesianIndices(region.size)])
end

_colored_face_indices(indices::CartesianIndices) = vec(collect(indices))

function _cell_subrange_indices(ranges::Tuple)
    return reduce(vcat, map(r -> vec(collect(r)), ranges))
end

function _colored_cell_region_indices(region::CellCheckerboard{N}) where {N}
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

function _apply_colored_face_update!(du_data, u_data, indices, offset)
    for IL in indices
        IR = IL + offset
        flux = u_data[IR] - u_data[IL]
        du_data[IL] -= flux       # boundary faces also touch a ghost cell (harmless)
        du_data[IR] += flux
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

        # every face touching the interior along dim, as one contiguous span:
        # (first_interior-1):last_interior in dim, full in every transverse dim.
        @test HaloArrays.interior_face_range(ha, 1) == (1:5, 2:6)
        @test HaloArrays.interior_face_range(ha, 2) == (2:5, 1:6)
        @test unit_vector(ha, 1) == CartesianIndex(1, 0)

        dim2_ranges = FaceRanges(ha)
        @test collect(interior_faces(dim2_ranges, 1)) == collect(CartesianIndices((1:5, 2:6)))
        @test collect(interior_faces(dim2_ranges, Dim(2))) == collect(CartesianIndices((2:5, 1:6)))
        @test unit_vector(ha, Dim(2)) == CartesianIndex(0, 1)
        @test unit_vector(dim2_ranges, Dim(2)) == CartesianIndex(0, 1)
        @test unit_vector(ha) == (CartesianIndex(1, 0), CartesianIndex(0, 1))
        @test unit_vector(Val(2), 2) == CartesianIndex(0, 1)
        @test unit_vector(Val(2)) == (CartesianIndex(1, 0), CartesianIndex(0, 1))

        one_cell = LocalHaloArray(Int, (1,), 1; boundary_condition=:repeating)
        one_cell_ranges = FaceRanges(one_cell)
        # one interior cell ⇒ two faces: low-ghost|cell and cell|high-ghost
        @test collect(interior_faces(one_cell_ranges, 1)) == [CartesianIndex(1), CartesianIndex(2)]

        range_struct = HaloArrays.FaceRanges(ha)
        @test collect(HaloArrays.interior_faces(range_struct, 1)) == collect(CartesianIndices((1:5, 2:6)))

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
        @test HaloArrays.unit_vector(range_struct, 1) == CartesianIndex(1, 0)

        # the window wraps the face range (no owned flags now)
        @test (@inferred interior_face_window(range_struct, Dim(1))) ==
              FaceWindow(CartesianIndex(1, 2), (5, 5), CartesianIndex(1, 0))
        @test interior_face_window(range_struct, Dim(2)) ==
              FaceWindow(CartesianIndex(2, 1), (4, 6), CartesianIndex(0, 1))

        # the two checkerboard colors disjointly partition all the faces
        faces1 = Set(collect(interior_faces(range_struct, 1)))
        c0 = _colored_face_indices(@inferred interior_faces(range_struct, Dim(1), 0))
        c1 = _colored_face_indices(@inferred interior_faces(range_struct, Dim(1), 1))
        @test isempty(intersect(Set(c0), Set(c1)))
        @test Set(vcat(c0, c1)) == faces1
        @test_throws ArgumentError interior_faces(range_struct, 1, -1)
        @test_throws ArgumentError interior_faces(range_struct, 1, 2)

        # colored window indices match the colored range
        @test _colored_region_indices(@inferred interior_face_window(range_struct, Dim(1), 0)) == c0
        @test _colored_region_indices(@inferred interior_face_window(range_struct, Dim(1), 1)) == c1
        @test_throws ArgumentError interior_face_window(range_struct, 1, -1)
        @test_throws ArgumentError interior_face_window(range_struct, 1, 2)

        # identical faces/windows across MPI, threaded, and collection backends
        topology = CartesianTopology(MPI.COMM_SELF, (1, 1); periodic=(false, false))
        mpi_ranges = FaceRanges(HaloArray(Int, (4, 5), 1, topology; boundary_condition=:repeating))
        threaded_ranges = FaceRanges(ThreadedHaloArray(Int, (4, 5), 1; dims=(1, 1), boundary_condition=:repeating))
        field_ranges = FaceRanges(MultiHaloArray((;
            rho=LocalHaloArray(Int, (4, 5), 1; boundary_condition=:repeating),
            mom=LocalHaloArray(Int, (4, 5), 1; boundary_condition=:repeating),
        )))
        array_field_ranges = FaceRanges(ArrayOfHaloArray([
            LocalHaloArray(Int, (4, 5), 1; boundary_condition=:repeating) for _ in 1:2, _ in 1:2
        ]))
        for other in (mpi_ranges, threaded_ranges, field_ranges, array_field_ranges)
            @test collect(interior_faces(other, 1)) == collect(interior_faces(range_struct, 1))
            @test unit_vector(other, 1) == CartesianIndex(1, 0)
            @test interior_face_window(other, 1) == interior_face_window(range_struct, 1)
            @test interior_face_window(other, 1, 1) == interior_face_window(range_struct, 1, 1)
            @test _colored_face_indices(interior_faces(other, 1, 0)) ==
                  _colored_face_indices(interior_faces(range_struct, 1, 0))
        end
        @test_throws BoundsError interior_faces(field_ranges, 3)
        @test_throws BoundsError interior_face_window(field_ranges, 3)

        # one interior cell ⇒ two boundary faces, no internal faces
        @test (@inferred interior_face_window(one_cell_ranges, Dim(1))) ==
              FaceWindow(CartesianIndex(1), (2,), CartesianIndex(1))
        oc0 = _colored_face_indices(interior_faces(one_cell_ranges, 1, 0))
        oc1 = _colored_face_indices(interior_faces(one_cell_ranges, 1, 1))
        @test Set(vcat(oc0, oc1)) == Set(collect(interior_faces(one_cell_ranges, 1)))
    end

    @testset "accumulate_flux_divergence!" begin
        nx = 5
        u = LocalHaloArray(Float64, (nx,), 1; boundary_condition=:repeating)
        interior_view(u) .= [1.0, 2.0, 4.0, 7.0, 11.0]
        synchronize_halo!(u)
        ranges = FaceRanges(u)
        flux(uL, uR) = 0.5 * (uL + uR)

        # Reference: scatter every face's flux onto both cells (boundary faces
        # write the ghost cells too, which is in-bounds and harmless).
        pu = parent(u)
        ref = zeros(nx + 2)
        for IL in 1:(nx + 1)
            F = flux(pu[IL], pu[IL + 1])
            ref[IL]     -= F
            ref[IL + 1] += F
        end

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
        for IL in 1:(nx + 1)
            F = flux(pv[IL], pv[IL + 1])
            ref_v[IL]     -= F
            ref_v[IL + 1] += F
        end
        @test parent(dua) ≈ ref      # first component matches the scalar field u
        @test parent(dub) ≈ ref_v    # second component matches the scalar field v
    end

    @testset "owned cell ranges" begin
        ha = LocalHaloArray(Int, (4, 5), 1; boundary_condition=:repeating)
        ranges = CellRanges(ha)
        owned_cells = @inferred interior_cells(ranges)

        @test collect(owned_cells) == collect(CartesianIndices((2:5, 2:6)))

        color0_ranges = @inferred interior_cells(ranges, 0)
        color1_ranges = @inferred interior_cells(ranges, 1)

        @test length(color0_ranges) == 2
        @test length(color1_ranges) == 2
        # Subranges are generated in storage coordinates (color shifted by the
        # tile's parity offset); this local array has offset 0, so color 0 is the
        # storage-even set and its subranges come out in storage-mask order.
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
        @test_throws ArgumentError interior_cells(ranges, -1)
        @test_throws ArgumentError interior_cells(ranges, 2)

        cell_region = @inferred interior_cell_window(ranges)
        color0_region = @inferred interior_cell_window(ranges, 0)
        color1_region = @inferred interior_cell_window(ranges, 1, Dim(1))
        color0_dim2_region = @inferred interior_cell_window(ranges, 0, Dim(2))

        @test cell_region == CellWindow(CartesianIndex(2, 2), (4, 5))
        @test color0_region == CellCheckerboard(CartesianIndex(2, 2), (2, 5), (4, 5), 0, 1)
        @test color1_region == CellCheckerboard(CartesianIndex(2, 2), (2, 5), (4, 5), 1, 1)
        @test color0_dim2_region == CellCheckerboard(CartesianIndex(2, 2), (4, 3), (4, 5), 0, 2)
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
        @test_throws ArgumentError interior_cell_window(ranges, -1)
        @test_throws ArgumentError interior_cell_window(ranges, 2)
        @test_throws ArgumentError interior_cell_window(ranges, 0, 0)
        @test_throws ArgumentError interior_cell_window(ranges, 0, 3)

        one_d = LocalHaloArray(Int, (4,), 1; boundary_condition=:repeating)
        one_d_ranges = CellRanges(one_d)
        one_d_color0 = @inferred interior_cells(one_d_ranges, 0)
        one_d_color1 = @inferred interior_cells(one_d_ranges, 1)

        @test collect(interior_cells(one_d_ranges)) == collect(CartesianIndices((2:5,)))
        @test length(one_d_color0) == 1
        @test length(one_d_color1) == 1
        # global index = storage - halo, so color 0 (global-even) is storage 3,5.
        @test _cell_subrange_indices(one_d_color0) == collect(CartesianIndices((3:2:5,)))
        @test _cell_subrange_indices(one_d_color1) == collect(CartesianIndices((2:2:4,)))

        one_cell = LocalHaloArray(Int, (1,), 1; boundary_condition=:repeating)
        one_cell_ranges = CellRanges(one_cell)
        one_cell_color0 = @inferred interior_cells(one_cell_ranges, 0)
        one_cell_color1 = @inferred interior_cells(one_cell_ranges, 1)
        one_cell_region_color0 = @inferred interior_cell_window(one_cell_ranges, 0)
        one_cell_region_color1 = @inferred interior_cell_window(one_cell_ranges, 1)

        @test collect(interior_cells(one_cell_ranges)) == [CartesianIndex(2)]
        # the lone cell has global index 1 (odd) → color 1, not color 0.
        @test isempty(_cell_subrange_indices(one_cell_color0))
        @test _cell_subrange_indices(one_cell_color1) == [CartesianIndex(2)]
        @test one_cell_region_color0 == CellCheckerboard(CartesianIndex(2), (1,), (1,), 0, 1, 1)
        @test one_cell_region_color1 == CellCheckerboard(CartesianIndex(2), (1,), (1,), 1, 1, 1)
        @test isempty(_colored_cell_region_indices(one_cell_region_color0))
        @test _colored_cell_region_indices(one_cell_region_color1) == [CartesianIndex(2)]

        three_d = LocalHaloArray(Int, (3, 4, 2), 1; boundary_condition=:repeating)
        three_d_ranges = CellRanges(three_d)
        three_d_color0 = @inferred interior_cells(three_d_ranges, 0)
        three_d_color1 = @inferred interior_cells(three_d_ranges, 1)
        three_d_color0_cells = _cell_subrange_indices(three_d_color0)
        three_d_color1_cells = _cell_subrange_indices(three_d_color1)
        three_d_color0_region = @inferred interior_cell_window(three_d_ranges, 0)
        three_d_color1_region = @inferred interior_cell_window(three_d_ranges, 1)

        @test length(three_d_color0) == 4
        @test length(three_d_color1) == 4
        @test Set(vcat(three_d_color0_cells, three_d_color1_cells)) ==
              Set(collect(interior_cells(three_d_ranges)))
        @test !_has_nearest_neighbor_conflict(three_d_color0_cells)
        @test !_has_nearest_neighbor_conflict(three_d_color1_cells)
        # 3-D halo=1: global origin (1,1,1) vs storage first (2,2,2) → parity 1.
        @test three_d_color0_region ==
              CellCheckerboard(CartesianIndex(2, 2, 2), (2, 4, 2), (3, 4, 2), 0, 1, 1)
        @test three_d_color1_region ==
              CellCheckerboard(CartesianIndex(2, 2, 2), (2, 4, 2), (3, 4, 2), 1, 1, 1)
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
            @test collect(interior_cells(other_ranges)) == collect(interior_cells(ranges))
            @test _cell_subrange_indices(interior_cells(other_ranges, 0)) == color0_cells
            @test _cell_subrange_indices(interior_cells(other_ranges, 1)) == color1_cells
            @test interior_cell_window(other_ranges) == cell_region
            @test interior_cell_window(other_ranges, 0) == color0_region
            @test interior_cell_window(other_ranges, 1) == color1_region
        end
    end

    @testset "checkerboard stays globally continuous across tile seams" begin
        # 2 tiles × 3 interior cells (odd local extent) — a storage-local parity
        # anchor would give each tile the same local coloring and double-color
        # the cells straddling the seam. Building the ranges per tile with its
        # own global origin must reproduce a single global red/black checkerboard.
        u = ThreadedHaloArray(Int, (3,), 1; dims=(2,), boundary_condition=:repeating)
        hw = halo_width(u)

        to_global(tid, cells) =
            [CartesianIndex(interior_to_global_index(u, tid, Tuple(I) .- hw)) for I in cells]

        global0 = CartesianIndex{1}[]
        global1 = CartesianIndex{1}[]
        for tid in 1:tile_count(u)
            cr = CellRanges(u, tid)
            append!(global0, to_global(tid, _cell_subrange_indices(interior_cells(cr, 0))))
            append!(global1, to_global(tid, _cell_subrange_indices(interior_cells(cr, 1))))
        end

        # the 6 global cells are partitioned exactly once
        @test Set(vcat(global0, global1)) == Set(collect(CartesianIndices((1:6,))))
        @test length(global0) + length(global1) == 6
        # colors follow GLOBAL parity, continuous across the tile-1/tile-2 seam
        @test Set(Tuple(I)[1] for I in global0) == Set([2, 4, 6])
        @test Set(Tuple(I)[1] for I in global1) == Set([1, 3, 5])
        # no two same-color cells are global neighbours → race-free updates
        @test !_has_nearest_neighbor_conflict(global0)
        @test !_has_nearest_neighbor_conflict(global1)
    end

    @testset "two-array kernels reject mismatched geometry" begin
        # axpy!/axpby!/swap!/rotate!/reflect!/dot and the multi-array reductions
        # index BOTH parents with one array's range under @inbounds — an
        # unchecked mismatch was an out-of-bounds read/write (axpy! crashed) or
        # a silently partial dot/mapreduce (zip truncates; Base throws).
        x = LocalHaloArray(Float64, (8,), 1; boundary_condition=:periodic)
        y = LocalHaloArray(Float64, (4,), 1; boundary_condition=:periodic)   # different size
        w = LocalHaloArray(Float64, (8,), 2; boundary_condition=:periodic)   # same size, different halo
        interior_view(x) .= 1.0; interior_view(y) .= 2.0; interior_view(w) .= 3.0

        # kernels indexing the PADDED parents need identical storage: both a
        # different size (y) and a different halo width (w) are refused.
        for bad in (y, w)
            @test_throws DimensionMismatch axpy!(1.0, x, bad)
            @test_throws DimensionMismatch axpby!(1.0, x, 2.0, bad)
            @test_throws DimensionMismatch HaloArrays.swap!(x, bad)
            @test_throws DimensionMismatch rotate!(x, bad, 0.6, 0.8)
            @test_throws DimensionMismatch reflect!(x, bad, 0.6, 0.8)
            @test_throws DimensionMismatch dot(x, bad)
        end
        # interior-view kernels (multi-array mapreduce, map!) only need equal
        # interiors: a different SIZE is refused, a different halo width works.
        @test_throws DimensionMismatch mapreduce(+, +, x, y)
        @test_throws DimensionMismatch map!(+, similar(x), x, y)
        @test mapreduce(+, +, x, w) == 8 * (1.0 + 3.0)
        let d = similar(x)
            @test map!(+, d, x, w) === d
            @test collect(interior_view(d)) == fill(4.0, 8)
        end

        # mismatched tile layouts are rejected too (same interior, tiled differently)
        t2 = ThreadedHaloArray(Float64, (4,), 1; dims=(2,), boundary_condition=:periodic)
        t4 = ThreadedHaloArray(Float64, (2,), 1; dims=(4,), boundary_condition=:periodic)
        @test_throws DimensionMismatch dot(t2, t4)
        @test_throws DimensionMismatch axpy!(1.0, t2, t4)

        # matched geometry is unaffected
        z = similar(x); interior_view(z) .= 2.0
        @test axpy!(1.0, x, z) === z
        @test collect(interior_view(z)) == fill(3.0, 8)
        @test dot(x, z) ≈ 24.0
        @test mapreduce(+, +, x, z) ≈ 32.0
    end

    @testset "FaceRanges require halo width >= 1" begin
        # The boundary faces scatter into ghost cells; with halo 0 the face range
        # would start at storage index 0 and the @inbounds flux loop corrupted
        # memory (crashed the process). Must refuse loudly instead.
        u0 = LocalHaloArray(Float64, (4,), 0; boundary_condition=:noboundary)
        @test_throws ArgumentError FaceRanges(u0)
        @test_throws ArgumentError HaloArrays.interior_face_range(u0, 1)
        u1 = LocalHaloArray(Float64, (4,), 1; boundary_condition=:repeating)
        @test FaceRanges(u1) isa FaceRanges   # halo >= 1 unaffected
    end

    @testset "permutedims/reverse refuse to mislabel the boundary condition" begin
        # Base's generic fallbacks permute/flip the data but copy the BC tuple
        # verbatim — attached to the wrong axes/sides, so the next
        # synchronize_halo! fills ghosts wrong. Refused with an escape hatch.
        u = LocalHaloArray(Float64, (3, 4), 1;
            boundary_condition=((:reflecting, :reflecting), (:periodic, :periodic)))
        v = LocalHaloArray(Float64, (4,), 1; boundary_condition=:periodic)
        t = ThreadedHaloArray(Float64, (2, 2), 1; dims=(2, 1), boundary_condition=:periodic)
        for x in (u, v, t)
            ndims(x) == 2 && @test_throws ArgumentError permutedims(x)
            ndims(x) == 1 && @test_throws ArgumentError permutedims(x)
            @test_throws ArgumentError permutedims(x, reverse(ntuple(identity, ndims(x))))
            @test_throws ArgumentError reverse(x; dims=1)
            @test_throws ArgumentError reverse!(x)
        end
    end

    @testset "interior_faces support a conservative flux update" begin
        u = LocalHaloArray(Int, (4,), 1; boundary_condition=:repeating)
        du = similar(u)

        parent(u) .= [100, 1, 2, 3, 4, 200]
        fill!(parent(du), 0)

        ranges = FaceRanges(u)
        offset = unit_vector(ranges, 1)

        # one loop over every face, scatter onto both cells
        for IL in interior_faces(ranges, 1)
            IR = IL + offset
            flux = parent(u)[IR] - parent(u)[IL]
            parent(du)[IL] -= flux
            parent(du)[IR] += flux
        end

        # interior update is exactly the conservative result …
        @test collect(interior_view(du)) == [-100, 0, 0, -195]
        # … and the two boundary faces also wrote their (harmless) ghost cells
        @test parent(du)[1] == 99
        @test parent(du)[end] == 196
    end

    @testset "colored face ranges support checkerboard update" begin
        u = LocalHaloArray(Int, (4,), 1; boundary_condition=:repeating)
        du = similar(u)

        parent(u) .= [100, 1, 2, 3, 4, 200]
        fill!(parent(du), 0)

        ranges = FaceRanges(u)
        offset = unit_vector(ranges, 1)

        for color in 0:1
            color_touches = zeros(Int, length(parent(u)))
            indices = interior_faces(ranges, 1, color)
            for IL in indices
                IR = IL + offset
                color_touches[IL] += 1   # scatter-both touches both cells
                color_touches[IR] += 1
            end
            # race-free: within one color no cell (interior or ghost) is touched twice
            @test maximum(color_touches) <= 1
            _apply_colored_face_update!(parent(du), parent(u), indices, offset)
        end

        @test collect(interior_view(du)) == [-100, 0, 0, -195]
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

    @testset "CellWindow cell_index and is_cell_index_inbounds" begin
        u = LocalHaloArray(Float64, (4,), 1; boundary_condition=:repeating)
        cr = CellRanges(u)
        region = interior_cell_window(cr)

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
        r2 = interior_cell_window(cr2)

        @test cell_index(r2, (1, 1)) == (2, 2)
        @test cell_index(r2, (3, 3)) == (4, 4)
        @test is_cell_index_inbounds(r2, (2, 2))
        @test !is_cell_index_inbounds(r2, (1, 2))   # ghost in dim 1
        @test !is_cell_index_inbounds(r2, (2, 5))   # out of bounds in dim 2
    end

    @testset "CellCheckerboard cell_index and is_cell_index_inbounds" begin
        # 1-D: 4 owned cells, halo=1 → storage 2:5
        u = LocalHaloArray(Float64, (4,), 1; boundary_condition=:repeating)
        cr = CellRanges(u)

        r0 = interior_cell_window(cr, 0)   # color 0: global-even cell
        r1 = interior_cell_window(cr, 1)   # color 1: global-odd cell

        # global index = storage - halo, so color 0 (global-even) is storage 3,5
        # color 0 maps J=(1,) → storage 3, J=(2,) → storage 5
        @test cell_index(r0, (1,)) == (3,)
        @test cell_index(r0, (2,)) == (5,)
        # color 1 maps J=(1,) → storage 2, J=(2,) → storage 4
        @test cell_index(r1, (1,)) == (2,)
        @test cell_index(r1, (2,)) == (4,)

        # verify each result has the correct global color: mod(sum(I)+parity,2) == color
        for (region, color) in ((r0, 0), (r1, 1))
            for j in 1:region.size[1]
                I = cell_index(region, (j,))
                @test mod(sum(I) + region.parity, 2) == color
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
        r2c0 = interior_cell_window(cr2, 0; compressed_dim=2)

        # launch size: (4, ceil(4/2)) = (4, 2)
        @test r2c0.size == (4, 2)
        @test r2c0.full_size == (4, 4)

        # every reconstructed cell should have color 0: mod(i+j, 2) == 0
        for ji in 1:r2c0.size[1], jj in 1:r2c0.size[2]
            I = cell_index(r2c0, (ji, jj))
            if is_cell_index_inbounds(r2c0, I)
                @test mod(sum(I) + r2c0.parity, 2) == 0
            end
        end
    end

    @testset "linear/slice indexing is refused with instructive errors" begin
        # The indexing contract (see the guide's Indexing section): full-dims
        # global scalar indexing only. Linear indexing and slices point the
        # user at interior_view/gather_haloarray instead of a bare
        # BoundsError / an obscure generic-fallback failure.
        u = LocalHaloArray(Float64, (2, 3), 1; boundary_condition=:periodic)
        interior_view(u) .= 1.0
        t = ThreadedHaloArray(Float64, (1, 3), 1; dims=(2, 1), boundary_condition=:periodic)
        for x in (u, t)
            @test_throws ArgumentError x[3]           # linear indexing
            @test_throws ArgumentError x[:]           # vec slice
            @test_throws ArgumentError x[:, 1]        # column slice
            @test_throws ArgumentError x[1:2, 1]      # range slice
            @test_throws ArgumentError x[[1, 2], 1]   # index-vector slice
            @test_throws ArgumentError x[:, 1] = [1.0, 2.0]
            # the supported scalar forms are unaffected
            @test x[2, 3] isa Float64
            @test x[2, 3, 1] == x[2, 3]               # trailing-1 contract
        end
        # out-of-range scalar indexing keeps throwing BoundsError
        @test_throws BoundsError u[0, 1]

        # isassigned must swallow the refusals (Base only catches BoundsError,
        # so the instructive ArgumentError would crash it on Julia 1.10)
        @test isassigned(u, 2, 3)
        @test !isassigned(u, 3)        # linear-indexing arity → false, not a throw
        @test !isassigned(u, 9, 9)     # out of range

        # the MaybeHaloArray wrapper gets the same instructive slice refusal
        m = MaybeHaloArray(u)
        @test_throws ArgumentError m[:, 1]
        @test_throws ArgumentError m[1:2, 1]
        @test_throws ArgumentError m[:, 1] = [1.0, 2.0]
        @test m[2, 3] == u[2, 3]       # scalar forms unaffected
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

    @testset "mutating drivers return their array on every backend" begin
        # Backend-agnostic chaining relies on `f!(u) === u` — one convention.
        nthreads = max(1, Threads.nthreads())
        us = Any[LocalHaloArray(Float64, (4, 4), 1; boundary_condition=:periodic),
                 ThreadedHaloArray(Float64, (4, 4), 1;
                     dims=(nthreads, 1), boundary_condition=:periodic)]
        for u in us
            @test halo_exchange!(u) === u
            @test start_halo_exchange!(u) === u
            @test finish_halo_exchange!(u) === u
            @test boundary_condition!(u) === u
            @test synchronize_halo!(u) === u
            @test fill_from_global_indices!(I -> Float64(I[1] + I[2]), u) === u
        end
    end

    @testset "Diagonal-of-halo-array mul!/ldiv! (weight preconditioners)" begin
        # OrdinaryDiffEq applies Diagonal(weight::halo array) preconditioners in
        # every implicit+Krylov solve; the generic LinearAlgebra kernels scalar-
        # index global indices (local-only under MPI). These must stay elementwise.
        # 1-D states: OrdinaryDiffEq wraps the (vec'd) weight vector, Diagonal(weight).
        nthreads = max(1, Threads.nthreads())
        for u in Any[LocalHaloArray(Float64, (8,), 1; boundary_condition=:periodic),
                     ThreadedHaloArray(Float64, (4,), 1;
                         dims=(nthreads,), boundary_condition=:periodic)]
            d   = fill_from_global_indices!(I -> 1.0 + 0.1I[1], similar(u))
            b   = fill_from_global_indices!(I -> Float64(I[1]^2), similar(u))
            out = similar(u)
            D   = Diagonal(d)

            @test mul!(out, D, b) === out
            @test collect(out) ≈ collect(d) .* collect(b)

            ref = 2.0 .* collect(d) .* collect(b) .+ 3.0 .* collect(out)
            @test mul!(out, D, b, 2.0, 3.0) === out
            @test collect(out) ≈ ref

            @test ldiv!(out, D, b) === out
            @test collect(out) ≈ collect(b) ./ collect(d)

            copyto!(out, b)
            @test ldiv!(D, out) === out
            @test collect(out) ≈ collect(b) ./ collect(d)
        end
    end

    @testset "ThreadedHaloArray iteration yields values, not indices" begin
        # Regression: iterate delegated to CartesianIndices and returned the
        # indices themselves, so collect/comprehensions silently gave 1,2,3,…
        u = ThreadedHaloArray(Float64, (4,), 1; dims=(max(1, Threads.nthreads()),),
            boundary_condition=:periodic)
        fill_from_global_indices!(I -> Float64(I[1]^2), u)
        n = length(u)
        @test collect(u) == Float64.((1:n) .^ 2)
        @test [x for x in u] == Float64.((1:n) .^ 2)
        @test first(u) == 1.0
    end

end
