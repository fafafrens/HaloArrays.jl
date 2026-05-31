using Test
using HaloArrays

@testset "AbstractArray{<:AbstractSingleHaloArray} fallbacks" begin

    # Build a plain Vector of LocalHaloArrays (NOT wrapped in ArrayOfHaloArray)
    arrays = [
        LocalHaloArray(Float64, (4, 4), 1; boundary_condition=:periodic),
        LocalHaloArray(Float64, (4, 4), 1; boundary_condition=:periodic),
    ]
    for a in arrays
        interior_view(a) .= 1.0
    end

    @testset "halo_width" begin
        @test halo_width(arrays) == 1
    end

    @testset "CellRanges fallback" begin
        cr = CellRanges(arrays)
        cells = get_owned_cells(cr)
        @test size(cells) == (4, 4)
        # colored subranges cover all owned cells
        c0 = get_colored_owned_cell_ranges(cr, 0)
        c1 = get_colored_owned_cell_ranges(cr, 1)
        @test sum(length, c0) + sum(length, c1) == 16
    end

    @testset "FaceRanges fallback" begin
        fr = FaceRanges(arrays)
        # left/right faces are slabs of width 1, internal face is (3,4)
        @test length(get_left_face(fr, 1))  == 4
        @test length(get_right_face(fr, 1)) == 4
        @test size(get_internal_face(fr))   == (3, 3)
        @test get_unit_vector(fr, 1) == CartesianIndex(1, 0)
        @test get_unit_vector(fr, 2) == CartesianIndex(0, 1)
    end
end

@testset "ArrayOfHaloArray tile fallbacks" begin
    nthreads  = max(1, Threads.nthreads())
    tsz       = (8,)       # renamed to avoid shadowing tile_size()
    tile_dims = (nthreads,)

    vel = ArrayOfHaloArray(ThreadedHaloArray, Float64, (2,), tsz, 1;
        dims=tile_dims, boundary_condition=:periodic)

    @testset "tile_count" begin
        @test tile_count(vel) == tile_count(vel[1])
    end

    @testset "tile_size" begin
        @test tile_size(vel) == tile_size(vel[1])
    end

    @testset "tile_coordinates" begin
        for t in 1:tile_count(vel)
            @test tile_coordinates(vel, t) == tile_coordinates(vel[1], t)
        end
    end

    @testset "neighbor_tile_id" begin
        tc = tile_count(vel)
        if tc >= 2
            @test neighbor_tile_id(vel, 1, 1, 2) == neighbor_tile_id(vel[1], 1, 1, 2)
        end
    end

    @testset "CellRanges on ArrayOfHaloArray" begin
        cr = CellRanges(vel)
        @test size(get_owned_cells(cr)) == tsz   # per-tile range (interior_range of one tile)
    end
end

@testset "MultiHaloArray neighbor_tile_id fallback" begin
    nthreads = max(1, Threads.nthreads())
    state = ThreadedMultiHaloArray(Float64, (8,), 1;
        dims=(nthreads,),
        boundary_conditions=(
            a=((Repeating(), Repeating()),),
            b=((Repeating(), Repeating()),),
        ))

    @testset "tile_count" begin
        @test tile_count(state) == nthreads
    end

    @testset "neighbor_tile_id" begin
        tc = tile_count(state)
        if tc >= 2
            @test neighbor_tile_id(state, 1, 1, 2) ==
                  neighbor_tile_id(state.arrays.a, 1, 1, 2)
        end
    end
end

@testset "AbstractArray{<:AbstractSingleHaloArray} tile fallbacks" begin
    nthreads = max(1, Threads.nthreads())
    arrays = [
        ThreadedHaloArray(Float64, (8,), 1; dims=(nthreads,), boundary_condition=:periodic),
        ThreadedHaloArray(Float64, (8,), 1; dims=(nthreads,), boundary_condition=:periodic),
    ]

    @testset "tile_count" begin
        @test tile_count(arrays) == nthreads
    end

    @testset "tile_size" begin
        @test tile_size(arrays) == (8,)
    end

    @testset "tile_coordinates" begin
        @test tile_coordinates(arrays, 1) == tile_coordinates(arrays[1], 1)
    end
end
