using Test
using HaloArrays

# Backend-preserving dims= reductions (serial backends + collections; the MPI
# HaloArray path is covered by test_mpi_reduction_plan.jl / test_reduce.jl):
# each backend returns its own kind of reduced array with the reduced
# dimensions dropped, carrying the kept dimensions' halo width and boundary
# conditions. Only distributed results are MaybeHaloArray-wrapped;
# is_active/interior_view/free! behave uniformly on every return kind.

_fill_global!(u, g) = (for I in CartesianIndices(axes(u)); u[Tuple(I)...] = g(Tuple(I)); end; u)

@testset "dims= reductions: serial backends + collections" begin
    g(I) = Float64(I[1] + 100 * I[2])
    GX, GY = 4, 6
    ref_d2 = [sum(g((i, j)) for j in 1:GY) for i in 1:GX]     # reduce dim 2
    ref_d1 = [sum(g((i, j)) for i in 1:GX) for j in 1:GY]     # reduce dim 1
    refmax_d2 = [maximum(g((i, j)) for j in 1:GY) for i in 1:GX]

    @testset "LocalHaloArray" begin
        lu = LocalHaloArray(Float64, (GX, GY), 2; boundary_condition=:periodic)
        fill_from_global_indices!(g, lu)

        r = sum(lu; dims=2)
        @test r isa LocalHaloArray                            # bare, backend kept
        @test is_active(r)
        @test interior_size(r) == (GX,)                       # dims dropped
        @test halo_width(r) == 2                              # kept halo width
        @test vec(collect(interior_view(r))) ≈ ref_d2
        @test vec(collect(interior_view(sum(lu; dims=1)))) ≈ ref_d1
        @test vec(collect(interior_view(maximum(lu; dims=2)))) ≈ refmax_d2

        # Explicit form agrees; free! is a safe no-op on serial results.
        re = mapreduce_haloarray_dims(abs2, +, lu, 2)
        @test vec(collect(interior_view(re))) ≈
              [sum(abs2(g((i, j))) for j in 1:GY) for i in 1:GX]
        @test free!(r) === r

        # dims=: stays the scalar whole-array reduction.
        @test sum(lu; dims=:) ≈ sum(lu)
    end

    @testset "ThreadedHaloArray (tiles along kept AND removed dims)" begin
        for layout in ((2, 1), (1, 2), (2, 2), (1, 3), (2, 3))
            ts = (GX ÷ layout[1], GY ÷ layout[2])
            all(ts .> 0) || continue
            tu = ThreadedHaloArray(Float64, ts, 1; dims=layout, boundary_condition=:periodic)
            _fill_global!(tu, g)

            for (d, keep, ref) in ((1, 2, ref_d1), (2, 1, ref_d2))
                r = sum(tu; dims=d)
                @test r isa ThreadedHaloArray                 # backend kept
                @test r.topology.dims == (layout[keep],)      # kept-dims tile layout
                @test tile_size(r) == (ts[keep],)
                @test thread_backend(r) === thread_backend(tu)
                @test vec(collect(r)) ≈ ref
                @test free!(r) === r
            end
            @test vec(collect(maximum(tu; dims=2))) ≈ refmax_d2

            # Folds with dims= on a tiled array are rejected (cross-tile
            # combine would be order-mixing), not silently wrong.
            @test_throws ArgumentError mapfoldl(identity, +, tu; dims=1)
        end
    end

    @testset "collections (Local- and Threaded-backed, both kinds)" begin
        lu = LocalHaloArray(Float64, (GX, GY), 1; boundary_condition=:periodic)
        fill_from_global_indices!(g, lu)
        tu = ThreadedHaloArray(Float64, (GX ÷ 2, GY), 1; dims=(2, 1), boundary_condition=:periodic)
        _fill_global!(tu, g)

        m = MultiHaloArray((; a=lu, b=copy(lu)))
        rm = sum(m; dims=2)
        @test rm isa MultiHaloArray                           # bare, same kind
        @test is_active(rm)
        @test vec(collect(interior_view(rm.arrays.a))) ≈ ref_d2
        @test vec(collect(interior_view(rm.arrays.b))) ≈ ref_d2
        @test free!(rm) === rm

        aoh = ArrayOfHaloArray([tu, copy(tu)])
        ra = sum(aoh; dims=2)
        @test ra isa ArrayOfHaloArray
        for fld in HaloArrays._fields(ra)
            @test fld isa ThreadedHaloArray
            @test vec(collect(fld)) ≈ ref_d2
        end

        # Explicit collection form agrees with the kwarg form.
        rme = mapreduce_haloarray_dims(identity, +, m, 2)
        @test vec(collect(interior_view(rme.arrays.a))) ≈ ref_d2
    end

    @testset "serial DimReductionPlan (backend-uniform hot loop)" begin
        lu = LocalHaloArray(Float64, (GX, GY), 1; boundary_condition=:periodic)
        fill_from_global_indices!(g, lu)
        tu = ThreadedHaloArray(Float64, (GX ÷ 2, GY), 1; dims=(2, 1), boundary_condition=:periodic)
        _fill_global!(tu, g)

        for u in (lu, tu)
            plan = DimReductionPlan(u, 2)
            @test plan isa DimReductionPlan
            r1 = reduce!(plan, identity, +, u)
            @test vec(collect(u isa ThreadedHaloArray ? r1 : interior_view(r1))) ≈ ref_d2
            r2 = reduce!(plan, identity, +, u)
            @test r2 === r1                                  # preallocated, overwritten
            @test reduce!(plan, identity, max, u) === r1     # geometry-only: any op
            @test free!(plan) === plan                       # no-op, stays usable
            @test reduce!(plan, identity, +, u) === r1
        end

        # geometry validation
        plan = DimReductionPlan(lu, 2)
        other = LocalHaloArray(Float64, (GX, GY + 2), 1; boundary_condition=:periodic)
        @test_throws DimensionMismatch reduce!(plan, identity, +, other)

        # a plan's eltype is fixed at construction: promoting ops error clearly,
        # and the output_eltype keyword opts in to the promoted type
        bl = LocalHaloArray(Bool, (GX, GY), 1; boundary_condition=:repeating)
        interior_view(bl) .= true
        bplan = DimReductionPlan(bl, 2)
        @test_throws ArgumentError reduce!(bplan, identity, +, bl)
        @test reduce!(bplan, identity, max, bl) isa LocalHaloArray   # non-promoting op fine
        bplan2 = DimReductionPlan(bl, 2; output_eltype=Int)
        @test vec(collect(interior_view(reduce!(bplan2, identity, +, bl)))) == fill(GY, GX)
    end

    @testset "one-shot Bool reductions promote like Base" begin
        bl = LocalHaloArray(Bool, (GX, GY), 1; boundary_condition=:repeating)
        interior_view(bl) .= true
        rb = sum(bl; dims=2)
        @test eltype(rb) == Int
        @test vec(collect(interior_view(rb))) == fill(GY, GX)

        bt = ThreadedHaloArray(Bool, (GX ÷ 2, GY), 1; dims=(2, 1), boundary_condition=:repeating)
        _fill_global!(bt, I -> true)
        rbt = sum(bt; dims=2)
        @test eltype(rbt) == Int
        @test vec(collect(rbt)) == fill(GY, GX)
    end

    @testset "MaybeHaloArray accessor pass-through" begin
        lu = LocalHaloArray(Float64, (GX, GY), 1; boundary_condition=:periodic)
        fill_from_global_indices!(g, lu)
        ma = HaloArrays.active(lu)
        @test interior_view(ma) == interior_view(lu)
        @test interior_range(ma) == interior_range(lu)
        mi = HaloArrays.inactive(lu)
        @test_throws ErrorException interior_view(mi)
        @test_throws ErrorException interior_range(mi)   # guarded like interior_view
    end

    @testset "argument guards" begin
        lu = LocalHaloArray(Float64, (GX, GY), 1; boundary_condition=:periodic)
        m  = MultiHaloArray((; a=lu))
        @test_throws ArgumentError sum(lu; dims=(1, 2))          # scalar via dims
        @test_throws ArgumentError sum(lu; dims=3)               # out of range
        @test_throws ArgumentError sum(lu; dims=2, init=0.0)     # init rejected
        @test_throws ArgumentError sum(m; dims=2, init=0.0)
        @test_throws ArgumentError mapreduce(tuple, +, lu, lu; dims=2)  # multi-array

        # Order-sensitive folds reject dims= on EVERY backend with the clean
        # ArgumentError (not Base's obscure "no method matching mapfoldl(…;dims)").
        tu = ThreadedHaloArray(Float64, (GX ÷ 2, GY), 1; dims=(2, 1), boundary_condition=:periodic)
        for u in (lu, tu)
            @test_throws ArgumentError mapfoldl(identity, +, u; dims=2)
            @test_throws ArgumentError mapfoldr(identity, +, u; dims=1)
        end
    end
end
