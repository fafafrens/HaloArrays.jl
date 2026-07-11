using MPI
using Test
using HaloArrays

# DimReductionPlan: the reusable (Cart_sub-based) dims-reduction must agree with
# mapreduce_haloarray_dims (the one-off Comm_split path) and with a serial
# reference, across: kept-dim split (all ranks active), reduced-dim split (only
# the coordinate-0 slice active), plan reuse after the data changes, a second op
# through the same plan, multi-dim removal in 3-D, and the free!/error paths.

@testset "DimReductionPlan (Cart_sub reusable reduction)" begin
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nr   = MPI.Comm_size(comm)
    @test nr > 1

    f(I) = I[1] + 100 * I[2]
    GX = 8 * nr
    GY = 12
    topo = CartesianTopology(comm, (nr, 1); periodic=(true, true))
    bc   = ((Periodic(), Periodic()), (Periodic(), Periodic()))
    u = HaloArray(Float64, (GX ÷ nr, GY), 1, topo; boundary_condition=bc)
    fill_from_global_indices!(f, u)

    @testset "reduce kept-split dim 2 (every rank active)" begin
        plan   = DimReductionPlan(u, 2)
        r_plan = reduce!(plan, identity, +, u)
        r_old  = mapreduce_haloarray_dims(identity, +, u, 2)

        @test r_plan isa MaybeHaloArray
        @test is_active(r_plan)
        @test is_active(r_plan) == is_active(r_old)
        ref = Float64[sum(f((i, j)) for j in 1:GY) for i in 1:GX]
        i0 = topo.cart_coords[1] * (GX ÷ nr)
        @test vec(collect(interior_view(parent(r_plan)))) ≈ ref[i0+1:i0+GX÷nr]
        @test collect(interior_view(parent(r_plan))) ≈ collect(interior_view(parent(r_old)))

        # Same plan, different op: max along dim 2.
        r_max     = reduce!(plan, identity, max, u)
        r_max_old = mapreduce_haloarray_dims(identity, max, u, 2)
        @test collect(interior_view(parent(r_max))) ≈ collect(interior_view(parent(r_max_old)))

        # Reuse after the data changes: output must refresh, in place.
        g(I) = 3 * f(I)
        fill_from_global_indices!(g, u)
        r2 = reduce!(plan, identity, +, u)
        @test r2 === r_plan            # preallocated output, overwritten
        @test vec(collect(interior_view(parent(r2)))) ≈ 3 .* ref[i0+1:i0+GX÷nr]
        fill_from_global_indices!(f, u)   # restore for the next testsets

        # Freed plan: idempotent free!, then reduce! must throw.
        free!(plan)
        free!(plan)
        @test_throws ArgumentError reduce!(plan, identity, +, u)
    end

    @testset "reduce rank-split dim 1 (only slice coordinate 0 active)" begin
        plan   = DimReductionPlan(u, 1)
        r_plan = reduce!(plan, identity, +, u)
        r_old  = mapreduce_haloarray_dims(identity, +, u, 1)

        @test is_active(r_plan) == (topo.cart_coords[1] == 0)
        @test is_active(r_plan) == is_active(r_old)
        if is_active(r_plan)
            ref = Float64[sum(f((i, j)) for i in 1:GX) for j in 1:GY]
            @test vec(collect(interior_view(parent(r_plan)))) ≈ ref
            @test collect(interior_view(parent(r_plan))) ≈ collect(interior_view(parent(r_old)))
        end
        free!(plan)
    end

    @testset "sum/maximum dims= kwarg (transient plan per call)" begin
        r_sum = sum(u; dims=2)
        r_ref = mapreduce_haloarray_dims(identity, +, u, 2)
        @test r_sum isa MaybeHaloArray
        @test is_active(r_sum) == is_active(r_ref)
        if is_active(r_sum)
            @test collect(interior_view(parent(r_sum))) ≈ collect(interior_view(parent(r_ref)))
        end

        # Fresh result per call: earlier results must not be overwritten.
        a = sum(u; dims=2)
        fill_from_global_indices!(I -> 2 * f(I), u)
        b = sum(u; dims=2)
        @test a !== b
        if is_active(a)
            @test collect(interior_view(parent(b))) ≈ 2 .* collect(interior_view(parent(a)))
        end
        fill_from_global_indices!(f, u)

        r_max  = maximum(u; dims=1)
        r_maxr = mapreduce_haloarray_dims(identity, max, u, 1)
        @test is_active(r_max) == is_active(r_maxr)
        if is_active(r_max)
            @test collect(interior_view(parent(r_max))) ≈ collect(interior_view(parent(r_maxr)))
        end

        r_f  = sum(abs2, u; dims=2)
        r_fr = mapreduce_haloarray_dims(abs2, +, u, 2)
        if is_active(r_f)
            @test collect(interior_view(parent(r_f))) ≈ collect(interior_view(parent(r_fr)))
        end

        # Order-sensitive folds and init stay rejected; scalar dims too.
        @test_throws ArgumentError mapfoldl(identity, +, u; dims=2)
        @test_throws ArgumentError sum(u; dims=2, init=0.0)
        @test_throws ArgumentError sum(u; dims=(1, 2))

        # dims=: is the whole-array scalar reduction, unchanged.
        @test sum(u; dims=:) ≈ sum(u)

        # Each result owns its sub-communicator: free! releases it (idempotent,
        # data stays readable), keeping communicator use bounded in loops.
        for r in (r_sum, r_ref, a, b, r_max, r_maxr, r_f, r_fr)
            free!(r)
            free!(r)
        end
        if is_active(r_sum)
            @test parent(r_sum).topology.cart_comm == MPI.COMM_NULL
            @test collect(interior_view(parent(r_sum))) ≈ collect(interior_view(parent(r_ref)))
        end
    end

    @testset "one-shot promotion on MPI (Bool counts like Base)" begin
        ntopo = CartesianTopology(comm, (nr, 1); periodic=(false, false))
        rbc   = ((Repeating(), Repeating()), (Repeating(), Repeating()))
        bh = HaloArray(Bool, (4, 3), 1, ntopo; boundary_condition=rbc)
        interior_view(bh) .= true
        rb = sum(bh; dims=2)
        @test eltype(rb) == Int
        if is_active(rb)
            @test vec(collect(interior_view(rb))) == fill(3, 4)
        end
        free!(rb)
    end

    @testset "dims= on an MPI-backed MultiHaloArray" begin
        m     = MultiHaloArray((; p=u, q=copy(u)))
        rm    = sum(m; dims=2)
        r_ref = mapreduce_haloarray_dims(identity, +, u, 2)
        @test rm isa MaybeHaloArray
        @test is_active(rm) == is_active(r_ref)
        if is_active(rm)
            @test collect(interior_view(parent(rm).arrays.p)) ≈ collect(interior_view(parent(r_ref)))
            @test collect(interior_view(parent(rm).arrays.q)) ≈ collect(interior_view(parent(r_ref)))
        end
        foreach(free!, (rm, r_ref))   # collection free! walks every field
    end

    @testset "3-D, remove two dims at once" begin
        topo3 = CartesianTopology(comm, (nr, 1, 1); periodic=(true, true, true))
        bc3   = ntuple(_ -> (Periodic(), Periodic()), 3)
        h(I)  = I[1] + 10 * I[2] + 100 * I[3]
        u3 = HaloArray(Float64, (4, 5, 6), 1, topo3; boundary_condition=bc3)
        fill_from_global_indices!(h, u3)

        plan   = DimReductionPlan(u3, (2, 3))
        r_plan = reduce!(plan, abs2, +, u3)
        r_old  = mapreduce_haloarray_dims(abs2, +, u3, (2, 3))
        @test is_active(r_plan) == is_active(r_old)
        if is_active(r_plan)
            @test collect(interior_view(parent(r_plan))) ≈ collect(interior_view(parent(r_old)))
        end
        free!(plan)
    end

    @testset "argument validation" begin
        @test_throws ArgumentError DimReductionPlan(u, (1, 2))   # scalar reduction
        @test_throws ArgumentError DimReductionPlan(u, 3)        # out of range

        plan = DimReductionPlan(u, 2)
        u_other = HaloArray(Float64, (GX ÷ nr, GY + 2), 1, topo; boundary_condition=bc)
        @test_throws DimensionMismatch reduce!(plan, identity, +, u_other)
        free!(plan)
    end

    MPI.Barrier(comm)
end
