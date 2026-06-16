using Test
using HaloArrays
using LinearAlgebra: dot, norm

@testset "Local and threaded reductions" begin
    local_u = LocalHaloArray(Int, (4,), 1; boundary_condition=:repeating)
    local_v = similar(local_u)
    interior_view(local_u) .= [1, 2, 3, 4]
    interior_view(local_v) .= [10, 20, 30, 40]

    @test mapreduce(identity, +, local_u) == 10
    @test reduce(+, local_u) == 10
    @test sum(local_u) == 10
    # dot is a two-argument reduction over owned cells
    @test dot(local_u, local_v) == dot([1, 2, 3, 4], [10, 20, 30, 40])   # 300
    @test dot(local_u, local_u) == sum(abs2, interior_view(local_u))
    @test sum(abs2, local_u) == 30

    # `dot` forwards to the interior dot, so its allocation must NOT scale with
    # the interior size (the old `mapreduce(dot, +, x, y)` materialized a full
    # interior-sized array — O(N) per call, in every Krylov inner loop).
    dot_alloc(n) = (a = LocalHaloArray(Float64, (n,), 1; boundary_condition=:periodic);
                    b = similar(a); fill!(a, 1.0); fill!(b, 2.0);
                    dot(a, b); @allocated dot(a, b))
    @test dot_alloc(16) == dot_alloc(16_000)
    @test maximum(local_u) == 4
    @test minimum(local_u) == 1
    @test mapfoldl(abs2, +, local_u) == 30
    @test mapfoldr(abs2, +, local_u) == 30
    @test mapreduce((x, y) -> x * y, +, local_u, local_v) == 300
    @test mapreduce(x -> x[1] * x[2], +, zip(local_u, local_v)) == 300
    @test all(x -> x > 0, local_u)
    @test any(x -> x == 3, local_u)

    maybe_local = MaybeHaloArray(local_u)
    @test sum(maybe_local) == 10
    @test maximum(maybe_local) == 4
    @test minimum(maybe_local) == 1

    local_fields = MultiHaloArray((; u=local_u, v=local_v))
    @test mapreduce(identity, +, local_fields) == 110
    @test sum(local_fields) == 110
    @test maximum(local_fields) == 40
    @test minimum(local_fields) == 1
    @test mapfoldl(identity, +, local_fields) == 110
    @test mapfoldr(identity, +, local_fields) == 110
    @test all(x -> x > 0, local_fields)
    @test any(x -> x == 40, local_fields)

    local_array_fields = ArrayOfHaloArray([local_u, local_v])
    @test mapreduce(identity, +, local_array_fields) == 110
    @test sum(local_array_fields) == 110
    @test maximum(local_array_fields) == 40
    @test minimum(local_array_fields) == 1
    @test all(x -> x > 0, local_array_fields)
    @test any(x -> x == 40, local_array_fields)

    threaded_u = ThreadedHaloArray(Int, (3,), 1; dims=(2,), boundary_condition=:repeating)
    threaded_v = similar(threaded_u)
    interior_view(threaded_u, 1) .= [1, 2, 3]
    interior_view(threaded_u, 2) .= [4, 5, 6]
    interior_view(threaded_v, 1) .= [10, 20, 30]
    interior_view(threaded_v, 2) .= [40, 50, 60]

    @test mapreduce(identity, +, threaded_u) == 21
    @test @inferred(mapreduce(identity, +, threaded_u)) == 21
    @test reduce(+, threaded_u) == 21
    @test sum(threaded_u) == 21
    @test sum(abs2, threaded_u) == 91
    @test maximum(threaded_u) == 6
    @test minimum(threaded_u) == 1
    @test mapfoldl(abs2, +, threaded_u) == 91
    @test mapfoldr(abs2, +, threaded_u) == 91
    @test mapreduce((x, y) -> x * y, +, threaded_u, threaded_v) == 910
    @test @inferred(mapreduce((x, y) -> x * y, +, threaded_u, threaded_v)) == 910
    @test mapreduce(x -> x[1] * x[2], +, zip(threaded_u, threaded_v)) == 910
    @test all(x -> x > 0, threaded_u)
    @test any(x -> x == 6, threaded_u)

    maybe_threaded = MaybeHaloArray(threaded_u)
    @test sum(maybe_threaded) == 21
    @test maximum(maybe_threaded) == 6
    @test minimum(maybe_threaded) == 1

    threaded_fields = MultiHaloArray((; u=threaded_u, v=threaded_v))
    @test mapreduce(identity, +, threaded_fields) == 231
    @test @inferred(mapreduce(identity, +, threaded_fields)) == 231
    @test sum(threaded_fields) == 231
    @test maximum(threaded_fields) == 60
    @test minimum(threaded_fields) == 1
    @test mapfoldl(identity, +, threaded_fields) == 231
    @test mapfoldr(identity, +, threaded_fields) == 231
    @test all(x -> x > 0, threaded_fields)
    @test any(x -> x == 60, threaded_fields)

    threaded_array_fields = ArrayOfHaloArray([threaded_u, threaded_v])
    @test mapreduce(identity, +, threaded_array_fields) == 231
    @test sum(threaded_array_fields) == 231
    @test maximum(threaded_array_fields) == 60
    @test minimum(threaded_array_fields) == 1
    @test all(x -> x > 0, threaded_array_fields)
    @test any(x -> x == 60, threaded_array_fields)

    @test mapreduce(identity, +, threaded_u; dims=1)[] == 21

    @testset "tile_mapreduce (cross-tile combine)" begin
        # The per-tile results are combined with `op` via the backend's
        # tile_mapreduce; it must match a plain mapreduce for any backend.
        for backend in (OhMyThreadsBackend(), SerialBackend())
            @test tile_mapreduce(backend, identity, +, [1, 2, 3]) == 6
            @test tile_mapreduce(backend, identity, *, [3, 4]) == 12
            @test tile_mapreduce(backend, identity, max, [3, 1, 4]) == 4
            @test tile_mapreduce(backend, identity, +, [7]) == 7   # single tile
        end
    end

    @testset "in-place BLAS-1 vector ops (axpy!/axpby!/lmul!/rmul!)" begin
        # These let Krylov.jl and other LinearAlgebra iterative solvers run on
        # halo arrays without scalar-indexing fallbacks; interior-only via broadcast.
        mk(v) = (a = LocalHaloArray(Float64, (4,), 1; boundary_condition=:periodic);
                 interior_view(a) .= v; a)
        x = mk([1.0, 2, 3, 4]); y = mk([10.0, 20, 30, 40])
        axpy!(2.0, x, y)
        @test collect(interior_view(y)) == [12.0, 24, 36, 48]
        @test collect(interior_view(x)) == [1.0, 2, 3, 4]           # x untouched
        lmul!(0.5, x)
        @test collect(interior_view(x)) == [0.5, 1.0, 1.5, 2.0]
        rmul!(x, 4.0)
        @test collect(interior_view(x)) == [2.0, 4.0, 6.0, 8.0]
        z = mk([1.0, 1, 1, 1]); axpby!(2.0, mk([1.0, 2, 3, 4]), 10.0, z)
        @test collect(interior_view(z)) == [12.0, 14, 16, 18]

        # threaded backend: same ops, per tile
        tx = ThreadedHaloArray(Float64, (3,), 1; dims=(2,), boundary_condition=:repeating)
        ty = similar(tx)
        for t in 1:tile_count(tx); interior_view(tx, t) .= 1.0; interior_view(ty, t) .= 2.0; end
        axpy!(3.0, tx, ty)
        @test all(==(5.0), interior_view(ty, 1)) && all(==(5.0), interior_view(ty, 2))
    end

    @testset "swap! + Givens/Householder (swap!/rotate!/reflect!)" begin
        # Complete the BLAS-1 surface (SSWAP, SROT). All elementwise → must agree
        # cell-for-cell with the same op on the collected interior vectors.
        mk(v) = (a = LocalHaloArray(Float64, (4,), 1; boundary_condition=:periodic);
                 interior_view(a) .= v; a)

        # swap!
        x = mk([1.0, 2, 3, 4]); y = mk([10.0, 20, 30, 40])
        swap!(x, y)
        @test collect(interior_view(x)) == [10.0, 20, 30, 40]
        @test collect(interior_view(y)) == [1.0, 2, 3, 4]

        # rotate! / reflect! vs the LinearAlgebra reference on plain vectors
        c, s = 0.6, 0.8                                   # c^2 + s^2 = 1
        for op in (rotate!, reflect!)
            x = mk([1.0, 2, 3, 4]); y = mk([10.0, 20, 30, 40])
            xr = collect(interior_view(x)); yr = collect(interior_view(y))
            op(x, y, c, s); op(xr, yr, c, s)
            @test collect(interior_view(x)) ≈ xr
            @test collect(interior_view(y)) ≈ yr
        end
        # rotate! is an isometry: it preserves the (global) norm of the pair
        x = mk([1.0, 2, 3, 4]); y = mk([10.0, 20, 30, 40])
        n0 = sqrt(norm(x)^2 + norm(y)^2)
        rotate!(x, y, c, s)
        @test sqrt(norm(x)^2 + norm(y)^2) ≈ n0

        # threaded backend
        tx = ThreadedHaloArray(Float64, (3,), 1; dims=(2,), boundary_condition=:repeating)
        ty = similar(tx)
        for t in 1:tile_count(tx); interior_view(tx, t) .= 1.0; interior_view(ty, t) .= 2.0; end
        rotate!(tx, ty, c, s)
        @test all(≈(c * 1.0 + s * 2.0), interior_view(tx, 1))
        @test all(≈(c * 2.0 - s * 1.0), interior_view(ty, 2))
        swap!(tx, ty)
        @test all(≈(c * 2.0 - s * 1.0), interior_view(tx, 1))

        # field collection: delegators apply the single-array kernel per field
        cx = LocalMultiHaloArray((; u = mk([1.0, 2, 3, 4]), v = mk([5.0, 6, 7, 8])))
        cy = LocalMultiHaloArray((; u = mk([10.0, 20, 30, 40]), v = mk([50.0, 60, 70, 80])))
        rotate!(cx, cy, c, s)
        @test collect(interior_view(cx.u)) ≈ c .* [1.0, 2, 3, 4] .+ s .* [10.0, 20, 30, 40]
        @test collect(interior_view(cy.v)) ≈ c .* [50.0, 60, 70, 80] .- s .* [5.0, 6, 7, 8]

        # No temporary vector: the per-call allocation is independent of the
        # vector length (0 on recent Julia, a small constant on some). A regression
        # to a temp-vector form would scale with N — that's what this guards. (An
        # exact `== 0` is too brittle across Julia versions.)
        function alloc_per_call(op!, n)
            a = LocalHaloArray(Float64, (n,), 1; boundary_condition=:periodic)
            b = LocalHaloArray(Float64, (n,), 1; boundary_condition=:periodic)
            op!(a, b)                        # warm up / compile
            return @allocated op!(a, b)
        end
        @test alloc_per_call((a, b) -> rotate!(a, b, c, s), 8) ==
              alloc_per_call((a, b) -> rotate!(a, b, c, s), 800)
        @test alloc_per_call(swap!, 8) == alloc_per_call(swap!, 800)
        @test alloc_per_call((a, b) -> reflect!(a, b, c, s), 8) ==
              alloc_per_call((a, b) -> reflect!(a, b, c, s), 800)
    end
end
