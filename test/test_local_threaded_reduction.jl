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
    # `norm` on a collection combines per-field norms (each already global); it must
    # NOT fall through to Base's generic `norm`, which iterates the collection's
    # scalar AbstractArray interface and allocates O(N) per call (Krylov hot path).
    @test norm(local_fields) ≈ sqrt(sum(abs2, [1, 2, 3, 4, 10, 20, 30, 40]))   # √3030
    @test norm(local_fields, Inf) == 40
    @test norm(local_fields, 1) == sum(abs, [1, 2, 3, 4, 10, 20, 30, 40])
    norm_alloc(n) = (a = LocalHaloArray(Float64, (n,), 1; boundary_condition=:periodic);
                     b = similar(a); fill!(a, 1.0); fill!(b, 2.0);
                     c = MultiHaloArray((; a, b)); norm(c); @allocated norm(c))
    @test norm_alloc(16) == norm_alloc(16_000)
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
    # Single-collection reductions fold over the fields directly (no intermediate
    # results container), so allocation is independent of the field count for BOTH
    # the array-backed ArrayOfHaloArray and the tuple-backed MultiHaloArray — the
    # old `map(eachfield)` materialized an O(#fields) Vector for the array case.
    aoha_alloc(n) = (fs = [LocalHaloArray(Float64, (8,), 1; boundary_condition=:periodic) for _ in 1:n];
                     for a in fs; fill!(a, 1.0); end;
                     c = ArrayOfHaloArray(fs); sum(c); @allocated sum(c))
    @test aoha_alloc(2) == aoha_alloc(16)

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

    # Reducing along every dimension via dims= is rejected on all backends
    # (use the scalar reduction); the sum itself is the no-dims form.
    @test_throws ArgumentError mapreduce(identity, +, threaded_u; dims=1)
    @test sum(threaded_u) == 21

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

        # On a dense Array parent these take a contiguous @simd path (not the strided
        # interior-view broadcast), so per-call allocation must be independent of the
        # vector length — a regression to a temp-allocating form would scale with N.
        blas1_alloc(op!, n) = (a = LocalHaloArray(Float64, (n,), 1; boundary_condition=:periodic);
                               b = LocalHaloArray(Float64, (n,), 1; boundary_condition=:periodic);
                               fill!(a, 1.0); fill!(b, 2.0); op!(a, b); @allocated op!(a, b))
        @test blas1_alloc((a, b) -> axpy!(2.0, a, b), 8)  == blas1_alloc((a, b) -> axpy!(2.0, a, b), 8000)
        @test blas1_alloc((a, b) -> axpby!(2.0, a, 0.5, b), 8) == blas1_alloc((a, b) -> axpby!(2.0, a, 0.5, b), 8000)
        @test blas1_alloc((a, _) -> rmul!(a, 2.0), 8)     == blas1_alloc((a, _) -> rmul!(a, 2.0), 8000)

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

        # MaybeHaloArray: BLAS-1 forwards to the inner array (completes the surface
        # alongside dot/norm/axpy!). The wrappers alias the inner arrays, so the
        # mutation is observable through them.
        ax = mk([1.0, 2, 3, 4]); ay = mk([10.0, 20, 30, 40])
        swap!(MaybeHaloArray(ax), MaybeHaloArray(ay))
        @test collect(interior_view(ax)) == [10.0, 20, 30, 40]
        @test collect(interior_view(ay)) == [1.0, 2, 3, 4]
        bx = mk([1.0, 2, 3, 4]); by = mk([10.0, 20, 30, 40])
        rotate!(MaybeHaloArray(bx), MaybeHaloArray(by), 0.6, 0.8)
        @test collect(interior_view(bx)) ≈ 0.6 .* [1.0, 2, 3, 4] .+ 0.8 .* [10.0, 20, 30, 40]
        cx = mk([1.0, 2, 3, 4]); cy = mk([10.0, 20, 30, 40])
        reflect!(MaybeHaloArray(cx), MaybeHaloArray(cy), 0.6, 0.8)
        @test collect(interior_view(cx)) ≈ 0.6 .* [1.0, 2, 3, 4] .+ 0.8 .* [10.0, 20, 30, 40]

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

    @testset "contiguous-aware interior sum/dot/norm (SIMD over padded parent)" begin
        # sum/dot/norm reduce over the raw `parent` (looping trailing cartesian
        # indices, @simd on the contiguous leading dim) instead of the strided
        # interior_view — ~5× faster. This guards the `p[i, J]` index logic across
        # shapes/eltypes the 1-D Int tests above don't cover: non-square 2-D, 3-D,
        # and complex (dot conjugates the first argument).
        for (T, dims) in ((Float64, (31, 17)), (Float64, (8, 9, 10)),
                          (Float64, (50,)), (ComplexF64, (16, 16)))
            u = LocalHaloArray(T, dims, 1; boundary_condition=:periodic)
            v = LocalHaloArray(T, dims, 1; boundary_condition=:periodic)
            interior_view(u) .= rand(T, dims)
            interior_view(v) .= rand(T, dims)
            iu = vec(collect(interior_view(u)))
            iv = vec(collect(interior_view(v)))
            @test sum(u)    ≈ sum(iu)
            @test dot(u, v) ≈ dot(iu, iv)          # conj(iu)·iv reference
            @test norm(u)   ≈ norm(iu)
            @test norm(u, 1)   ≈ norm(iu, 1)
            @test norm(u, Inf) ≈ norm(iu, Inf)
        end
    end

    # The Local (single-block) path must stay allocation-free: the tile drivers
    # take closures, and a regression (e.g. a Core.Box from a reassigned capture,
    # or a lost @inline) would show up as heap allocations on the Krylov hot path.
    # Older Julia versions leave small allocations even for correct code, so the
    # guarantee is asserted on 1.12+ only (cf. the threaded path, which inherently
    # allocates task objects and is not asserted).
    if VERSION >= v"1.12"
        @testset "Local path is allocation-free (closure elimination)" begin
            x = LocalHaloArray(Float64, (64, 64), 1; boundary_condition=:periodic)
            fill!(x, 1.0)
            y = similar(x); fill!(y, 2.0)
            alloc(f, args...) = (f(args...); @allocated f(args...))   # warm, then measure
            @test alloc(rmul!, x, 1.000001) == 0
            @test alloc(lmul!, 1.000001, x) == 0
            @test alloc(axpy!, 1e-9, x, y) == 0
            @test alloc(axpby!, 1e-9, x, 0.999999, y) == 0
            @test alloc(swap!, x, y) == 0
            @test alloc((a, b) -> rotate!(a, b, 0.8, 0.6), x, y) == 0
            @test alloc((a, b) -> reflect!(a, b, 0.8, 0.6), x, y) == 0
            @test alloc(fill!, x, 1.5) == 0
            @test alloc(copyto!, y, x) == 0
            @test alloc(sum, x) == 0
            @test alloc(norm, x) == 0
            @test alloc(dot, x, y) == 0
            @test alloc(u -> mapreduce(abs2, +, u), x) == 0
            @test alloc(u -> any(>(0.5), u), x) == 0
            @test alloc(u -> all(>(0.0), u), x) == 0
        end
    end

    @testset "== compares interiors, ignores ghosts" begin
        for mk in (n -> LocalHaloArray(Float64, (n, 4), 1; boundary_condition=:periodic),
                   n -> ThreadedHaloArray(Float64, (n ÷ 2, 4), 1;
                       dims=(2, 1), boundary_condition=:periodic))
            u = mk(6); v = mk(6)
            fill_from_global_indices!(I -> Float64(I[1] * I[2]), u)
            fill_from_global_indices!(I -> Float64(I[1] * I[2]), v)
            @test u == v
            synchronize_halo!(u)          # u ghosts valid, v ghosts stale
            @test u == v                  # ghosts must not affect equality
            v[3, 2] = -99.0
            @test !(u == v)               # one interior cell differs
            @test !(u == mk(4))           # size mismatch → false, no throw
        end
    end

    @testset "p-norm special cases match Base (p = 1, 3, ±Inf, 0)" begin
        vals = [3.0, -4.0, 0.0, 1.0, -2.0, 5.0]
        for u in Any[LocalHaloArray(Float64, (6,), 1; boundary_condition=:periodic),
                     ThreadedHaloArray(Float64, (3,), 1;
                         dims=(2,), boundary_condition=:periodic)]
            fill_from_global_indices!(I -> vals[I[1]], u)
            for p in (1, 3, Inf, -Inf, 0)   # -Inf = min |x|, 0 = count of nonzeros
                @test norm(u, p) ≈ norm(vals, p)
            end
        end
    end
end
