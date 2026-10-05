using Test
using HaloArrays
using LinearAlgebra: dot
using Polyester  # loads HaloArraysPolyesterExt so PolyesterBackend works
using OhMyThreads  # loads HaloArraysOhMyThreadsExt so OhMyThreadsBackend works
using StaticArrays: SVector

@testset "Thread backends" begin
    backends = (ThreadsBackend(), OhMyThreadsBackend(), SerialBackend(), PolyesterBackend())

    function build(backend)
        u = ThreadedHaloArray(Float64, (8, 8), 1; dims=(2, 2),
            boundary_condition=:periodic, thread_backend=backend)
        for I in CartesianIndices(axes(u))
            u[Tuple(I)...] = sinpi(I[1] / 8) + 2 * I[2]
        end
        synchronize_halo!(u)
        return u
    end

    # Serial backend is the ground truth; every backend must agree with it.
    ref    = build(SerialBackend())
    refsum = sum(ref)
    refmax = maximum(ref)
    refmin = minimum(ref)
    refany = any(>(5), ref)
    refall = all(>(-100), ref)
    refdot = dot(ref, ref)

    @testset "$(nameof(typeof(b)))" for b in backends
        u = build(b)

        # the backend is carried by the array and is part of its concrete type
        @test thread_backend(u) === b
        @test thread_backend(similar(u)) === b   # propagated through similar

        # reductions route through tile_mapreduce(thread_backend(u), …)
        @test sum(u)         ≈ refsum
        @test maximum(u)     ≈ refmax
        @test minimum(u)     ≈ refmin
        @test sum(abs2, u)   ≈ sum(abs2, ref)
        @test any(>(5), u)   == refany
        @test all(>(-100), u) == refall
        @test dot(u, u)      ≈ refdot

        # broadcast routes through tile_foreach(thread_backend(u), …)
        v = similar(u)
        v .= u .* 2
        @test sum(v) ≈ 2 * refsum
        @test thread_backend(v) === b

        # do-block forms (closure lands in the first argument slot)
        acc = zeros(Int, tile_count(u))
        tile_foreach(b, 1:tile_count(u)) do tile
            acc[tile] = tile
        end
        @test acc == collect(1:tile_count(u))
        s = tile_mapreduce(+, b, 1:tile_count(u)) do tile
            2 * tile
        end
        @test s == 2 * sum(1:tile_count(u))

        # array-level forms: dispatch through the array's own tile driver
        fill!(acc, 0)
        tile_foreach(u) do tile
            acc[tile] = tile
        end
        @test acc == collect(1:tile_count(u))
        s2 = tile_mapreduce(+, u) do tile
            2 * tile
        end
        @test s2 == 2 * sum(1:tile_count(u))

        # fill! and the threaded synchronize variant respect the backend
        fill!(u, 3.0)
        synchronize_halo!(u; threads=true)
        @test maximum(u) ≈ 3.0
        @test minimum(u) ≈ 3.0
    end

    # backend is compile-time information: different backends → different types
    @test typeof(build(SerialBackend())) !== typeof(build(OhMyThreadsBackend()))

    # default backend is Base threads (no extra package)
    udefault = ThreadedHaloArray(Float64, (4,), 1; dims=(1,), boundary_condition=:periodic)
    @test thread_backend(udefault) === ThreadsBackend()

    # array-level forms on a single-block array: one tile, run inline
    ul = LocalHaloArray(Float64, (4, 4), 1; boundary_condition=:periodic)
    hits = Int[]
    tile_foreach(ul) do tile
        push!(hits, tile)
    end
    @test hits == [1]
    sl = tile_mapreduce(+, ul) do tile
        10 * tile
    end
    @test sl == 10
end

@testset "Polyester reductions: parallel, ordered, typed" begin
    P = PolyesterBackend()
    # ordered combine for a non-commutative op, uneven chunking, any length
    for n in 1:7
        @test tile_mapreduce(P, i -> [i], vcat, 1:n) == collect(1:n)
        @test tile_mapreduce(P, i -> string(i), *, 1:n) == join(1:n)
    end
    # the chunk result type follows `op`, not just `f` (Bool + Bool is Int)
    @test tile_mapreduce(P, isodd, +, 1:6) === 3
    # a non-indexable iterator takes the fallback path and still agrees
    @test tile_mapreduce(P, identity, +, (i for i in 1:5)) == 15
    # the reduction-clause path (+ * min max & | on plain-bits results) agrees
    # with Base, keeps the sign of zero, and handles static vectors
    for n in 1:7
        @test tile_mapreduce(P, i -> Float64(i), +, 1:n) == sum(Float64, 1:n)
        @test tile_mapreduce(P, i -> i, *, 1:n) == prod(1:n)
        @test tile_mapreduce(P, i -> Float64(i), max, 1:n) == n
        @test tile_mapreduce(P, i -> -i, min, 1:n) == -n
        @test tile_mapreduce(P, isodd, &, 1:n) == all(isodd, 1:n)
        @test tile_mapreduce(P, isodd, |, 1:n) == any(isodd, 1:n)
        @test tile_mapreduce(P, i -> SVector(i, 2.0i), +, 1:n) == sum(i -> SVector(i, 2.0i), 1:n)
    end
    # each thread's partial starts at Polyester's zero, so an all-(-0.0) sum is
    # +0.0 on the threaded path (Base: -0.0); the value is zero either way
    @test tile_mapreduce(P, i -> -0.0, +, 1:3) == 0.0
    @test isnan(tile_mapreduce(P, i -> i == 2 ? NaN : 1.0, max, 1:3))
    if Threads.nthreads() > 1                                   # allocation-free, call count independent
        g(i) = Float64(i)
        calls(n) = (s = 0.0; for _ in 1:n; s += tile_mapreduce(P, g, +, 1:Threads.nthreads()); end; s)
        calls(1)
        @test @allocated(calls(1)) == @allocated(calls(100))
    end
    # with more than one thread, the chunks run on more than one thread (the
    # first chunk used to be reduced on the calling thread before the others)
    if Threads.nthreads() > 1
        n = Threads.nthreads()
        ids = zeros(Int, n)
        tile_mapreduce(P, i -> (ids[i] = Threads.threadid(); 1), +, 1:n)
        @test length(unique(ids)) > 1
    end
end

@testset "Base-threads reductions and loops: parallel, ordered, typed" begin
    B = ThreadsBackend()
    for n in 1:7
        @test tile_mapreduce(B, i -> [i], vcat, 1:n) == collect(1:n)     # chunk order kept
        @test tile_mapreduce(B, i -> string(i), *, 1:n) == join(1:n)
        @test tile_mapreduce(B, i -> Float64(i), +, 1:n) == sum(Float64, 1:n)
        @test tile_mapreduce(B, i -> SVector(i, 2.0i), +, 1:n) == sum(i -> SVector(i, 2.0i), 1:n)
        seen = zeros(Int, n)
        tile_foreach(B, i -> (seen[i] += 1), 1:n)
        @test all(==(1), seen)                                          # every tile once
    end
    @test tile_mapreduce(B, isodd, +, 1:6) === 3                         # type follows `op`
    @test tile_mapreduce(B, identity, +, (i for i in 1:5)) == 15         # non-indexable tiles
    @test (@inferred tile_mapreduce(B, i -> Float64(i), +, 1:4)) == 10.0 # type-stable
    if Threads.nthreads() > 1
        # Every chunk waits (bounded) until all chunks have started: with
        # trivial work the calling thread may otherwise run a queued task itself
        # while it waits, so the thread ids would not prove concurrency.
        n = Threads.nthreads()
        function rendezvous!(started, ids, i)
            Threads.atomic_add!(started, 1)
            t0 = time()
            while started[] < length(ids) && time() - t0 < 5
                GC.safepoint()                    # let another thread's GC proceed
                ccall(:jl_cpu_pause, Cvoid, ())
            end
            ids[i] = Threads.threadid()
            return started[] == length(ids)
        end
        ids, started = zeros(Int, n), Threads.Atomic{Int}(0)
        tile_foreach(B, i -> (rendezvous!(started, ids, i); nothing), 1:n)
        @test started[] == n && length(unique(ids)) == n                 # really parallel
        rids, rstarted = zeros(Int, n), Threads.Atomic{Int}(0)
        @test tile_mapreduce(B, i -> rendezvous!(rstarted, rids, i), &, 1:n)
        @test length(unique(rids)) == n
    end
end
