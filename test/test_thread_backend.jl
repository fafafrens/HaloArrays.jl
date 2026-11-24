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

# A backend that counts how often the tiles are started: one `tile_foreach` is
# one sweep (on a task-based backend, one round of spawns).
struct CountingBackend <: ThreadBackend end
const SWEEPS = Ref(0)
HaloArrays.tile_foreach(::CountingBackend, f, itr; scheduler=nothing) =
    (SWEEPS[] += 1; foreach(f, itr); nothing)
HaloArrays.tile_mapreduce(::CountingBackend, f, op, itr; scheduler=nothing) =
    mapreduce(f, op, itr)
sweeps(f) = (SWEEPS[] = 0; f(); SWEEPS[])

@testset "Collections: one tile sweep per bulk write" begin
    names = (:a, :b, :c)
    threaded(b) = MultiHaloArray(ThreadedHaloArray, Float64, (4, 6), 1; fields=names,
        boundary_condition=:periodic, dims=(2, 2), thread_backend=b)
    local_() = MultiHaloArray(LocalHaloArray, Float64, (8, 12), 1; fields=names,
        boundary_condition=:periodic)
    init!(c, s) = (foreach(k -> fill_from_global_indices!(I -> s * (k + I[1] + 10 * I[2]), c[names[k]]), 1:3); c)
    same(c, ref) = all(n -> gather_haloarray(c[n]) == gather_haloarray(ref[n]), names)
    A = reshape(collect(1.0:96.0), 8, 12)      # a plain array: global interior coordinates

    ul, vl, wl = init!(local_(), 1.0), init!(local_(), -0.5), local_()
    @testset "$(nameof(typeof(b)))" for b in (ThreadsBackend(), OhMyThreadsBackend(),
            SerialBackend(), PolyesterBackend(), CountingBackend())
        u, v, w = init!(threaded(b), 1.0), init!(threaded(b), -0.5), threaded(b)
        a = u[:b]                              # a single array applies to every field

        @test same(fill!(w, 3.0), fill!(wl, 3.0))
        @test same(w .= u .+ 2 .* v, wl .= ul .+ 2 .* vl)
        @test same(w .= u .* 0.5, wl .= ul .* 0.5)
        @test same(w .= u .* a, wl .= ul .* ul[:b])
        @test same(w .= u .+ A, wl .= ul .+ A)
        @test same(u .- v, ul .- vl)           # out of place
        @test same(zero(u), zero(ul))

        hits = zeros(Int, tile_count(u))
        tile_foreach(u) do t
            hits[t] += length(tile_parent(u, t))   # every field of this tile
        end
        @test hits == fill(3, tile_count(u))
        @test tile_mapreduce(t -> 10 * t, +, u) == 10 * sum(1:tile_count(u))
    end

    # (other names than in the loop above: rebinding `u` would box it there)
    cu, cv, cw = init!(threaded(CountingBackend()), 1.0), init!(threaded(CountingBackend()), 2.0), threaded(CountingBackend())
    @test sweeps(() -> fill!(cw, 1.0)) == 1
    @test sweeps(() -> (cw .= cu .+ 2 .* cv)) == 1
    @test sweeps(() -> (cw .= cu .+ A)) == 1
    @test sweeps(() -> tile_foreach(t -> nothing, cu)) == 1
    @test sweeps(() -> fill!(cu[:a], 1.0)) == 1          # a single array: unchanged

    # collections of collections: still one sweep, every leaf written
    nested(b) = ArrayOfHaloArray([threaded(b), threaded(b)])
    n, m = nested(CountingBackend()), nested(CountingBackend())
    @test sweeps(() -> fill!(m, 2.0)) == 1
    @test sweeps(() -> (n .= m .* 3)) == 1
    @test all(c -> all(k -> all(==(6.0), gather_haloarray(c[k])), names), parent(n))

    # a field-count mismatch is refused before any tile runs
    two = MultiHaloArray(ThreadedHaloArray, Float64, (4, 6), 1; fields=(:a, :b),
        boundary_condition=:periodic, dims=(2, 2))
    three = threaded(ThreadsBackend())
    @test_throws DimensionMismatch three .= two .+ 1

    # ... at every nesting level: the first inner state must stay untouched when
    # the second one mismatches (the tiles run in tasks; nothing may be half written)
    pair(q) = MultiHaloArray((; p = threaded(ThreadsBackend()), q))
    dest, src = pair(threaded(ThreadsBackend())), pair(two)
    fill!(dest, 7.0); fill!(src, 1.0)
    @test_throws DimensionMismatch dest .= src .* 2
    @test all(n -> all(==(7.0), gather_haloarray(dest[:p][n])), names)

    # one sweep has one tiling and one thread backend: both are fixed at construction
    one_tile = MultiHaloArray(ThreadedHaloArray, Float64, (8, 12), 1; fields=names,
        boundary_condition=:periodic, dims=(1, 1))
    @test_throws DimensionMismatch ArrayOfHaloArray([one_tile, three])      # nested, other tile grid
    @test_throws DimensionMismatch ArrayOfHaloArray([three, one_tile])
    @test_throws ArgumentError MultiHaloArray((; a = three[:a], b = threaded(SerialBackend())[:a]))
    @test thread_backend(three) === ThreadsBackend()
    @test thread_backend(ArrayOfHaloArray([three, three])) === ThreadsBackend()

    # fields of different types (boundary conditions are type parameters) are
    # walked statically: nothing allocates beyond the backend's own tasks
    mixed(make; kw...) = MultiHaloArray((;
        a = make(; boundary_condition=:periodic, kw...),
        b = make(; boundary_condition=((Reflecting(), Repeating()), (Repeating(), Reflecting())), kw...)))
    tmixed() = mixed((; kw...) -> ThreadedHaloArray(Float64, (4, 6), 1; dims=(2, 2), kw...); thread_backend=SerialBackend())
    lmixed() = mixed((; kw...) -> LocalHaloArray(Float64, (8, 12), 1; kw...))
    bcast!(w, u, v) = (w .= u .+ 2 .* v; nothing)
    for make in (tmixed, lmixed)
        mu, mv, mw = fill!(make(), 1.0), fill!(make(), 2.0), make()
        bcast!(mw, mu, mv)
        @test all(==(5.0), gather_haloarray(mw[:a])) && all(==(5.0), gather_haloarray(mw[:b]))
        @test (@allocated bcast!(mw, mu, mv)) == 0
        @test (@allocated fill!(mw, 0.0)) == 0
    end
end
