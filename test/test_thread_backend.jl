using Test
using HaloArrays
using LinearAlgebra: dot
using Polyester  # loads HaloArraysPolyesterExt so PolyesterBackend works

@testset "Thread backends" begin
    backends = (OhMyThreadsBackend(), SerialBackend(), PolyesterBackend())

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
        synchronize_halo_threads!(u)
        @test maximum(u) ≈ 3.0
        @test minimum(u) ≈ 3.0
    end

    # backend is compile-time information: different backends → different types
    @test typeof(build(SerialBackend())) !== typeof(build(OhMyThreadsBackend()))

    # default backend is OhMyThreads
    udefault = ThreadedHaloArray(Float64, (4,), 1; dims=(1,), boundary_condition=:periodic)
    @test thread_backend(udefault) === OhMyThreadsBackend()

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
