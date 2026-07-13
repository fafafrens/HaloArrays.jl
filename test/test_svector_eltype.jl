using Test
using HaloArrays
using StaticArrays
using LinearAlgebra: norm, dot

# ============================================================
# Halo arrays with an SVector element type — the "array of structs" layout for a
# fixed-size N-component field (one padded array, ghosts only on the spatial
# dimensions, the cell state contiguous). This exercises that the core
# operations — synchronize_halo!, broadcast, inter-tile halo exchange, and
# reductions — work elementwise on SVector cells, not just on scalars.
# ============================================================

@testset "SVector element type" begin
    V = SVector{3,Float64}

    @testset "LocalHaloArray{SVector}: synchronize, fill!, broadcast, reductions" begin
        u = LocalHaloArray(V, (4,), 1; boundary_condition=:periodic)
        interior_view(u) .= [SVector(Float64(i), 2i, 3i) for i in 1:4]
        synchronize_halo!(u)
        d = parent(u)
        @test d[1]   == SVector(4.0, 8.0, 12.0)   # periodic: left ghost = interior[end]
        @test d[end] == SVector(1.0, 2.0, 3.0)    # periodic: right ghost = interior[1]

        # fill! is interior-only; ghosts are left to synchronize_halo!
        sentinel = SVector(9.0, 9.0, 9.0)
        parent(u)[1] = sentinel
        fill!(u, SVector(1.0, 1.0, 1.0))
        @test all(==(SVector(1.0, 1.0, 1.0)), interior_view(u))
        @test parent(u)[1] == sentinel

        # broadcast is interior-only and yields a halo array of the same type
        u .*= 2.0
        @test interior_view(u)[1] == SVector(2.0, 2.0, 2.0)
        w = u .+ u
        @test w isa typeof(u)
        @test interior_view(w)[1] == SVector(4.0, 4.0, 4.0)

        # reductions act over the interior, componentwise on the SVector
        @test sum(u) == SVector(8.0, 8.0, 8.0)                 # 4 cells * (2,2,2)
        @test mapreduce(x -> sum(abs2, x), +, u) == 4 * 12.0   # 4 cells * (2²+2²+2²)
    end

    @testset "LocalHaloArray{SVector}: reflecting / antireflecting fill the whole vector" begin
        for (bc, expect) in ((:reflecting, SVector(1.0, 2.0)),
                             (:antireflecting, SVector(-1.0, -2.0)))
            r = LocalHaloArray(SVector{2,Float64}, (3,), 1; boundary_condition=bc)
            interior_view(r) .= [SVector(1.0, 2.0), SVector(3.0, 4.0), SVector(5.0, 6.0)]
            synchronize_halo!(r)
            @test parent(r)[1] == expect    # left ghost mirrors (or negates) interior[1]
        end
    end

    @testset "2-D LocalHaloArray{SVector}: synchronize both dimensions" begin
        g = LocalHaloArray(SVector{2,Float64}, (3, 3), 1; boundary_condition=:periodic)
        for I in CartesianIndices(interior_view(g))
            i, j = Tuple(I)
            interior_view(g)[I] = SVector(Float64(i), Float64(j))
        end
        synchronize_halo!(g)
        d = parent(g)
        @test sum(g) == SVector(18.0, 18.0)
        @test d[1, 3] == SVector(3.0, 2.0)   # dim-1 left ghost wraps i: 1 -> 3
        @test d[5, 3] == SVector(1.0, 2.0)   # dim-1 right ghost wraps i: 3 -> 1
        @test d[3, 1] == SVector(2.0, 3.0)   # dim-2 left ghost wraps j: 1 -> 3
    end

    @testset "ThreadedHaloArray{SVector}: inter-tile halo exchange" begin
        t = ThreadedHaloArray(V, (3,), 1; dims=(2,), boundary_condition=:periodic)
        interior_view(t, 1) .= [SVector(1.0, 1.0, 1.0), SVector(2.0, 2.0, 2.0), SVector(3.0, 3.0, 3.0)]
        interior_view(t, 2) .= [SVector(4.0, 4.0, 4.0), SVector(5.0, 5.0, 5.0), SVector(6.0, 6.0, 6.0)]
        synchronize_halo!(t)
        tp1 = tile_parent(t, 1)
        tp2 = tile_parent(t, 2)
        @test tp1[end] == SVector(4.0, 4.0, 4.0)   # tile1 right ghost = tile2 interior[1]
        @test tp2[1]   == SVector(3.0, 3.0, 3.0)   # tile2 left  ghost = tile1 interior[end]
        @test tp2[end] == SVector(1.0, 1.0, 1.0)   # periodic wrap: tile2 right ghost = tile1 interior[1]
        @test tp1[1]   == SVector(6.0, 6.0, 6.0)   # periodic wrap: tile1 left  ghost = tile2 interior[end]

        t .*= 2.0
        @test interior_view(t, 1)[1] == SVector(2.0, 2.0, 2.0)
        @test sum(t) == SVector(42.0, 42.0, 42.0)
    end

    @testset "norm / dot return the scalar Base does for SVector cells" begin
        # `norm(u)` used `abs2(::SVector)` and `dot(u,u)` did `SVector*SVector` —
        # both undefined, so these threw. They must fold each cell's Euclidean
        # contribution into a scalar, exactly like Base on the interior array.
        u = LocalHaloArray(V, (4,), 1; boundary_condition=:periodic)
        interior_view(u) .= [SVector(Float64(i), 2i, 3i) for i in 1:4]
        ref = collect(interior_view(u))

        @test norm(u)      ≈ norm(ref)
        @test norm(u)      isa Float64
        @test dot(u, u)    ≈ dot(ref, ref)
        @test dot(u, u)    isa Float64
        @test norm(u, 1)   ≈ norm(ref, 1)
        @test norm(u, Inf) ≈ norm(ref, Inf)
        @test norm(u)      ≈ sqrt(dot(u, u))

        # 2-D
        g = LocalHaloArray(SVector{2,Float64}, (3, 3), 1; boundary_condition=:periodic)
        for I in CartesianIndices(interior_view(g))
            i, j = Tuple(I)
            interior_view(g)[I] = SVector(Float64(i), Float64(j))
        end
        gref = collect(interior_view(g))
        @test norm(g)   ≈ norm(gref)
        @test dot(g, g) ≈ dot(gref, gref)

        # threaded (per-tile reduction combines to the same global scalar)
        t = ThreadedHaloArray(V, (3,), 1; dims=(2,), boundary_condition=:periodic)
        HaloArrays.fill_from_global_indices!(I -> SVector(Float64(I[1]), 0.0, -1.0), t)
        tref = [SVector(Float64(i), 0.0, -1.0) for i in 1:6]
        @test norm(t)   ≈ norm(tref)
        @test dot(t, t) ≈ dot(tref, tref)
    end

    @testset "copy / zero / similar preserve the SVector eltype" begin
        u = LocalHaloArray(V, (4,), 1; boundary_condition=:periodic)
        fill!(u, SVector(1.0, 2.0, 3.0))
        c = copy(u)
        @test c !== u
        @test eltype(c) === V
        @test interior_view(c)[1] == SVector(1.0, 2.0, 3.0)
        z = zero(u)
        @test all(==(zero(V)), interior_view(z))
        s = similar(u)
        @test eltype(s) === V
        @test size(s) == size(u)
    end
end
