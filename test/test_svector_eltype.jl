using Test
using HaloArrays
using StaticArrays
using LinearAlgebra: norm, dot
import LinearAlgebra
using OrdinaryDiffEq: ODEProblem, Tsit5, solve
using DiffEqBase: NonNumberEltypeError

# A struct that acts as a scalar cell: it has the vector-space contract (zero,
# +, -, *, abs2, dot, norm) but is not iterable, like a user's field bundle.
struct ScalarCell
    a::Float64
    b::SVector{2,Float64}
end
Base.zero(::Type{ScalarCell}) = ScalarCell(0.0, zero(SVector{2,Float64}))
Base.zero(::ScalarCell) = zero(ScalarCell)
Base.:+(x::ScalarCell, y::ScalarCell) = ScalarCell(x.a + y.a, x.b + y.b)
Base.:-(x::ScalarCell, y::ScalarCell) = ScalarCell(x.a - y.a, x.b - y.b)
Base.:-(x::ScalarCell) = ScalarCell(-x.a, -x.b)
Base.:*(s::Number, x::ScalarCell) = ScalarCell(s * x.a, s * x.b)
Base.:*(x::ScalarCell, s::Number) = s * x
Base.abs2(x::ScalarCell) = abs2(x.a) + sum(abs2, x.b)
LinearAlgebra.dot(x::ScalarCell, y::ScalarCell) = x.a * y.a + dot(x.b, y.b)
LinearAlgebra.norm(x::ScalarCell) = sqrt(abs2(x))
Base.:(==)(x::ScalarCell, y::ScalarCell) = x.a == y.a && x.b == y.b

_norm(u) = norm(u)
_dot(u, v) = dot(u, v)
# Allocation per call must not scale with the array: a regression to a
# temp-allocating (or boxing, type-unstable) form would. Measured inside a
# function, since a Float64 returned at top level boxes (16 B) on Julia 1.10.
function _reduction_alloc(T, x, n)
    a = LocalHaloArray(T, (n,), 1; boundary_condition=:periodic)
    fill!(a, x)
    norm(a); dot(a, a)
    return (@allocated(norm(a)), @allocated(dot(a, a)))
end

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

    @testset "norm / dot for nested static and custom cells" begin
        # `_elt_abs2` assumed every non-number cell iterates over numbers:
        # an SVector of SMatrix (gauge links) hit `abs2(::SMatrix)`, and a
        # struct cell hit `iterate`. Both must reduce like Base, type-stably.
        Link = SVector{2, SMatrix{2,2,Float64,4}}
        w = LocalHaloArray(Link, (4,), 1; boundary_condition=:periodic)
        interior_view(w) .= [Link(SMatrix{2,2}(i, 0.0, 0.0, i), SMatrix{2,2}(0.0, 1.0, 1.0, 0.0)) for i in 1:4]
        sq = sum(i -> 2i^2 + 2, 1:4)
        @test norm(w) ≈ sqrt(sq)
        @test dot(w, w) ≈ sq
        w2 = similar(w); w2 .= 2 .* w
        @test dot(w, w2) ≈ 2sq
        @test norm(w, Inf) ≈ maximum(i -> sqrt(2i^2 + 2), 1:4)
        @test Base.return_types(_norm, (typeof(w),)) == [Float64]
        @test Base.return_types(_dot, (typeof(w), typeof(w))) == [Float64]
        link = Link(SMatrix{2,2}(1.0, 0.0, 0.0, 1.0), SMatrix{2,2}(0.0, 1.0, 1.0, 0.0))
        @test _reduction_alloc(Link, link, 8) == _reduction_alloc(Link, link, 8_000)

        c = LocalHaloArray(ScalarCell, (4,), 1; boundary_condition=:periodic)
        interior_view(c) .= [ScalarCell(Float64(i), SVector(1.0, 2.0i)) for i in 1:4]
        synchronize_halo!(c)
        @test parent(c)[1] == ScalarCell(4.0, SVector(1.0, 8.0))
        sc = sum(i -> i^2 + 1 + 4i^2, 1:4)
        @test norm(c) ≈ sqrt(sc)
        @test dot(c, c) ≈ sc
        @test norm(c, 1) ≈ sum(i -> sqrt(i^2 + 1 + 4i^2), 1:4)
        @test sum(c) == ScalarCell(10.0, SVector(4.0, 20.0))
        @test Base.return_types(_norm, (typeof(c),)) == [Float64]
        cell = ScalarCell(1.0, SVector(1.0, 2.0))
        @test _reduction_alloc(ScalarCell, cell, 8) == _reduction_alloc(ScalarCell, cell, 8_000)

        tc = ThreadedHaloArray(ScalarCell, (2,), 1; dims=(2,), boundary_condition=:periodic)
        HaloArrays.fill_from_global_indices!(I -> ScalarCell(Float64(I[1]), SVector(1.0, 2.0 * I[1])), tc)
        @test norm(tc) ≈ sqrt(sc)
        @test dot(tc, tc) ≈ sc

        mc = MultiHaloArray((x=c, y=w))
        @test norm(mc) ≈ sqrt(sc + sq)
        @test dot(mc, mc) ≈ sc + sq
    end

    @testset "OrdinaryDiffEq refuses non-Number cells with SciML's own error" begin
        # DiffEqBase applies this check only to a plain Array state; without the
        # extension's guard a halo array of SVector cells failed deep in the
        # solver's initialization instead.
        decay!(du, u, p, t) = (du .= -u; nothing)
        v = LocalHaloArray(V, (4,), 1; boundary_condition=:periodic)
        fill!(v, SVector(1.0, 2.0, 3.0))
        @test_throws NonNumberEltypeError solve(ODEProblem(decay!, v, (0.0, 1.0)), Tsit5())
        c = LocalHaloArray(ScalarCell, (4,), 1; boundary_condition=:periodic)
        @test_throws NonNumberEltypeError solve(ODEProblem(decay!, c, (0.0, 1.0)), Tsit5())
        w = LocalHaloArray(Float64, (4,), 1; boundary_condition=:periodic)
        fill!(w, 2.0)
        sol = solve(ODEProblem(decay!, w, (0.0, 1.0)), Tsit5(); reltol=1e-8, abstol=1e-10)
        @test norm(sol.u[end]) ≈ exp(-1.0) * norm(w) rtol=1e-6
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
