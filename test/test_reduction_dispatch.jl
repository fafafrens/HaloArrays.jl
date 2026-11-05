using Test
using HaloArrays
using LinearAlgebra

# Dispatch holes closed in 0.17: a callable struct as the reducer, a mixed
# single/collection dot, copyto! across element types, scalar indexing on a
# LocalHaloArray, array-valued cells in dot.

struct Sq end
(::Sq)(x) = x^2
struct Pos end
(::Pos)(x) = x > 0
_getij(u) = u[2, 3]
_setij!(u) = (u[2, 3] = 1.0; nothing)
# measured inside a function: at testset level `@allocated` boxes a scalar result on Julia 1.10
_alloc(f, x) = (f(x); @allocated f(x))
_alloc(f, x, y) = (f(x, y); @allocated f(x, y))

@testset "reductions take any callable, not only a Function" begin
    u = LocalHaloArray(Float64, (6, 5), 1; boundary_condition = :periodic)
    interior_view(u) .= reshape(1:30, 6, 5)
    @test mapreduce(Sq(), +, u) == sum(abs2, 1:30)
    @test mapfoldl(Sq(), +, u) == sum(abs2, 1:30)
    @test mapfoldr(Sq(), +, u) == sum(abs2, 1:30)
    @test any(Pos(), u) && all(Pos(), u)
    @test mapreduce(abs2, +, u) == sum(abs2, 1:30) && any(>(0), u)   # plain functions as before
    for (f, sig) in ((mapreduce, (Sq, typeof(+), typeof(u))), (any, (Pos, typeof(u))), (all, (Pos, typeof(u))),
                     (any, (typeof(>(0)), typeof(u))))
        @test which(f, sig).module === HaloArrays
    end

    t = ThreadedHaloArray(Float64, (6, 6), 1; dims = (2, 1), boundary_condition = :periodic)
    fill!(t, 2.0)
    @test mapreduce(Sq(), +, t) == 2 * sum(t) && all(Pos(), t)    # Sq(2) = 2 · 2 on every cell

    state = MultiHaloArray(LocalHaloArray, Float64, (6, 5), 1; fields = (:a, :b), boundary_condition = :periodic)
    fill!(state, 2.0)
    @test mapreduce(Sq(), +, state) == 4 * 60 && all(Pos(), state)
    @test which(any, (Pos, typeof(state))).module === HaloArrays
    @test which(mapreduce, (Sq, typeof(+), typeof(state))).module === HaloArrays
end

@testset "mixed dot is an error, copyto! works across element types" begin
    u = LocalHaloArray(Float64, (6, 5), 1; boundary_condition = :periodic)
    state = MultiHaloArray(LocalHaloArray, Float64, (6, 5), 1; fields = (:a, :b), boundary_condition = :periodic)
    fill!(state, 2.0)
    @test_throws ArgumentError dot(u, state)
    @test_throws ArgumentError dot(state, u)
    s32 = similar(state, Float32)
    @test which(copyto!, (typeof(s32), typeof(state))).module === HaloArrays
    copyto!(s32, state)
    @test eltype(s32) == Float32 && all(==(2f0), s32)
end

@testset "scalar indexing of a LocalHaloArray is allocation-free" begin
    u = LocalHaloArray(Float64, (6, 5), 1; boundary_condition = :periodic)
    interior_view(u) .= reshape(1:30, 6, 5)
    @test _getij(u) == 14.0                                     # column-major: (2, 3) ↦ 2 + 6·2
    @test _alloc(_getij, u) == 0
    @test _alloc(_setij!, u) == 0
    @test _getij(u) == 1.0
    @test_throws BoundsError u[0, 1]
    @test_throws BoundsError u[7, 1]
end

@testset "array-valued cells: dot through LinearAlgebra, allocation-free" begin
    cells = [[1.0, 2.0] for _ in 1:6, _ in 1:5]           # 4×3 interior at halo 1
    u = LocalHaloArray(cells, 1, :periodic)
    @test dot(u, u) == 5.0 * 12
    @test _alloc(dot, u, u) == 0
end
