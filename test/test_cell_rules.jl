using Test
using HaloArrays
using StaticArrays

# The rule contract: a boundary condition is `bc(u, k)` for one ghost cell.

struct DirichletRule{T} <: AbstractBoundaryCondition
    value::T
end
(bc::DirichletRule)(u, k) = 2bc.value - u(k - 1)

# the reference: the same rule written out on the storage, side by side
function _reference_fill!(a::AbstractArray{T,N}, bc, s, d, hw) where {T,N}
    sd = size(a, d)
    ghost = s == 1 ? (1:hw) : (sd-hw+1:sd)
    wall  = s == 1 ? hw + 1 : sd - hw
    into  = s == 1 ? 1 : -1
    for I in CartesianIndices(ntuple(i -> i == d ? ghost : (hw+1:size(a, i)-hw), N))
        k = s == 1 ? wall - I[d] : I[d] - wall
        cell(j) = a[CartesianIndex(ntuple(i -> i == d ? wall + into*j : I[i], N))]
        a[I] = bc(k, cell)
    end
    a
end

@testset "cell rules" begin

    @testset "built-ins vs explicit formulas, $(length(dims))D hw=$hw" for (dims, hw) in
            (((9,), 1), ((9,), 2), ((5, 6), 1), ((4, 5, 6), 2))
        x = rand(dims .+ 2hw...)
        rules = (
            (Repeating(),      (k, c) -> c(0)),
            (Mirror(1),        (k, c) -> c(k - 1)),
            (Mirror(-1),       (k, c) -> -c(k - 1)),
            (Extrapolation(1), (k, c) -> (1 + k) * c(0) - k * c(1)),
            (Extrapolation(2), (k, c) -> (1 + k) * (2 + k) / 2 * c(0) - k * (2 + k) * c(1) + k * (1 + k) / 2 * c(2)),
            (DirichletRule(0.25), (k, c) -> 0.5 - c(k - 1)),
        )
        for (bc, formula) in rules, d in 1:length(dims), s in 1:2
            bcs = ntuple(i -> i == d ? (s == 1 ? (bc, NoBoundaryCondition()) : (NoBoundaryCondition(), bc)) :
                                       (NoBoundaryCondition(), NoBoundaryCondition()), length(dims))
            u = LocalHaloArray(copy(x), hw, bcs)
            synchronize_halo!(u)
            @test parent(u) ≈ _reference_fill!(copy(x), formula, s, d, hw)
        end
    end

    @testset "far: the periodic rule equals the periodic copy" begin
        x = rand(8, 7)
        ref = LocalHaloArray(copy(x), 2, :periodic); synchronize_halo!(ref)
        # the same rule spelled as a closure (goes through the generic fill, not the slab copy)
        u = LocalHaloArray(copy(x), 2, (u, k) -> far(u)(k - 1)); synchronize_halo!(u)
        @test parent(u) == parent(ref)
        @test u.boundary_condition[1][1] isa FunctionBC
        # Extrapolation lowers to a stencil; everything else stays itself
        g = FaceGeometry(parent(ref), Side(1), Dim(1), Val(2), CartesianIndex(-1, 1))
        @test prepare_boundary(Periodic(), g) === Periodic()         # the copy conditions are their own rule
        @test prepare_boundary(Mirror(-1), g) === Mirror(-1)
        @test prepare_boundary(Extrapolation(3), g) isa Stencil{2,4,Float64}   # order 3: 4 cells per layer
        @test prepare_boundary(DirichletRule(1.0), g) isa DirichletRule
        # the weights follow the field's scalar type: Float32 field → Float32 weights, Int field → Int
        g32 = FaceGeometry(zeros(Float32, 6, 6), Side(1), Dim(1), Val(2), CartesianIndex(-1, 1))
        @test prepare_boundary(Extrapolation(1), g32) isa Stencil{2,2,Float32}
        @test prepare_boundary(Extrapolation(1), g32).w[1][1] === 2.0f0
        gi = FaceGeometry(zeros(Int, 6, 6), Side(1), Dim(1), Val(2), CartesianIndex(-1, 1))
        @test prepare_boundary(Extrapolation(2), gi) isa Stencil{2,3,Int}
        gv = FaceGeometry(zeros(SVector{2,Float32}, 6, 6), Side(1), Dim(1), Val(2), CartesianIndex(-1, 1))
        @test prepare_boundary(Extrapolation(1), gv) isa Stencil{2,2,Float32}
        # a weight Float32 cannot represent exactly is rejected at preparation
        @test_throws InexactError HaloArrays._weight(Float32, 2^24 + 1)
        @test HaloArrays._weight(Float32, 2^24) === Float32(2^24)
    end

    @testset "copies are copies: BigFloat keeps its precision under a lower default" begin
        vals = setprecision(BigFloat, 256) do
            [BigFloat(1) / BigFloat(3) + i for i in 1:4]           # 256-bit values
        end
        for bc in (Periodic(), Repeating(), Mirror(1))
            u = LocalHaloArray(BigFloat, (4,), 2; boundary_condition = bc)
            interior_view(u) .= vals
            setprecision(BigFloat, 64) do
                synchronize_halo!(u)
            end
            @test all(x -> precision(x) == 256, parent(u))
            src = bc isa Periodic ? vals[3:4] : bc isa Repeating ? vals[[1, 1]] : vals[[2, 1]]
            @test parent(u)[1:2] == src
        end
    end

    @testset "zero Dirichlet == Antireflecting on every ghost layer" begin
        a = LocalHaloArray(Float64, (4,), 2; boundary_condition = Antireflecting())
        d = LocalHaloArray(Float64, (4,), 2; boundary_condition = DirichletRule(0.0))
        interior_view(a) .= 1:4; interior_view(d) .= 1:4
        synchronize_halo!(a); synchronize_halo!(d)
        @test parent(d) == parent(a) == [-2, -1, 1, 2, 3, 4, -4, -3]
    end

    @testset "extrapolation weights: exact, overflow-checked, any order" begin
        # order 7 on a degree-7 polynomial, halo 3: exact in Float64 and in Int
        f(x) = sum(x^m for m in 0:7)
        u = LocalHaloArray(Int, (10,), 3; boundary_condition = Extrapolation(7))
        interior_view(u) .= f.(1:10); synchronize_halo!(u)
        @test parent(u)[1:3] == f.(-2:0) && parent(u)[14:16] == f.(11:13)
        # the weights of the base branch for a constant field sum to 1 on every layer
        c = LocalHaloArray(Int, (12,), 4; boundary_condition = Extrapolation(7))
        interior_view(c) .= 5; synchronize_halo!(c)
        @test all(==(5), parent(c))
        @test HaloArrays._lagrange_weight(0, 1, 1) == 2 && HaloArrays._lagrange_weight(1, 1, 1) == -1
        @test_throws OverflowError HaloArrays._lagrange_weight(3, 7, 10^17)
        # the order is a type parameter: the stencil is sized for any order (order 8, degree-8 polynomial, halo 1)
        f8(x) = sum(x^m for m in 0:8)
        u8 = LocalHaloArray(Int, (10,), 1; boundary_condition = Extrapolation(8))
        interior_view(u8) .= f8.(1:10); synchronize_halo!(u8)
        @test parent(u8)[1] == f8(0) && parent(u8)[12] == f8(11)
        @test Extrapolation(2) === Extrapolation{2}() && repr(Extrapolation(2)) == "Extrapolation(2)"
        @test_throws ArgumentError Extrapolation(-1)
    end

    @testset "interior extent: a rule never reads a ghost" begin
        # Mirror at halo 2 reads two interior cells: interior 1 is rejected, interior 2 accepted
        @test_throws ArgumentError synchronize_halo!(LocalHaloArray(Float64, (1,), 2; boundary_condition = Mirror(1)))
        @test synchronize_halo!(LocalHaloArray(Float64, (2,), 2; boundary_condition = Mirror(1))) isa LocalHaloArray
        # same on a threaded tile (tiles of 1 cell, halo 2)
        @test_throws ArgumentError synchronize_halo!(ThreadedHaloArray(Float64, (1,), 2; dims = (2,), boundary_condition = Mirror(-1)))
        # Repeating only needs the wall cell
        @test synchronize_halo!(LocalHaloArray(Float64, (1,), 2; boundary_condition = Repeating())) isa LocalHaloArray
    end

    @testset "zero halo width: nothing to fill, no error" begin
        e = LocalHaloArray(Float64, (0,), 0; boundary_condition = Repeating())
        @test synchronize_halo!(e) === e
        z = LocalHaloArray(Float64, (5,), 0; boundary_condition = Extrapolation(1))
        @test synchronize_halo!(z) === z
    end

    @testset "global_index: position-dependent rule, Local vs 2x2 Threaded" begin
        f(I) = 1000 * I[1] + I[2]
        rule = (u, k) -> Float64(f(Tuple(global_index(u))))
        ul = LocalHaloArray(Float64, (4, 4), 1; boundary_condition = rule)
        interior_view(ul) .= 0.0; boundary_condition!(ul)
        @test parent(ul)[1, 2] == f((0, 1))          # low-x ghost of global cell (1,1): x=0, y=1
        @test parent(ul)[6, 5] == f((5, 4))          # high-x ghost: x=N+1
        @test parent(ul)[3, 1] == f((2, 0))          # low-y ghost: y=0
        ut = ThreadedHaloArray(Float64, (2, 2), 1; dims = (2, 2), boundary_condition = rule)
        for t in 1:tile_count(ut); interior_view(ut, t) .= 0.0; end
        boundary_condition!(ut)
        vals(u::LocalHaloArray) = sort([v for d in 1:2, s in 1:2 for v in vec(ghost_view(u, Side(s), Dim(d)))])
        vals(u::ThreadedHaloArray) = sort([v for t in 1:tile_count(u), d in 1:2, s in 1:2 if neighbor_tile_id(u, t, d, s) == 0
                                           for v in vec(ghost_view(u, Side(s), Dim(d), t))])
        @test vals(ut) == vals(ul)
    end

    @testset "rules on tiles and vector elements" begin
        # threaded: physical faces only, same values as the Local block
        x = rand(6, 6)                                   # the 6×6 interior, 2×2 tiles of 3×3
        bcs = ((Mirror(-1), Extrapolation(1)), (DirichletRule(1.0), Repeating()))
        ul = LocalHaloArray(Float64, (6, 6), 1; boundary_condition = bcs)
        interior_view(ul) .= x; synchronize_halo!(ul)
        ut = ThreadedHaloArray(Float64, (3, 3), 1; dims = (2, 2), boundary_condition = bcs)
        for t in 1:tile_count(ut)
            c = HaloArrays.tile_coordinates(ut, t)
            interior_view(ut, t) .= x[(c[1]-1)*3+1:(c[1]-1)*3+3, (c[2]-1)*3+1:(c[2]-1)*3+3]
        end
        synchronize_halo!(ut)
        for t in 1:tile_count(ut), d in 1:2, s in 1:2
            neighbor_tile_id(ut, t, d, s) == 0 || continue
            c = HaloArrays.tile_coordinates(ut, t)
            g = ghost_view(ut, Side(s), Dim(d), t)
            r = ghost_view(ul, Side(s), Dim(d))
            off = d == 1 ? (c[2]-1)*3 : (c[1]-1)*3
            @test vec(g) ≈ vec(d == 1 ? r[:, off+1:off+3] : r[off+1:off+3, :])
        end
        # SVector elements: the rule's arithmetic is the element type's
        v = LocalHaloArray(SVector{2,Float64}, (5,), 1; boundary_condition = ((Mirror(-1), Extrapolation(1)),))
        interior_view(v) .= [SVector(i, 2i) for i in 1.0:5.0]
        synchronize_halo!(v)
        @test parent(v)[1] == -SVector(1.0, 2.0)
        @test parent(v)[7] == SVector(6.0, 12.0)
    end

    @testset "errors and allocations" begin
        w = LocalHaloArray(Float64, (2,), 1; boundary_condition = ((Extrapolation(2), Extrapolation(2)),))
        @test_throws ArgumentError synchronize_halo!(w)
        @test HaloArrays.to_bc((u, k) -> u(0)) isa FunctionBC
        u = LocalHaloArray(Float64, (16, 16), 2;
                boundary_condition = ((Mirror(-1), Extrapolation(2)), (DirichletRule(1.0), (u, k) -> u(0) + k)))
        interior_view(u) .= 1.0
        synchronize_halo!(u)
        @test (@allocated synchronize_halo!(u)) == 0
    end
end
