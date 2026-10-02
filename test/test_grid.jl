using Test
using HaloArrays
using StaticArrays

# Cell geometry (grid.jl): axis descriptions, construction on every serial
# backend (ghosts included), and the metric helpers against exact integrals.

_interior(u) = CartesianIndices(interior_range(u))
# Allocation must not grow with the call count: `@allocated` of a Float64
# result reports a 16 B box on Julia 1.10 even inside a function, so compare
# one call against a thousand instead of asserting zero. The arguments travel
# as a tuple (a splatted Vararg would not specialise and would allocate).
_calls(f::F, n, args::Tuple) where {F} = (s = 0.0; for _ in 1:n; s += f(args...); end; s)
function _allocation_free(f::F, args...) where {F}
    a = args
    _calls(f, 1, a)
    return @allocated(_calls(f, 1, a)) == @allocated(_calls(f, 1000, a))
end
# storage indices that are not corner ghosts (ghost in at most one direction)
_noncorner(data, hw) = (I for I in CartesianIndices(data)
    if count(d -> !(hw < I[d] <= size(data, d) - hw), 1:ndims(data)) <= 1)

@testset "cell geometry" begin
    @testset "axes" begin
        e = cell_edges((0, 0.1, 1), (2, 3))
        @test e ≈ [0, 0.05, 0.1, 0.4, 0.7, 1.0]
        @test_throws ArgumentError cell_edges((0, 1), (2, 3))
        @test_throws ArgumentError cell_edges((0, 1), (0,))
        @test_throws ArgumentError EdgeAxis([1.0, 0.0])
        @test_throws ArgumentError EdgeAxis([1.0])
        @test coordinate_names(Cartesian(), Val(2)) == (:x, :y)
        @test coordinate_names(Cylindrical(), Val(3)) == (:r, :θ, :z)
        @test coordinate_names(Polar(), Val(2)) == (:r, :θ)
        @test_throws ArgumentError coordinate_names(Polar(), Val(3))
        @test coordinate_names(Spherical(), Val(2)) == (:r, :θ)
        @test_throws ArgumentError coordinate_names(Spherical(), Val(4))
    end

    @testset "construction and ghost cells (LocalHaloArray)" begin
        u = LocalHaloArray(Float64, (8, 4), 2; boundary_condition=:reflecting)
        g = cell_geometry(u, (UniformAxis(0, 1), EdgeAxis(cell_edges((0, 0.1, 1), (1, 3)))))
        @test g isa MultiHaloArray
        @test propertynames(g) == (:x, :y, :hx, :hy)
        @test interior_size(g.x) == (8, 4) && halo_width(g.x) == 2
        gs = field_storages(g)
        # uniform axis: centres at (i - 1/2) h, continued through the ghosts on
        # physical edges by the geometry's own FunctionBC
        @test gs.x[:, 3] ≈ [(i - 0.5) / 8 for i in -1:10]
        @test all(gs.hx[I] ≈ 1 / 8 for I in _noncorner(gs.hx, 2))
        # edge axis: interior centres/widths from the edges, ghosts continue
        # with the edge cell's width
        @test gs.y[3, 3:6] ≈ [0.05, 0.25, 0.55, 0.85]
        @test gs.hy[3, 3:6] ≈ [0.1, 0.3, 0.3, 0.3]
        @test gs.y[3, 1:2] ≈ [-0.15, -0.05]
        @test gs.y[3, 7:8] ≈ [1.15, 1.45]
        # corner ghosts are not filled (library convention) but deterministic
        @test gs.x[1, 1] == 0 && gs.hy[12, 8] == 0
        @test g.x.boundary_condition[1][1] isa FunctionBC
        # synchronizing again changes nothing
        before = copy(gs.x); synchronize_halo!(g)
        @test field_storages(g).x == before
        # periodic directions wrap like any other field
        p = LocalHaloArray(Float64, (8, 4), 2; boundary_condition=:periodic)
        gp = cell_geometry(p, UniformAxis(0, 1), UniformAxis(0, 1))
        ps = field_storages(gp)
        @test ps.x[1:2, 3] ≈ [6.5 / 8, 7.5 / 8] && ps.x[11:12, 3] ≈ [0.5 / 8, 1.5 / 8]
        @test gp.x.boundary_condition == ((Periodic(), Periodic()), (Periodic(), Periodic()))
        # vararg and tuple forms agree; Float32 storage on request
        g32 = cell_geometry(u, UniformAxis(0, 1), EdgeAxis(cell_edges((0, 0.1, 1), (1, 3))); T=Float32)
        @test eltype(g32.x) == Float32
        @test all(field_storages(g32).x[I] ≈ gs.x[I] for I in _noncorner(gs.x, 2))
        # errors
        @test_throws DimensionMismatch cell_geometry(u, (UniformAxis(0, 1),))
        @test_throws DimensionMismatch cell_geometry(u, (UniformAxis(0, 1), EdgeAxis([0.0, 1.0])))
        @test_throws ArgumentError cell_geometry(u, (UniformAxis(0, 1), 1.0))
        # a collection source uses its spatial layout
        m = MultiHaloArray(LocalHaloArray, Float64, (8, 4), 2; fields=(:a, :b))
        gm = cell_geometry(m, UniformAxis(0, 1), UniformAxis(0, 1))
        @test interior_size(gm.x) == (8, 4) && halo_width(gm.x) == 2
    end

    @testset "ThreadedHaloArray tiles agree with the single block" begin
        t = ThreadedHaloArray(Float64, (4, 4), 1; dims=(2, 2), boundary_condition=:periodic)
        l = LocalHaloArray(t)
        axes2 = (UniformAxis(0, 1), EdgeAxis(cell_edges((0, 0.25, 1), (2, 6))))
        gt = cell_geometry(t, axes2)
        gl = cell_geometry(l, axes2)
        @test gt.x isa ThreadedHaloArray && tile_count(gt.x) == 4
        ls = field_storages(gl)
        for tid in 1:tile_count(t)
            ts = tile_parent(gt, tid)
            for I in _noncorner(ts.x, 1)          # tile corners are never exchanged
                G = interior_to_global_index(t, tid, ntuple(d -> 1, 2)) .+ (Tuple(I) .- 1 .- 1)
                J = CartesianIndex(G .+ 1)             # storage index in the single block
                @test ts.x[I] ≈ ls.x[J]
                @test ts.y[I] ≈ ls.y[J]
                @test ts.hy[I] ≈ ls.hy[J]
            end
        end
    end

    @testset "Cartesian metric" begin
        u = LocalHaloArray(Float64, (6, 5, 4), 1)
        g = cell_geometry(u, (UniformAxis(0, 3), UniformAxis(-1, 1), EdgeAxis(cell_edges((0, 1, 5), (1, 3)))))
        sys = Cartesian()
        I = CartesianIndex(3, 3, 3)                   # global cell (2, 2, 2) with halo 1
        @test cell_center(sys, g, I) ≈ SVector(0.75, -0.4, 1 + 4 / 3 / 2)
        @test cell_width(sys, g, I) ≈ SVector(0.5, 0.4, 4 / 3)
        @test cell_volume(sys, g, I) ≈ 0.5 * 0.4 * 4 / 3
        @test sum(cell_volume(sys, g, J) for J in _interior(u)) ≈ 3 * 2 * 5
        @test face_area(sys, g, Dim(1), I) ≈ 0.4 * 4 / 3
        @test face_area(sys, g, Dim(3), I) ≈ 0.5 * 0.4
        @test face_center(sys, g, Dim(1), I) ≈ SVector(1.0, -0.4, 1 + 4 / 3 / 2)
        @test face_distance(sys, g, Dim(3), CartesianIndex(3, 3, 2)) ≈ (1 + 4 / 3) / 2   # graded cells
        @test face_distance(sys, g, Dim(1), CartesianIndex(3, 3, 3)) ≈ 0.5
        @test face_normal(sys, g, Dim(2), I) == SVector(0.0, 1.0, 0.0)
        # consistency: the plus face of I - e_d is the minus face of I
        @test face_center(sys, g, Dim(3), CartesianIndex(3, 3, 2))[3] ≈
            cell_center(sys, g, I)[3] - cell_width(sys, g, I)[3] / 2
    end

    @testset "cylindrical metric is exact" begin
        c = LocalHaloArray(Float64, (10, 7, 5), 1)
        g = cell_geometry(c, (UniformAxis(0, 2), UniformAxis(0, 2π), UniformAxis(0, 3)); system=Cylindrical())
        @test propertynames(g) == (:r, :θ, :z, :hr, :hθ, :hz)
        sys = Cylindrical()
        @test sum(cell_volume(sys, g, I) for I in _interior(c)) ≈ π * 2^2 * 3
        # outer radial faces: the lateral surface 2π R H; axis faces have zero area
        @test sum(face_area(sys, g, Dim(1), I) for I in _interior(c) if I[1] == 11) ≈ 2π * 2 * 3
        @test all(face_area(sys, g, Dim(1), CartesianIndex(1, j, k)) ≈ 0 for j in 2:8, k in 2:6)
        # z faces: the disk π R²; θ faces: the rectangle R H
        @test sum(face_area(sys, g, Dim(3), I) for I in _interior(c) if I[3] == 6) ≈ π * 2^2
        @test sum(face_area(sys, g, Dim(2), I) for I in _interior(c) if I[2] == 4) ≈ 2 * 3
        # distances: r Δθ along θ, Δz along z, Δr along r
        I = CartesianIndex(5, 4, 3)
        c, w = cell_center(sys, g, I), cell_width(sys, g, I)
        @test face_distance(sys, g, Dim(2), I) ≈ c[1] * w[2]
        @test face_distance(sys, g, Dim(3), I) ≈ w[3]
        @test face_distance(sys, g, Dim(1), I) ≈ w[1]
        # axisymmetric (r, z) and radial-only forms: per unit angle
        c2 = LocalHaloArray(Float64, (10, 5), 1)
        g2 = cell_geometry(c2, (UniformAxis(0, 2), UniformAxis(0, 3)); system=Cylindrical())
        @test propertynames(g2) == (:r, :z, :hr, :hz)
        @test sum(cell_volume(sys, g2, I) for I in _interior(c2)) ≈ 2^2 / 2 * 3
        @test face_distance(sys, g2, Dim(2), CartesianIndex(4, 3)) ≈ cell_width(sys, g2, CartesianIndex(4, 3))[2]
        c1 = LocalHaloArray(Float64, (10,), 1)
        g1 = cell_geometry(c1, EdgeAxis(cell_edges((0, 1, 2), (4, 6))); system=Cylindrical())
        @test sum(cell_volume(sys, g1, I) for I in _interior(c1)) ≈ 2^2 / 2
    end

    @testset "polar metric is exact" begin
        d = LocalHaloArray(Float64, (10, 8), 1)
        g = cell_geometry(d, (EdgeAxis(cell_edges((0, 0.5, 2), (3, 7))), UniformAxis(0, 2π)); system=Polar())
        @test propertynames(g) == (:r, :θ, :hr, :hθ)
        sys = Polar()
        @test sum(cell_volume(sys, g, I) for I in _interior(d)) ≈ π * 2^2            # disk
        @test sum(face_area(sys, g, Dim(1), I) for I in _interior(d) if I[1] == 11) ≈ 2π * 2   # circumference
        @test all(face_area(sys, g, Dim(1), CartesianIndex(1, j)) ≈ 0 for j in 2:9)         # centre
        @test sum(face_area(sys, g, Dim(2), I) for I in _interior(d) if I[2] == 3) ≈ 2       # a radius
        I = CartesianIndex(5, 4)
        # distances are physical: r Δθ along the angle, Δr along the radius
        c, w = cell_center(sys, g, I), cell_width(sys, g, I)
        @test face_distance(sys, g, Dim(2), I) ≈ c[1] * w[2]
        @test face_distance(sys, g, Dim(1), CartesianIndex(3, 4)) ≈ (cell_width(sys, g, CartesianIndex(3, 4))[1] + cell_width(sys, g, CartesianIndex(4, 4))[1]) / 2
        @test face_center(sys, g, Dim(1), I)[1] ≈ cell_center(sys, g, I)[1] + cell_width(sys, g, I)[1] / 2
        @test face_normal(sys, g, Dim(2), I) == SVector(0.0, 1.0)
        f(g, I) = cell_volume(sys, g, I) + face_area(sys, g, Dim(1), I) + face_area(sys, g, Dim(2), I)
        @test _allocation_free(f, g, I)
    end

    @testset "spherical metric is exact" begin
        s = LocalHaloArray(Float64, (10, 6, 4), 1)
        g = cell_geometry(s, (UniformAxis(0, 2), UniformAxis(0, π), UniformAxis(0, 2π)); system=Spherical())
        @test propertynames(g) == (:r, :θ, :φ, :hr, :hθ, :hφ)
        sys = Spherical()
        @test sum(cell_volume(sys, g, I) for I in _interior(s)) ≈ 4 / 3 * π * 2^3
        @test sum(face_area(sys, g, Dim(1), I) for I in _interior(s) if I[1] == 11) ≈ 4π * 2^2
        @test sum(face_area(sys, g, Dim(2), I) for I in _interior(s) if I[2] == 4) ≈ π * 2^2   # equatorial disk
        @test sum(face_area(sys, g, Dim(3), I) for I in _interior(s) if I[3] == 2) ≈ π * 2^2 / 2 # meridian half-disk
        # distances: Δr, r Δθ, r sin θ Δφ
        I = CartesianIndex(5, 4, 3)
        c, w = cell_center(sys, g, I), cell_width(sys, g, I)
        @test face_distance(sys, g, Dim(1), I) ≈ w[1]
        @test face_distance(sys, g, Dim(2), I) ≈ c[1] * w[2]
        @test face_distance(sys, g, Dim(3), I) ≈ c[1] * sin(c[2]) * w[3]
        s2 = LocalHaloArray(Float64, (10, 6), 1)
        g2 = cell_geometry(s2, (UniformAxis(0, 2), UniformAxis(0, π)); system=Spherical())
        @test sum(cell_volume(sys, g2, I) for I in _interior(s2)) ≈ 4 / 3 * π * 2^3 / (2π)
        s1 = LocalHaloArray(Float64, (10,), 1)
        g1 = cell_geometry(s1, UniformAxis(0, 2); system=Spherical())
        @test sum(cell_volume(sys, g1, I) for I in _interior(s1)) ≈ 2^3 / 3
    end

    @testset "map_dims: static directions, inlined do-block" begin
        @test map_dims(D -> D, Val(3)) === (Dim(1), Dim(2), Dim(3))
        @test map_dims(identity, Val(0)) === ()
        @test map_dims(D -> unit_vector(Val(2), D), Val(2)) === unit_vector(Val(2))
        bc = ((Reflecting(), Reflecting()), (Periodic(), Periodic()))
        u  = LocalHaloArray(Float64, (12, 8), 1; boundary_condition=bc)
        t  = ThreadedHaloArray(Float64, (12, 4), 1; dims=(1, 2), boundary_condition=bc)
        ax = (EdgeAxis(cell_edges((0, 0.3, 1), (4, 8))), UniformAxis(0, 2π))
        g, gt = cell_geometry(u, ax; system=Polar()), cell_geometry(t, ax; system=Polar())
        sys = Polar()
        # the flux sum written with the do-block …
        flux_do(g, I, tile) = sum(map_dims(Val(2)) do D
            Im = I - unit_vector(Val(2), D)
            face_area(sys, g, D, I, tile) / face_distance(sys, g, D, I, tile) +
            face_area(sys, g, D, Im, tile) / face_distance(sys, g, D, Im, tile)
        end)
        # … equals the hand-written static recursion bit for bit
        rec(g, I, tile, ::Val{0}) = 0.0
        rec(g, I, tile, ::Val{D}) where {D} = (Im = I - unit_vector(Val(2), D);
            face_area(sys, g, Dim(D), I, tile) / face_distance(sys, g, Dim(D), I, tile) +
            face_area(sys, g, Dim(D), Im, tile) / face_distance(sys, g, Dim(D), Im, tile) +
            rec(g, I, tile, Val(D - 1)))
        for I in (CartesianIndex(2, 2), CartesianIndex(7, 5), CartesianIndex(13, 5))   # inside both the block and a tile
            @test flux_do(g, I, nothing) == rec(g, I, nothing, Val(2))
            @test flux_do(gt, I, 2) == rec(gt, I, 2, Val(2))
        end
        # and stays allocation-free on a single block and on a tile
        @test _allocation_free(flux_do, g, CartesianIndex(7, 5), nothing)
        @test _allocation_free(flux_do, gt, CartesianIndex(7, 5), 2)
    end

    @testset "helpers are allocation-free (single block and tile)" begin
        u = LocalHaloArray(Float64, (6, 5, 4), 1)
        t = ThreadedHaloArray(Float64, (6, 5, 2), 1; dims=(1, 1, 2))
        for (sys, axes3) in ((Cartesian(), (UniformAxis(0, 1), UniformAxis(0, 1), UniformAxis(0, 1))),
                             (Cylindrical(), (UniformAxis(0, 1), UniformAxis(0, 2π), UniformAxis(0, 1))),
                             (Spherical(), (UniformAxis(0, 1), UniformAxis(0, π), UniformAxis(0, 2π))))
            f(g, I, tile) = cell_volume(sys, g, I, tile) + face_area(sys, g, Dim(1), I, tile) +
                face_area(sys, g, Dim(2), I, tile) + face_area(sys, g, Dim(3), I, tile) +
                face_distance(sys, g, Dim(2), I, tile) + sum(face_center(sys, g, Dim(1), I, tile)) +
                sum(cell_center(sys, g, I, tile) + cell_width(sys, g, I, tile) + face_normal(sys, g, Dim(3), I, tile))
            g  = cell_geometry(u, axes3; system=sys)
            gt = cell_geometry(t, axes3; system=sys)
            I = CartesianIndex(3, 3, 3)
            @test _allocation_free(f, g, I, nothing)
            @test _allocation_free(f, gt, I, 2)
            @test f(gt, I, 2) ≈ f(g, CartesianIndex(3, 3, 5), nothing)   # tile 2 starts at global z = 3
        end
    end
end
