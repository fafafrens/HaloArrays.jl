using HaloArrays
using Test
import HaloArrays: ghost_origin

# Collect every physical-boundary ghost value (interior span in the off-axis dims),
# so two backends over the same global domain can be compared.
function _boundary_ghost_values(u::LocalHaloArray)
    vals = Float64[]
    for d in 1:2, s in 1:2
        append!(vals, vec(collect(ghost_view(u, Side(s), Dim(d)))))
    end
    sort(vals)
end
function _boundary_ghost_values(u::ThreadedHaloArray)
    vals = Float64[]
    for t in 1:tile_count(u), d in 1:2, s in 1:2
        neighbor_tile_id(u, t, d, s) == 0 || continue   # physical edges only
        append!(vals, vec(collect(ghost_view(u, Side(s), Dim(d), t))))
    end
    sort(vals)
end

@testset "FunctionBC" begin

    @testset "reproduces built-ins bit-for-bit (Local, hw=$hw)" for hw in (1, 2)
        data = reshape(collect(1.0:64.0), 8, 8)
        for (builtin, rule) in ((Reflecting(), (u, k) -> u(k - 1)),
                                (Antireflecting(), (u, k) -> -u(k - 1)),
                                (Repeating(), (u, k) -> u(0)),
                                (Extrapolation(1), (u, k) -> (1 + k) * u(0) - k * u(1)),
                                (Periodic(), (u, k) -> far(u)(k - 1)))
            ub = LocalHaloArray(Float64, (8, 8), hw; boundary_condition = builtin)
            uf = LocalHaloArray(Float64, (8, 8), hw; boundary_condition = FunctionBC(rule))
            interior_view(ub) .= data; interior_view(uf) .= data
            boundary_condition!(ub); boundary_condition!(uf)
            @test parent(uf) == parent(ub)
        end
        # a bare function in the tuple is the same as FunctionBC
        ug = LocalHaloArray(Float64, (8, 8), hw; boundary_condition = (u, k) -> u(0))
        @test ug.boundary_condition[1][1] isa FunctionBC
    end

    @testset "custom Dirichlet + Neumann (Local)" begin
        # constant ghost value fills every face, interior untouched
        u = LocalHaloArray(Float64, (8, 8), 1; boundary_condition = FunctionBC((u, k) -> 9.0))
        interior_view(u) .= 1.0; boundary_condition!(u)
        @test all(interior_view(u) .== 1.0)
        for d in 1:2, s in 1:2
            @test all(ghost_view(u, Side(s), Dim(d)) .== 9.0)
        end
        # Dirichlet wall value g: reflection about the wall value
        g = 3.0
        ud = LocalHaloArray(Float64, (8, 8), 1; boundary_condition = FunctionBC((u, k) -> 2g - u(k - 1)))
        interior_view(ud) .= 1.0; boundary_condition!(ud)
        @test all(ghost_view(ud, Side(1), Dim(1)) .== 5.0)
        # zero-flux Neumann (ghost = wall cell) == Repeating at hw=1
        un = LocalHaloArray(Float64, (8, 8), 1; boundary_condition = FunctionBC((u, k) -> u(0)))
        ur = LocalHaloArray(Float64, (8, 8), 1; boundary_condition = Repeating())
        data = reshape(collect(1.0:64.0), 8, 8)
        interior_view(un) .= data; interior_view(ur) .= data
        boundary_condition!(un); boundary_condition!(ur)
        @test parent(un) == parent(ur)
    end

    @testset "allocation-free hot path (Local)" begin
        u = LocalHaloArray(Float64, (8, 8), 1; boundary_condition = FunctionBC((u, k) -> 9.0))
        interior_view(u) .= 1.0; boundary_condition!(u)         # warm up
        @test (@allocated boundary_condition!(u)) == 0
    end

    @testset "ghost_origin global indices (Local, hw=1)" begin
        u = LocalHaloArray(Float64, (4, 4), 1)
        @test ghost_origin(u, Side(1), Dim(1)) == CartesianIndex(0, 1)   # low-x:  x=0, y=1
        @test ghost_origin(u, Side(2), Dim(1)) == CartesianIndex(5, 1)   # high-x: x=N+1
        @test ghost_origin(u, Side(1), Dim(2)) == CartesianIndex(1, 0)   # low-y:  y=0
        @test ghost_origin(u, Side(2), Dim(2)) == CartesianIndex(1, 5)   # high-y: y=N+1
    end

    @testset "position-dependent BC: Local vs 2x2 Threaded agree" begin
        f(I) = 1000 * I[1] + I[2]                       # injective in the global index
        posbc = FunctionBC((u, k) -> Float64(f(Tuple(global_index(u)))))
        ul = LocalHaloArray(Float64, (4, 4), 1; boundary_condition = posbc)
        interior_view(ul) .= 0.0; boundary_condition!(ul)

        # per-tile (2,2) × dims (2,2) ⇒ global (4,4), matching the Local domain
        ut = ThreadedHaloArray(Float64, (2, 2), 1; dims = (2, 2), boundary_condition = posbc)
        for t in 1:tile_count(ut); interior_view(ut, t) .= 0.0; end
        boundary_condition!(ut)

        # Same global domain ⇒ identical set of boundary ghost values, which only
        # holds if the per-tile global offset is correct.
        @test _boundary_ghost_values(ut) == _boundary_ghost_values(ul)
        @test f((0, 1)) in _boundary_ghost_values(ul)   # low-x ghost of global cell (1,1)
        @test f((0, 1)) in _boundary_ghost_values(ut)
    end
end
