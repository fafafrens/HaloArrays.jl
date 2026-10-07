using Test, HaloArrays, MPI
using StaticArrays: SVector
MPI.Initialized() || MPI.Init()

# Minimum time over `reps` evaluations of f(args...), in ns. Enough repetitions
# to be stable to a few percent; the comparisons below use generous factors so
# a loaded CI machine does not fail them.
function min_ns(f, args...; reps = 200, inner = 1000)
    best = Inf
    for _ in 1:reps
        t = time_ns()
        for _ in 1:inner
            f(args...)
        end
        best = min(best, (time_ns() - t) / inner)
    end
    return best
end

# Hand-written reference: the form `sitevector` replaces.
read_hand(c, I) = @inbounds SVector{4}(siteview(c, I))
read_static(c, I) = @inbounds sitevector(c, I)
read_static_tile(c, I, tile) = @inbounds sitevector(c, I, tile)
read_nested(u, I) = @inbounds sitevector(u, I)
# A kernel in the intended style: read a static vector, write back through the view.
function kernel!(du, u, I)
    @inbounds begin
        q = sitevector(u, I)
        siteview(du, I) .-= 2 .* q
    end
    return nothing
end

@testset "sitevector" begin
    c = MultiHaloArray(LocalHaloArray, Float64, (6, 5), 1;
                       fields=(:E, :Mx, :My, :D), boundary_condition=:periodic)
    w = MultiHaloArray(LocalHaloArray, Float64, (6, 5), 1;
                       fields=(:pixx, :piyy, :pixy, :piB, :nux, :nuy), boundary_condition=:periodic)
    I = first(interior_cells(CellRanges(c)))
    siteview(c, I) .= [1.0, 2.0, 3.0, 4.0]
    siteview(w, I) .= 10.0:10.0:60.0

    @testset "values and static size from the type" begin
        q = @inferred sitevector(c, I)
        @test q isa SVector{4,Float64}
        @test q == SVector(1.0, 2.0, 3.0, 4.0)
        @test q == siteview(c, I)
        @test sitevector(c, I, 1) == q                     # Local collections accept tile 1
        # Independent copy: later writes to the fields do not change it.
        parent(c.E)[I] = 100.0
        @test q[1] == 1.0
        @test sitevector(c, I)[1] == 100.0
        parent(c.E)[I] = 1.0

        # Nested collection: leaf count is the sum over children, siteview order.
        u = MultiHaloArray((; c, w))
        qu = @inferred sitevector(u, I)
        @test qu isa SVector{10,Float64}
        @test qu == vcat(sitevector(c, I), sitevector(w, I))
        @test qu == collect(siteview(u, I))

        # Single halo array: one-element vector.
        s = LocalHaloArray(Float64, (6, 5), 1; boundary_condition=:periodic)
        parent(s)[I] = 7.0
        @test (@inferred sitevector(s, I)) === SVector(7.0)

        # Mixed element types promote like siteview.
        m = MultiHaloArray((; a = c.E, b = LocalHaloArray(Float32, (6, 5), 1; boundary_condition=:periodic)))
        @test (@inferred sitevector(m, I)) isa SVector{2,Float64}

        # Threaded storage needs the tile, like siteview.
        t = MultiHaloArray(ThreadedHaloArray, Float64, (6, 5), 1; dims=(1, 1),
                           fields=(:a, :b, :c), boundary_condition=:periodic)
        It = first(interior_cells(CellRanges(t)))
        siteview(t, It, 1) .= [1.0, 2.0, 3.0]
        @test (@inferred sitevector(t, It, 1)) === SVector(1.0, 2.0, 3.0)
        @test_throws ArgumentError sitevector(t, It)

        # Array-backed field containers have a runtime field count: no method.
        arr = ArrayOfHaloArray(LocalHaloArray, Float64, (4,), (6, 5), 1; boundary_condition=:periodic)
        @test_throws MethodError sitevector(arr, I)
        # ...unless the container is a tuple, which the nested case covers.
    end

    @testset "allocation" begin
        read_static(c, I); read_hand(c, I)
        @test @allocated(read_static(c, I)) == 0
        u = MultiHaloArray((; c, w)); read_nested(u, I)
        @test @allocated(read_nested(u, I)) == 0
        t = MultiHaloArray(ThreadedHaloArray, Float64, (6, 5), 1; dims=(1, 1),
                           fields=(:a, :b, :c), boundary_condition=:periodic)
        It = first(interior_cells(CellRanges(t))); read_static_tile(t, It, 1)
        @test @allocated(read_static_tile(t, It, 1)) == 0
        du = similar(c); fill!(du, 0.0); kernel!(du, c, I)
        @test @allocated(kernel!(du, c, I)) == 0
        @test siteview(du, I) == -4 .* siteview(c, I)
    end

    @testset "timing: no slower than the hand-written SVector{N}(siteview)" begin
        # Both should compile to N loads; allow a factor 3 for measurement noise.
        hand   = min_ns(read_hand, c, I)
        static = min_ns(read_static, c, I)
        @test static <= 3 * hand + 2.0           # +2 ns absolute slack at the ns scale
        # And well under the cost of materialising a Vector from the view.
        read_vector(c, I) = @inbounds collect(siteview(c, I))
        @test static < min_ns(read_vector, c, I)
        @info "sitevector timing" hand_ns = hand sitevector_ns = static
    end
end
