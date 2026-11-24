using Test, HaloArrays, MPI
using StaticArrays: SVector
MPI.Initialized() || MPI.Init()

# Measurement helpers. Every measured function returns `nothing`: on Julia 1.10
# a static vector returned through the `@allocated` / timing boundary is boxed
# (48 bytes for an SVector{4,Float64}), which is not an allocation of the read.
# Each helper sweeps the interior cells, so the read is not loop-invariant and
# cannot be hoisted, and accumulates into `acc`, allocated once outside.
const acc = Ref(0.0)

# Hand-written reference: the form `sitevector` replaces.
function read_hand!(acc, c, tile = nothing)
    s = 0.0
    @inbounds for I in interior_cells(CellRanges(c)); s += sum(SVector{4}(siteview(c, I, tile))); end
    acc[] = s; nothing
end
function read_static!(acc, c, tile = nothing)
    s = 0.0
    @inbounds for I in interior_cells(CellRanges(c)); s += sum(sitevector(c, I, tile)); end
    acc[] = s; nothing
end
function read_vector!(acc, c, tile = nothing)   # materialising a Vector, for the timing contrast
    s = 0.0
    @inbounds for I in interior_cells(CellRanges(c)); s += sum(collect(siteview(c, I, tile))); end
    acc[] = s; nothing
end
# A kernel in the intended style: read a static vector, write back through the view.
function kernel!(du, u, I)
    @inbounds begin
        q = sitevector(u, I)
        siteview(du, I) .-= 2 .* q
    end
    return nothing
end

# The tile-storage forms in a loop that reads and writes: the tile is resolved once.
function tile_kernel!(du, u)
    tile_foreach(u) do t
        s, d = tile_parent(u, t), tile_parent(du, t)
        @inbounds for I in interior_cells(CellRanges(u, t))
            q = sitevector(s, I)
            siteview(d, I) .-= 2 .* q
        end
    end
    return nothing
end

# Minimum time of f(args...) over `reps` evaluations, in ns per site read (the
# helpers read every interior cell). Stable to a few percent; the comparisons
# below use generous factors so a loaded CI machine does not fail them.
function min_ns_per_read(f, c, args...; reps = 200, inner = 20)
    n = length(interior_cells(CellRanges(c)))
    best = Inf
    for _ in 1:reps
        t = time_ns()
        for _ in 1:inner
            f(acc, c, args...)
        end
        best = min(best, (time_ns() - t) / (inner * n))
    end
    return best
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
        uu = MultiHaloArray((; u, c))                        # two levels deep: 10 + 4
        @test (@inferred sitevector(uu, I)) isa SVector{14,Float64}

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

        # Array-backed field containers have a runtime field count: a clear error,
        # on their own and nested inside a MultiHaloArray (which is a valid state).
        arr = ArrayOfHaloArray(LocalHaloArray, Float64, (4,), (6, 5), 1; boundary_condition=:periodic)
        @test_throws ArgumentError sitevector(arr, I)
        nested_arr = MultiHaloArray((; q = ArrayOfHaloArray([copy(c.E), copy(c.Mx)]), p = arr))
        @test siteview(nested_arr, I) isa AbstractArray          # siteview itself is fine
        @test_throws ArgumentError sitevector(nested_arr, I)
        err = try sitevector(nested_arr, I); nothing catch e; e end
        @test occursin("runtime number of fields", err.msg)
    end

    @testset "allocation" begin
        u = MultiHaloArray((; c, w))
        t = MultiHaloArray(ThreadedHaloArray, Float64, (6, 5), 1; dims=(1, 1),
                           fields=(:a, :b, :c), boundary_condition=:periodic)
        du = similar(c); fill!(du, 0.0)
        read_static!(acc, c); read_static!(acc, u); read_static!(acc, t, 1); kernel!(du, c, I)
        @test @allocated(read_static!(acc, c)) == 0
        @test @allocated(read_static!(acc, u)) == 0                    # nested 4 + 6
        @test @allocated(read_static!(acc, t, 1)) == 0                 # threaded, with tile
        @test @allocated(kernel!(du, c, I)) == 0
        @test siteview(du, I) == -4 .* siteview(c, I)
    end

    @testset "timing: no slower than the hand-written SVector{N}(siteview)" begin
        # Both should compile to N loads; allow a factor 3 for measurement noise.
        read_hand!(acc, c); read_static!(acc, c); read_vector!(acc, c)
        hand   = min_ns_per_read(read_hand!, c)
        static = min_ns_per_read(read_static!, c)
        @test static <= 3 * hand + 2.0           # +2 ns absolute slack at the ns scale
        # And well under the cost of materialising a Vector from the view.
        @test static < min_ns_per_read(read_vector!, c)
        @info "sitevector timing" hand_ns = hand sitevector_ns = static
    end

    @testset "on tile storage: tile_parent(state, t)" begin
        t = MultiHaloArray(ThreadedHaloArray, Float64, (6, 5), 1; dims=(2, 1),
                           fields=(:a, :b, :c), boundary_condition=:periodic)
        fill_from_global_indices!(J -> (J[1] + 10J[2]) .* (1.0, 2.0, 3.0), t)
        It = first(interior_cells(CellRanges(t)))

        # Same values as the three-argument forms, on every tile.
        for tile in 1:tile_count(t)
            s = tile_parent(t, tile)
            q = @inferred sitevector(s, It)
            @test q isa SVector{3,Float64}
            @test q === sitevector(t, It, tile)
            v = @inferred siteview(s, It)
            @test v == siteview(t, It, tile)
            @test size(v) == (3,) && eltype(v) == Float64
            @test parent(v) === s
        end

        # Writes reach the fields; a tuple of arrays works like a NamedTuple.
        s = tile_parent(t, 2)
        siteview(s, It) .= SVector(1.0, 2.0, 3.0)
        @test sitevector(t, It, 2) === SVector(1.0, 2.0, 3.0)
        siteview(Tuple(s), It) .+= SVector(1.0, 1.0, 1.0)
        @test sitevector(Tuple(s), It) === SVector(2.0, 3.0, 4.0)
        siteview(s, It)[2] = -1.0
        @test tile_parent(t.b, 2)[It] == -1.0
        @test_throws BoundsError siteview(s, It)[4]

        # One array: a one-element vector. Local and MPI blocks have one tile.
        @test (@inferred sitevector(s.a, It)) === SVector(2.0)
        siteview(s.a, It) .= 5.0
        @test tile_parent(t.a, 2)[It] == 5.0
        @test sitevector(tile_parent(c, 1), I) === sitevector(c, I)

        # Nested like the collection, leaves in siteview order; mixed types promote.
        u = MultiHaloArray((; c, w))
        su = tile_parent(u, 1)
        @test (@inferred sitevector(su, I)) === sitevector(u, I)
        @test siteview(su, I) == siteview(u, I)
        siteview(su, I)[5] = -10.0
        @test parent(w.pixx)[I] == -10.0
        parent(w.pixx)[I] = 10.0
        f32 = LocalHaloArray(Float32, (6, 5), 1; boundary_condition=:periodic)
        mixed = (parent(c.E), (parent(f32), parent(c.D)))
        @test (@inferred sitevector(mixed, I)) isa SVector{3,Float64}
        @test eltype(siteview(mixed, I)) == Float64
        siteview(mixed, I)[2] = 0.5
        @test parent(f32)[I] === 0.5f0

        # Aliasing rules of siteview hold.
        v = siteview(s, It)
        @test_throws ArgumentError v .= view(v, 3:-1:1)
        v .= copy(view(v, 3:-1:1))
        @test v == [4.0, -1.0, 5.0]

        # Halo arrays keep their own methods; an array of storages is refused.
        @test_throws ArgumentError sitevector(t, It)
        arr = ArrayOfHaloArray(LocalHaloArray, Float64, (4,), (6, 5), 1; boundary_condition=:periodic)
        @test_throws ArgumentError siteview(tile_parent(arr, 1), I)
        @test_throws ArgumentError sitevector(tile_parent(arr, 1), I)
        @test_throws ArgumentError sitevector((; p = tile_parent(arr, 1)), I)

        # A read-and-write tile loop: same result as the three-argument forms, no allocation.
        for backend in (SerialBackend(), ThreadsBackend())
            x  = MultiHaloArray(ThreadedHaloArray, Float64, (6, 5), 1; dims=(2, 2),
                                fields=(:a, :b, :c), boundary_condition=:periodic,
                                thread_backend=backend)
            fill_from_global_indices!(J -> (J[1] - J[2]) .* (1.0, 2.0, 3.0), x)
            dx = similar(x); fill!(dx, 0.0)
            ref = similar(x); fill!(ref, 0.0)
            tile_kernel!(dx, x)
            for tile in 1:tile_count(x), J in interior_cells(CellRanges(x, tile))
                siteview(ref, J, tile) .-= 2 .* sitevector(x, J, tile)
            end
            @test all(interior_view(dx.b, k) == interior_view(ref.b, k) for k in 1:tile_count(x))
            @test gather_haloarray(dx.c) == gather_haloarray(ref.c)
        end
        xs = MultiHaloArray(ThreadedHaloArray, Float64, (6, 5), 1; dims=(2, 2),
                            fields=(:a, :b, :c), boundary_condition=:periodic,
                            thread_backend=SerialBackend())
        dxs = similar(xs); fill!(dxs, 0.0)
        dl = similar(c); fill!(dl, 0.0)
        tile_kernel!(dxs, xs); tile_kernel!(dl, c)
        @test @allocated(tile_kernel!(dxs, xs)) == 0
        @test @allocated(tile_kernel!(dl, c)) == 0
    end
end
