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
end
