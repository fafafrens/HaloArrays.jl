using Test
using MPI
using HaloArrays

# One constructor per family: the single kinds take `storage=` and normalize
# integer arguments; the collection constructors resolve the boundary
# conditions and pass the topology and every other keyword down to the field
# constructor, for every backend.

sevens(T, dims...) = fill(T(7), dims...)
self1d = CartesianTopology(MPI.COMM_SELF, (1,); periodic=(true,))

@testset "single kinds: storage= and integer normalization" begin
    u = LocalHaloArray(Float32, (3,), 1; boundary_condition=:periodic, storage=sevens)
    @test parent(u) == fill(7.0f0, 5) && eltype(u) === Float32
    v = LocalHaloArray((Int32(3), Int32(2)), Int8(1); storage=sevens)      # Float64 default, Int normalization
    @test eltype(v) === Float64 && halo_width(v) == 1 && size(v) == (3, 2) && all(==(7.0), parent(v))

    t = ThreadedHaloArray(Float32, (2,), 1; dims=(2,), boundary_condition=:periodic, storage=sevens)
    @test all(tile -> tile == fill(7.0f0, 4), t.data) && tile_count(t) == 2

    h = HaloArray(Float32, (3,), 1, self1d; boundary_condition=:periodic, storage=sevens)
    @test parent(h) == fill(7.0f0, 5) && eltype(h) === Float32
    hd = HaloArray((3,), 1, self1d; boundary_condition=:periodic)           # Float64 forwarder, positional topology
    @test eltype(hd) === Float64 && hd.topology === self1d
    hw = HaloArray(fill(1.0, 5), 1, self1d, :periodic)                       # wrap existing storage
    @test parent(hw) == fill(1.0, 5)
end

@testset "MultiHaloArray: one method, keywords pass through" begin
    s = MultiHaloArray(LocalHaloArray, Float32, (3,), 1; fields=(:a, :b), boundary_condition=:periodic, storage=sevens)
    @test propertynames(s) == (:a, :b) && parent(s.a) == fill(7.0f0, 5) && eltype(s) === Float32
    sd = MultiHaloArray(LocalHaloArray, (3,), 1; boundary_conditions=(a=:periodic, b=:repeating))
    @test eltype(sd) === Float64 && sd.b.boundary_condition[1][1] isa Repeating
    th = MultiHaloArray(ThreadedHaloArray, (2,), 1; dims=(2,), fields=(:a,), boundary_condition=:periodic,
                        thread_backend=SerialBackend(), storage=sevens)
    @test thread_backend(th.a) isa SerialBackend && all(==(7.0), th.a.data[2])
    m = MultiHaloArray(HaloArray, Float64, (3,), 1, self1d; fields=(:a, :b), boundary_condition=:periodic)
    @test m.a isa HaloArray && m.a.topology === self1d
    @test_throws ArgumentError MultiHaloArray(LocalHaloArray, Float64, (3,), 1)   # neither fields nor boundary_conditions
end

@testset "ArrayOfHaloArray: shape from field_shape or from the conditions array" begin
    a = ArrayOfHaloArray(LocalHaloArray, Float32, (2,), (3,), 1; boundary_condition=:periodic, storage=sevens)
    @test field_shape(a) == (2,) && parent(a[2]) == fill(7.0f0, 5)
    b = ArrayOfHaloArray(LocalHaloArray, Float32, (3,), 1; boundary_conditions=[:periodic, :repeating, :periodic])
    @test field_shape(b) == (3,) && b[2].boundary_condition[1][1] isa Repeating
    c = ArrayOfHaloArray(ThreadedHaloArray, (2, 2), (2,), 1; dims=(2,), storage=sevens)   # Float64 forwarder
    @test field_shape(c) == (2, 2) && eltype(c) === Float64 && all(==(7.0), c[2, 2].data[1])
    d = ArrayOfHaloArray(HaloArray, Float64, (2,), (3,), 1, self1d; boundary_condition=:periodic)   # MPI with field_shape
    @test field_shape(d) == (2,) && d[1] isa HaloArray
    e = ArrayOfHaloArray(HaloArray, Float64, (3,), 1, self1d; boundary_conditions=fill(:periodic, 2, 1))
    @test field_shape(e) == (2, 1)
    @test_throws DimensionMismatch ArrayOfHaloArray(LocalHaloArray, Float32, (2, 2), (3,), 1;
        boundary_conditions=fill(:repeating, 3))
end
