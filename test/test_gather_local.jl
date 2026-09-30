using Test
using MPI
using HaloArrays

MPI.Initialized() || MPI.Init()

@testset "gather_haloarray on serial backends and collections" begin
    expected = [10.0 * i + j for i in 1:4, j in 1:3]

    u = LocalHaloArray(Float64, (4, 3), 1; boundary_condition=:periodic)
    HaloArrays.fill_from_global_indices!(I -> 10.0 * I[1] + I[2], u)
    A = gather_haloarray(u)
    @test A isa Matrix{Float64}
    @test A == expected
    @test gather_haloarray(u; root=0) == expected      # root is accepted and ignored
    A[1, 1] = -1.0                                       # a copy, not a view of the storage
    @test interior_view(u)[1, 1] == 11.0

    # Tiles are stitched in global order.
    t = ThreadedHaloArray(Float64, (2, 3), 1; dims=(2, 1), boundary_condition=:periodic)
    HaloArrays.fill_from_global_indices!(I -> 10.0 * I[1] + I[2], t)
    @test gather_haloarray(t) == expected
    t3 = ThreadedHaloArray(Float64, (2, 3, 2), 1; dims=(2, 1, 2), boundary_condition=:periodic)
    HaloArrays.fill_from_global_indices!(I -> 100.0 * I[1] + 10.0 * I[2] + I[3], t3)
    @test gather_haloarray(t3) == [100.0 * i + 10.0 * j + k for i in 1:4, j in 1:3, k in 1:4]

    # Collections: field axes first, then the spatial axes.
    aoh = ArrayOfHaloArray(LocalHaloArray, Float64, (2,), (4, 3), 1; boundary_condition=:periodic)
    for k in 1:2
        HaloArrays.fill_from_global_indices!(I -> k * (10.0 * I[1] + I[2]), parent(aoh)[k])
    end
    G = gather_haloarray(aoh)
    @test size(G) == (2, 4, 3)
    @test G[1, :, :] == expected
    @test G[2, :, :] == 2 .* expected

    @test gather_haloarray(aoh; root=1) == G          # root is ignored on serial collections too
    m = MultiHaloArray((rho=u, p=parent(aoh)[2]))
    @test gather_haloarray(m; root=1).p == 2 .* expected
    nt = gather_haloarray(m)
    @test keys(nt) == (:rho, :p)
    @test nt.rho == expected
    @test nt.p == 2 .* expected

    # A single-rank MPI array gives the same array as its serial twin.
    topo = CartesianTopology(MPI.COMM_SELF, (1, 1); periodic=(true, true))
    h = HaloArray(Float64, (4, 3), 1, topo; boundary_condition=:periodic)
    HaloArrays.fill_from_global_indices!(I -> 10.0 * I[1] + I[2], h)
    @test gather_haloarray(h) == expected
end
