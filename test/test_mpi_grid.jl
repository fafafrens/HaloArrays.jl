using Test
using MPI
using HaloArrays

# Cell geometry on a distributed HaloArray: every rank fills its interior from
# the global index and one synchronize completes the ghosts (exchange across
# ranks and periodic edges, FunctionBC on physical edges) — so the local
# storages must equal the corresponding window of the serial geometry built
# with the same periodicity.

@testset "MPI cell geometry" begin
    comm = MPI.COMM_WORLD
    nr   = MPI.Comm_size(comm)
    @test nr > 1

    owned = (6, 5)
    GX, GY = owned[1] * nr, owned[2]
    axes2 = (EdgeAxis(cell_edges((0, 0.5, 2), (GX ÷ 2, GX - GX ÷ 2))), UniformAxis(-1, 1))

    for periodic in (false, true)
        topo = CartesianTopology(comm, (nr, 1); periodic=(periodic, false))
        bc   = ((periodic ? Periodic() : Reflecting(), periodic ? Periodic() : Reflecting()),
                (Repeating(), Repeating()))
        u = HaloArray(Float64, owned, 1, topo; boundary_condition=bc)
        g = cell_geometry(u, axes2; system=Cylindrical())
        @test propertynames(g) == (:r, :z, :hr, :hz)
        @test g.r isa HaloArray

        ref = cell_geometry(LocalHaloArray(Float64, (GX, GY), 1; boundary_condition=bc), axes2;
                            system=Cylindrical())
        rs  = field_storages(ref)
        gs  = field_storages(g)
        origin = interior_to_global_index(u, (1, 1))
        noncorner = (I for I in CartesianIndices(gs.r)
            if count(d -> !(1 < I[d] <= size(gs.r, d) - 1), 1:2) <= 1)   # corners are never filled
        for I in noncorner
            J = CartesianIndex(origin .+ Tuple(I) .- 1)   # ref storage index (same halo width)
            @test gs.r[I] == rs.r[J]
            @test gs.z[I] == rs.z[J]
            @test gs.hr[I] == rs.hr[J]
            @test gs.hz[I] == rs.hz[J]
        end
        # the global volume is the sum of the ranks' interior volumes
        V = sum(cell_volume(g, I) for I in CartesianIndices(interior_range(u)))
        @test MPI.Allreduce(V, +, comm) ≈ 2^2 / 2 * 2
        # the geometry's boundary condition mirrors the topology's periodicity
        if periodic
            @test g.r.boundary_condition[1] == (Periodic(), Periodic())
        else
            @test g.r.boundary_condition[1][1] isa FunctionBC
        end
    end
end
