using Test
using MPI
using HaloArrays

@testset "MaybeHaloArray and MaybeActive" begin
    topology = CartesianTopology(MPI.COMM_SELF, (1,); periodic=(false,))
    ha = HaloArray(Int, (4,), 1, topology; boundary_condition=:repeating)
    for i in eachindex(ha)
        ha[i] = i
    end

    maybe = MaybeHaloArray(ha)
    @test isactive(maybe)
    @test length(maybe) == 4
    @test unwrap(maybe) === ha

    shifted = maybe .+ 10
    @test shifted isa MaybeHaloArray
    @test isactive(shifted)
    @test collect(interior_view(unwrap(shifted))) == [i + 10 for i in 1:4]

    dest = similar(maybe)
    dest .= 3 .* maybe
    @test collect(interior_view(unwrap(dest))) == [3 * i for i in 1:4]

    inactive_topology = HaloArrays.inactive_cartesian_topology((1,))
    inactive_ha = HaloArray(Int, (4,), 1, inactive_topology; boundary_condition=:repeating)
    inactive_maybe = MaybeHaloArray(inactive_ha)
    @test !isactive(inactive_maybe)
    @test length(inactive_maybe) == 0
    @test_throws ErrorException unwrap(inactive_maybe)

    inactive_result = inactive_maybe .+ 1
    @test inactive_result isa MaybeHaloArray
    @test !isactive(inactive_result)

    u = copy(ha)
    v = copy(ha)
    v .= 10 .* v
    multi = MultiHaloArray((; u, v); check=true)
    maybe_multi = MaybeHaloArray(multi)

    shifted_multi = maybe_multi .+ 5
    @test shifted_multi isa MaybeHaloArray
    @test isactive(shifted_multi)
    shifted_fields = unwrap(shifted_multi)
    @test collect(interior_view(shifted_fields.arrays.u)) == [i + 5 for i in 1:4]
    @test collect(interior_view(shifted_fields.arrays.v)) == [10 * i + 5 for i in 1:4]

    active_value = active(7)
    inactive_value = inactive(7)
    @test isactive(active_value)
    @test !isactive(inactive_value)
    @test get(active_value, 0) == 7
    @test get(inactive_value, 0) == 0
    @test unsafe_get(inactive_value) == 7
end
