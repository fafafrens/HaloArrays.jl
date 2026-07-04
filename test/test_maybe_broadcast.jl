using Test
using MPI
using HaloArrays

@testset "MaybeHaloArray" begin
    topology = CartesianTopology(MPI.COMM_SELF, (1,); periodic=(false,))
    ha = HaloArray(Int, (4,), 1, topology; boundary_condition=:repeating)
    ha_interior = interior_view(ha)
    for i in eachindex(ha_interior)
        ha_interior[i] = i
    end

    maybe = MaybeHaloArray(ha)
    @test maybe isa AbstractArray{Int,1}
    @test is_active(maybe)
    @test length(maybe) == 4
    @test parent(maybe) === ha
    @test eltype(maybe) === Int
    @test HaloArrays.unwrap(maybe) === ha
    @test halo_width(maybe) == 1

    shifted = maybe .+ 10
    @test shifted isa MaybeHaloArray
    @test is_active(shifted)
    @test collect(interior_view(HaloArrays.unwrap(shifted))) == [i + 10 for i in 1:4]

    dest = similar(maybe)
    dest .= 3 .* maybe
    @test collect(interior_view(HaloArrays.unwrap(dest))) == [3 * i for i in 1:4]

    copied_into = similar(maybe)
    fill!(copied_into, -1)
    @test copyto!(copied_into, maybe) === copied_into
    @test collect(interior_view(HaloArrays.unwrap(copied_into))) == [i for i in 1:4]

    zero_maybe = zero(maybe)
    @test zero_maybe isa MaybeHaloArray
    @test is_active(zero_maybe)
    @test all(==(0), HaloArrays.unwrap(zero_maybe))
    @test fill!(zero_maybe, 6) === zero_maybe
    @test all(==(6), HaloArrays.unwrap(zero_maybe))

    inactive_topology = HaloArrays.inactive_cartesian_topology((1,))
    inactive_ha = HaloArray(Int, (4,), 1, inactive_topology; boundary_condition=:repeating)
    inactive_maybe = MaybeHaloArray(inactive_ha)
    @test !is_active(inactive_maybe)
    @test length(inactive_maybe) == 0
    @test isempty(eachindex(inactive_maybe))
    @test_throws ErrorException HaloArrays.unwrap(inactive_maybe)

    inactive_result = inactive_maybe .+ 1
    @test inactive_result isa MaybeHaloArray
    @test !is_active(inactive_result)

    inactive_dest = similar(inactive_maybe)
    @test !is_active(inactive_dest)
    @test copyto!(inactive_dest, inactive_maybe) === inactive_dest
    @test fill!(inactive_dest, 2) === inactive_dest
    @test !is_active(zero(inactive_maybe))

    u = copy(ha)
    v = copy(ha)
    v .= 10 .* v
    multi = MultiHaloArray((; u, v))
    maybe_multi = MaybeHaloArray(multi)
    @test maybe_multi isa AbstractArray{Int,2}
    @test halo_width(maybe_multi) == 1

    shifted_multi = maybe_multi .+ 5
    @test shifted_multi isa MaybeHaloArray
    @test is_active(shifted_multi)
    shifted_fields = HaloArrays.unwrap(shifted_multi)
    @test collect(interior_view(shifted_fields.arrays.u)) == [i + 5 for i in 1:4]
    @test collect(interior_view(shifted_fields.arrays.v)) == [10 * i + 5 for i in 1:4]

    local_ha = LocalHaloArray(Int, (4,), 1; boundary_condition=:repeating)
    maybe_local = MaybeHaloArray(local_ha)
    @test is_active(maybe_local)
    @test halo_width(maybe_local) == 1
    interior_view(local_ha) .= [4, 5, 6, 7]
    @test eltype(typeof(maybe_local)) === Int
    @test maybe_local[3] == 6
    maybe_local[3] = 9
    @test maybe_local[3] == 9

    maybe_resized = similar(maybe_local, Float32, (6,))
    @test maybe_resized isa MaybeHaloArray
    @test is_active(maybe_resized)
    @test eltype(typeof(maybe_resized)) === Float32
    @test size(HaloArrays.unwrap(maybe_resized)) == (6,)
    @test storage_size(HaloArrays.unwrap(maybe_resized)) == (8,)

    threaded_ha = ThreadedHaloArray(Int, (2,), 1; dims=(2,), boundary_condition=:repeating)
    maybe_threaded = MaybeHaloArray(threaded_ha)
    @test is_active(maybe_threaded)
    @test halo_width(maybe_threaded) == 1

    @test_throws ArgumentError maybe .+ ha
    @test_throws ArgumentError ha .+ maybe
    @test_throws ArgumentError maybe_local .+ local_ha
    @test_throws ArgumentError local_ha .+ maybe_local
    @test_throws ArgumentError maybe_threaded .+ threaded_ha
    @test_throws ArgumentError threaded_ha .+ maybe_threaded
    @test_throws ArgumentError maybe_multi .+ multi
    @test_throws ArgumentError multi .+ maybe_multi

    u = LocalHaloArray(Float64, (4,), 1; boundary_condition=:repeating)
    @test is_active(HaloArrays.active(u))
    @test !is_active(HaloArrays.inactive(u))
    @test HaloArrays.getdata(MaybeHaloArray(u)) === u
end
