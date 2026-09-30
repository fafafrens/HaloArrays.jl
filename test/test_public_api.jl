using Test
using MPI
using HaloArrays

@testset "deprecated names still work and agree with the new spellings" begin
    u = LocalHaloArray(Float64, (4, 3), 1; boundary_condition=:periodic)
    fill!(u, 2.0)
    @test global_size(u) == size(u) == (4, 3)
    @test mapreduce_haloarray_dims(identity, +, u, 2) == sum(u; dims=2)
    old = LocalMultiHaloArray(Float64, (4, 3), 1; fields=(:a, :b), boundary_condition=:periodic)
    new = MultiHaloArray(LocalHaloArray, Float64, (4, 3), 1; fields=(:a, :b), boundary_condition=:periodic)
    @test propertynames(old) == propertynames(new) == (:a, :b)
    @test LocalMultiHaloArray((; a=u, b=copy(u))) isa MultiHaloArray
    t = ThreadedHaloArray(Float64, (2, 3), 1; dims=(2, 1), boundary_condition=:periodic)
    fill!(t, 1.0)
    @test ThreadedMultiHaloArray(Float64, (2, 3), 1; dims=(2, 1), fields=(:a,), boundary_condition=:periodic) isa MultiHaloArray
    @test synchronize_halo_threads!(t) === t
    @test halo_exchange_threads!(t) === t
    @test boundary_condition_threads!(t) === t
    @test synchronize_halo!(t; threads=true) === t
    @test synchronize_halo!(u; threads=true) === u          # accepted and ignored off-thread
    @test synchronize_halo!(new; threads=true) === new
end

@testset "public API exports" begin
    ha = HaloArray(Float64, (5,), 2; boundary_condition=:repeating)

    @test ha isa HaloArray
    @test ha isa AbstractDistributedHaloArray
    @test ha isa AbstractSingleHaloArray
    @test ha isa AbstractHaloArray
    @test ha isa AbstractArray{Float64,1}
    @test eltype(typeof(ha)) === Float64
    @test interior_size(ha) == (5,)
    @test interior_size(ha) == (5,)
    @test storage_size(ha) == (9,)
    @test halo_width(ha) == 2
    @test size(ha) == size(ha)
    @test axes(ha) == map(Base.OneTo, size(ha))
    @test interior_axes(ha) == axes(interior_view(ha))
    @test length(axes(ha, 1)) == size(ha, 1)
    @test length(interior_axes(ha, 1)) == interior_size(ha, 1)
    @test length(ha) == prod(size(ha))
    owned_first = ha.topology.cart_coords[1] * interior_size(ha, 1) + 1
    owned_last = owned_first + interior_size(ha, 1) - 1
    @test interior_to_global_index(ha, (1,)) == (owned_first,)
    @test interior_to_global_index(ha, (5,)) == (owned_last,)
    @test HaloArrays.global_to_storage_index(ha, (owned_first,)) == (3,)
    @test HaloArrays.global_to_storage_index(ha, (owned_last,)) == (7,)
    @test_throws BoundsError interior_to_global_index(ha, (0,))

    fill!(ha, 3.0)
    @test all(interior_view(ha) .== 3.0)

    fill_from_global_indices!(ha) do I
        I[1]
    end
    @test collect(interior_view(ha)) == collect(owned_first:owned_last)

    interior_view(ha)[1] = 4.0
    @test interior_view(ha)[1] == 4.0
    @test parent(ha)[3] == 4.0

    boundary_condition!(ha)
    if ha.topology.neighbors[1][1] == MPI.PROC_NULL
        @test parent(ha)[1:2] == [4.0, 4.0]
    end
    if ha.topology.neighbors[1][2] == MPI.PROC_NULL
        @test parent(ha)[8:9] == [owned_last, owned_last]
    end

    resized_global_size = ntuple(d -> 3 * ha.topology.dims[d], Val(ndims(ha)))
    resized = similar(ha, Float32, resized_global_size)
    @test resized isa HaloArray
    @test eltype(resized) === Float32
    @test size(resized) == resized_global_size
    @test size(resized) == size(resized)
    @test interior_size(resized) == (3,)
    @test storage_size(resized) == (7,)

    resized_same_eltype = similar(ha, resized_global_size)
    @test resized_same_eltype isa HaloArray
    @test eltype(resized_same_eltype) === Float64
    @test size(resized_same_eltype) == resized_global_size
    @test interior_size(resized_same_eltype) == (3,)

    local_ha = LocalHaloArray(Float64, (3,), 1; boundary_condition=:repeating)
    @test local_ha isa AbstractSerialHaloArray
    @test local_ha isa AbstractSingleHaloArray
    @test local_ha isa AbstractHaloArray
    @test local_ha isa AbstractArray{Float64,1}
    @test eltype(typeof(local_ha)) === Float64
    @test interior_size(local_ha) == (3,)
    @test size(local_ha) == interior_size(local_ha)
    @test axes(local_ha) == map(Base.OneTo, size(local_ha))
    @test interior_axes(local_ha) == axes(interior_view(local_ha))
    interior_view(local_ha) .= [1.0, 2.0, 3.0]
    @test start_halo_exchange!(local_ha) === local_ha
    @test finish_halo_exchange!(local_ha) === local_ha
    @test synchronize_halo!(local_ha) === local_ha
    @test parent(local_ha) == [1.0, 1.0, 2.0, 3.0, 3.0]
end
