using Test
using MPI
using HaloArrays
using LinearAlgebra: dot, norm

function _periodic_bc(::Val{N}) where {N}
    return ntuple(_ -> (Periodic(), Periodic()), Val(N))
end

@testset "MPI reductions" begin
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    @test nranks > 1

    topology = CartesianTopology(comm, (0,); periodic=(true,))
    ha = HaloArray(Float64, (4,), 1, topology; boundary_condition=_periodic_bc(Val(1)))

    ha_interior = interior_view(ha)
    for i in eachindex(ha_interior)
        ha_interior[i] = rank + i / 10
    end

    local_sum = sum(interior_view(ha))
    @test mapreduce(identity, +, ha) ≈ MPI.Allreduce(local_sum, MPI.SUM, topology.cart_comm)
    @test reduce(+, ha) ≈ MPI.Allreduce(local_sum, MPI.SUM, topology.cart_comm)
    @test sum(ha) ≈ mapreduce(identity, +, ha)
    @test sum(abs2, ha) ≈ mapreduce(abs2, +, ha)
    # init= is seeded once globally, not once per rank (would over-count by
    # (nranks-1)*init across the Allreduce).
    @test mapreduce(identity, +, ha; init=1000.0) ≈ 1000.0 + mapreduce(identity, +, ha)
    @test sum(ha; init=1000.0) ≈ 1000.0 + sum(ha)
    @test maximum(ha) ≈ mapreduce(identity, max, ha)
    @test minimum(ha) ≈ mapreduce(identity, min, ha)

    # dot is a global reduction (Allreduce), consistent with norm
    wv = similar(ha)
    interior_view(wv) .= 2.0
    local_dot = sum(interior_view(ha) .* interior_view(wv))
    @test dot(ha, wv) ≈ MPI.Allreduce(local_dot, MPI.SUM, topology.cart_comm)
    @test dot(ha, ha) ≈ norm(ha)^2
    @test any(x -> x < 0, ha) == false
    @test all(x -> x >= 0, ha) == true

    maybe_ha = MaybeHaloArray(ha)
    @test sum(maybe_ha) ≈ mapreduce(identity, +, maybe_ha)
    @test maximum(maybe_ha) ≈ mapreduce(identity, max, maybe_ha)
    @test minimum(maybe_ha) ≈ mapreduce(identity, min, maybe_ha)

    u = copy(ha)
    v = similar(ha)
    v_interior = interior_view(v)
    for i in eachindex(v_interior)
        v_interior[i] = 10 * rank + i
    end
    fields = MultiHaloArray((; u, v))

    local_field_sum = sum(interior_view(u)) + sum(interior_view(v))
    @test mapreduce(identity, +, fields) ≈ MPI.Allreduce(local_field_sum, MPI.SUM, topology.cart_comm)
    @test all(x -> x >= 0, fields)
    @test any(x -> x == 1, fields)

    array_fields = ArrayOfHaloArray([u, v])
    @test mapreduce(identity, +, array_fields) ≈ MPI.Allreduce(local_field_sum, MPI.SUM, topology.cart_comm)
    @test sum(array_fields) ≈ mapreduce(identity, +, array_fields)
    @test maximum(array_fields) ≈ mapreduce(identity, max, array_fields)
    @test minimum(array_fields) ≈ mapreduce(identity, min, array_fields)
    @test all(x -> x >= 0, array_fields)
    @test any(x -> x == 1, array_fields)
end

@testset "MPI dimension reductions" begin
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    topology = CartesianTopology(comm, (0, 0); periodic=(true, true))
    halo = 1
    local_dims = (2, 3)

    ha = HaloArray(Int, local_dims, halo, topology; boundary_condition=_periodic_bc(Val(2)))
    fill!(parent(ha), -10_000)
    ha_interior = interior_view(ha)
    for i in 1:local_dims[1], j in 1:local_dims[2]
        ha_interior[i, j] = 1000 * rank + 10 * i + j
    end

    # `dims=` through mapreduce/sum/maximum routes to a DimReductionPlan cached
    # on the topology and must agree with the explicit mapreduce_haloarray_dims
    # path; order-sensitive folds stay rejected.
    kw_sum = sum(ha; dims=1)
    kw_ref = mapreduce_haloarray_dims(identity, +, ha, 1)
    @test kw_sum isa MaybeHaloArray
    @test is_active(kw_sum) == is_active(kw_ref)
    if is_active(kw_sum)
        @test collect(interior_view(parent(kw_sum))) == collect(interior_view(parent(kw_ref)))
    end
    kw_max = maximum(ha; dims=2)
    kw_max_ref = mapreduce_haloarray_dims(identity, max, ha, 2)
    @test is_active(kw_max) == is_active(kw_max_ref)
    if is_active(kw_max)
        @test collect(interior_view(parent(kw_max))) == collect(interior_view(parent(kw_max_ref)))
    end
    @test_throws ArgumentError mapfoldl(identity, +, ha; dims=1)
    foreach(free!, (kw_sum, kw_ref, kw_max, kw_max_ref))

    maybe_reduced = mapreduce_haloarray_dims(identity, +, ha, (1,))

    if topology.cart_coords[1] == 0
        @test is_active(maybe_reduced)
        reduced = HaloArrays.unwrap(maybe_reduced)
        reduced_global_size = (topology.dims[2] * local_dims[2],)
        reduced_owned_size = (local_dims[2],)
        @test size(reduced) == reduced_global_size
        @test global_size(reduced) == reduced_global_size
        @test interior_size(reduced) == reduced_owned_size
        @test halo_width(reduced) == halo

        expected = zeros(Int, local_dims[2])
        for x in 0:(topology.dims[1] - 1)
            source_rank = MPI.Cart_rank(topology.cart_comm, (x, topology.cart_coords[2]))
            for j in 1:local_dims[2], i in 1:local_dims[1]
                expected[j] += 1000 * source_rank + 10 * i + j
            end
        end
        @test collect(interior_view(reduced)) == expected
    else
        @test !is_active(maybe_reduced)
    end

    u = copy(ha)
    v = similar(ha)
    v_interior = interior_view(v)
    for i in 1:local_dims[1], j in 1:local_dims[2]
        v_interior[i, j] = 10_000 * rank + 100 * i + j
    end

    # dims are collection-global: field axis is 1, spatial axes 2… — so spatial
    # dim 1 is dims=(2,) here (fields u,v are 2-D).
    maybe_fields = HaloArrays.mapreduce_mhaloarray_dims(identity, +, MultiHaloArray((; u, v)), (2,))
    if topology.cart_coords[1] == 0
        @test is_active(maybe_fields)
        fields = HaloArrays.unwrap(maybe_fields)
        @test fields isa MultiHaloArray
        reduced_global_size = (topology.dims[2] * local_dims[2],)
        reduced_owned_size = (local_dims[2],)
        @test size(fields.arrays.u) == reduced_global_size
        @test size(fields.arrays.v) == reduced_global_size
        @test interior_size(fields.arrays.u) == reduced_owned_size
        @test interior_size(fields.arrays.v) == reduced_owned_size
    else
        @test !is_active(maybe_fields)
    end
end

@testset "collective == (all ranks agree)" begin
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    topology = CartesianTopology(comm, (0,); periodic=(true,))
    u = HaloArray(Float64, (4,), 1, topology; boundary_condition=_periodic_bc(Val(1)))
    v = HaloArray(Float64, (4,), 1, topology; boundary_condition=_periodic_bc(Val(1)))
    fill_from_global_indices!(I -> Float64(I[1]^2), u)
    fill_from_global_indices!(I -> Float64(I[1]^2), v)

    @test u == v                       # identical → true everywhere

    # ghosts must not affect equality (u synced, v stale)
    synchronize_halo!(u)
    @test u == v

    # perturb ONE interior cell on ONE rank: the generic iterate-based ==
    # would return true on every other rank; the collective == must return
    # false on ALL ranks, and agree.
    if rank == 0
        interior_view(v)[2] = -99.0
    end
    eq = (u == v)
    @test eq == false
    agreement = MPI.Allgather(eq, comm)
    @test all(==(false), agreement)
end
