using Test
using MPI
using HaloArrays

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

    for i in eachindex(ha)
        ha[i] = rank + i / 10
    end

    local_sum = sum(interior_view(ha))
    @test mapreduce(identity, +, ha) ≈ MPI.Allreduce(local_sum, MPI.SUM, topology.cart_comm)
    @test reduce(+, ha) ≈ MPI.Allreduce(local_sum, MPI.SUM, topology.cart_comm)
    @test any(x -> x < 0, ha) == false
    @test all(x -> x >= 0, ha) == true

    u = copy(ha)
    v = similar(ha)
    for i in eachindex(v)
        v[i] = 10 * rank + i
    end
    fields = MultiHaloArray((; u, v); check=true)

    local_field_sum = sum(interior_view(u)) + sum(interior_view(v))
    @test mapreduce(identity, +, fields) ≈ MPI.Allreduce(local_field_sum, MPI.SUM, topology.cart_comm)
    @test all(x -> x >= 0, fields)
    @test any(x -> x == 1, fields)
end

@testset "MPI dimension reductions" begin
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    topology = CartesianTopology(comm, (0, 0); periodic=(true, true))
    halo = 1
    local_size = (2, 3)

    ha = HaloArray(Int, local_size, halo, topology; boundary_condition=_periodic_bc(Val(2)))
    fill!(parent(ha), -10_000)
    for i in 1:local_size[1], j in 1:local_size[2]
        ha[i, j] = 1000 * rank + 10 * i + j
    end

    maybe_reduced = mapreduce_haloarray_dims(identity, +, ha, (1,))

    if topology.cart_coords[1] == 0
        @test isactive(maybe_reduced)
        reduced = unwrap(maybe_reduced)
        @test size(reduced) == (local_size[2],)
        @test halo_width(reduced) == halo

        expected = zeros(Int, local_size[2])
        for x in 0:(topology.dims[1] - 1)
            source_rank = MPI.Cart_rank(topology.cart_comm, (x, topology.cart_coords[2]))
            for j in 1:local_size[2], i in 1:local_size[1]
                expected[j] += 1000 * source_rank + 10 * i + j
            end
        end
        @test collect(interior_view(reduced)) == expected
    else
        @test !isactive(maybe_reduced)
    end

    u = copy(ha)
    v = similar(ha)
    for i in 1:local_size[1], j in 1:local_size[2]
        v[i, j] = 10_000 * rank + 100 * i + j
    end

    maybe_fields = mapreduce_mhaloarray_dims(identity, +, MultiHaloArray((; u, v); check=true), (1,))
    if topology.cart_coords[1] == 0
        @test isactive(maybe_fields)
        fields = unwrap(maybe_fields)
        @test fields isa MultiHaloArray
        @test size(fields.arrays.u) == (local_size[2],)
        @test size(fields.arrays.v) == (local_size[2],)
    else
        @test !isactive(maybe_fields)
    end
end
