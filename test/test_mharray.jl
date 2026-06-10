using Test
using MPI
using HaloArrays

if !MPI.Initialized()
    MPI.Init()
end

function _test_topology(dims::NTuple{N,Int}) where {N}
    return CartesianTopology(MPI.COMM_SELF, dims; periodic=ntuple(_ -> false, Val(N)))
end

@testset "MultiHaloArray" begin
    topology = _test_topology((1, 1))
    u = HaloArray(Float64, (3, 2), 1, topology; boundary_condition=:repeating)
    v = HaloArray(Int, (3, 2), 1, topology; boundary_condition=:repeating)

    u_interior = interior_view(u)
    v_interior = interior_view(v)
    for i in 1:size(u, 1), j in 1:size(u, 2)
        u_interior[i, j] = i + j / 10
        v_interior[i, j] = 10 * i + j
    end

    fields = MultiHaloArray((; u, v))

    @test fields isa MultiHaloArray
    @test fields isa AbstractArray{Float64,3}
    @test eltype(fields) === Float64
    @test ndims(fields) == 3
    @test HaloArrays.n_field(fields) == 2
    @test fields[:u] === u
    @test fields[:v] === v
    @test eltype(typeof(fields)) === Float64
    @test isactive(fields)

    views = interior_view(fields)
    @test keys(views) == (:u, :v)
    @test collect(views.u) == [i + j / 10 for i in 1:3, j in 1:2]
    @test collect(views.v) == [10 * i + j for i in 1:3, j in 1:2]

    shifted = fields .+ 2
    @test shifted isa MultiHaloArray
    @test collect(interior_view(shifted.arrays.u)) == [i + j / 10 + 2 for i in 1:3, j in 1:2]
    @test collect(interior_view(shifted.arrays.v)) == [10 * i + j + 2 for i in 1:3, j in 1:2]

    dest = similar(fields)
    dest .= 2 .* fields
    @test collect(interior_view(dest.arrays.u)) == [2 * (i + j / 10) for i in 1:3, j in 1:2]
    @test collect(interior_view(dest.arrays.v)) == [2 * (10 * i + j) for i in 1:3, j in 1:2]

    @test length(fields) == prod(size(fields))
    @test first(eachindex(fields)) == CartesianIndex(1, 1, 1)

    copied_into = similar(fields)
    fill!(copied_into, -1)
    @test copyto!(copied_into, fields) === copied_into
    @test collect(interior_view(copied_into.arrays.u)) == collect(interior_view(fields.arrays.u))
    @test collect(interior_view(copied_into.arrays.v)) == collect(interior_view(fields.arrays.v))

    zero_fields = zero(fields)
    @test zero_fields isa MultiHaloArray
    @test all(==(0), zero_fields)
    @test fill!(zero_fields, 7) === zero_fields
    @test all(==(7), zero_fields)

    from_bcs = MultiHaloArray(HaloArray, Float64, (3, 2), 1, topology;
        boundary_conditions=(; rho=:repeating, mom=:repeating))
    @test from_bcs isa MultiHaloArray
    @test from_bcs[:rho] isa HaloArray
    @test size(from_bcs) == (2, 3, 2)
    @test size(from_bcs) == global_size(from_bcs)
    @test axes(from_bcs) == map(Base.OneTo, global_size(from_bcs))
    @test owned_axes(from_bcs) == map(Base.OneTo, owned_size(from_bcs))
    @test owned_size(from_bcs) == (2, 3, 2)
    @test global_size(from_bcs) == (2, 3, 2)
    @test eltype(from_bcs) === Float64

    local_fields = MultiHaloArray(LocalHaloArray, Int, (3,), 1;
        boundary_conditions=(; rho=:repeating, mom=:antireflecting))
    interior_view(local_fields.arrays.rho) .= [1, 2, 3]
    interior_view(local_fields.arrays.mom) .= [10, 20, 30]

    @test local_fields isa MultiHaloArray
    @test local_fields isa AbstractArray{Int,2}
    @test local_fields[:rho] isa LocalHaloArray
    @test size(local_fields) == (2, 3)
    @test size(local_fields) == global_size(local_fields)
    @test owned_axes(local_fields) == map(Base.OneTo, owned_size(local_fields))
    @test owned_size(local_fields) == (2, 3)
    @test local_fields[1] === local_fields.arrays.rho
    @test local_fields[1, 2] == 2
    local_fields[2, 3] = 35
    @test local_fields[2, 3] == 35
    @test interior_view(local_fields.arrays.mom)[3] == 35
    local_fields[2, 3] = 30

    shifted_local = local_fields .+ 4
    @test shifted_local isa MultiHaloArray
    @test shifted_local.arrays.rho isa LocalHaloArray
    @test collect(interior_view(shifted_local.arrays.rho)) == [5, 6, 7]
    @test collect(interior_view(shifted_local.arrays.mom)) == [14, 24, 34]

    local_dest = similar(local_fields)
    local_dest .= 2 .* local_fields .+ shifted_local
    @test collect(interior_view(local_dest.arrays.rho)) == [7, 10, 13]
    @test collect(interior_view(local_dest.arrays.mom)) == [34, 64, 94]

    resized_local = similar(local_fields, Float32, (2, 5))
    @test resized_local isa MultiHaloArray
    @test eltype(resized_local) === Float32
    @test size(resized_local) == (2, 5)
    @test owned_size(resized_local) == (2, 5)
    @test_throws DimensionMismatch similar(local_fields, Float32, (3, 5))

    synchronize_halo!(local_fields)
    @test parent(local_fields.arrays.rho) == [1, 1, 2, 3, 3]
    @test parent(local_fields.arrays.mom) == [-10, 10, 20, 30, -30]

    threaded_fields = MultiHaloArray(ThreadedHaloArray, Int, (3,), 1;
        dims=(2,),
        boundary_conditions=(; rho=:repeating, mom=:repeating))
    interior_view(threaded_fields.arrays.rho, 1) .= [1, 2, 3]
    interior_view(threaded_fields.arrays.rho, 2) .= [4, 5, 6]
    interior_view(threaded_fields.arrays.mom, 1) .= [10, 20, 30]
    interior_view(threaded_fields.arrays.mom, 2) .= [40, 50, 60]

    @test threaded_fields isa MultiHaloArray
    @test threaded_fields isa AbstractArray{Int,2}
    @test threaded_fields[:rho] isa ThreadedHaloArray
    @test size(threaded_fields) == (2, 6)
    @test size(threaded_fields) == global_size(threaded_fields)
    @test owned_axes(threaded_fields) == map(Base.OneTo, owned_size(threaded_fields))
    @test owned_size(threaded_fields) == (2, 6)

    shifted_threaded = threaded_fields .+ 3
    @test shifted_threaded isa MultiHaloArray
    @test shifted_threaded.arrays.rho isa ThreadedHaloArray
    @test collect(interior_view(shifted_threaded.arrays.rho, 1)) == [4, 5, 6]
    @test collect(interior_view(shifted_threaded.arrays.rho, 2)) == [7, 8, 9]
    @test collect(interior_view(shifted_threaded.arrays.mom, 1)) == [13, 23, 33]
    @test collect(interior_view(shifted_threaded.arrays.mom, 2)) == [43, 53, 63]

    threaded_dest = similar(threaded_fields)
    threaded_dest .= threaded_fields .+ shifted_threaded
    @test collect(interior_view(threaded_dest.arrays.rho, 1)) == [5, 7, 9]
    @test collect(interior_view(threaded_dest.arrays.rho, 2)) == [11, 13, 15]
    @test collect(interior_view(threaded_dest.arrays.mom, 1)) == [23, 43, 63]
    @test collect(interior_view(threaded_dest.arrays.mom, 2)) == [83, 103, 123]

    threaded_copy = similar(threaded_fields)
    @test copyto!(threaded_copy, threaded_fields) === threaded_copy
    for name in keys(threaded_fields.arrays), tile_id in 1:tile_count(threaded_fields)
        @test tile_parent(threaded_copy.arrays[name], tile_id) ==
              tile_parent(threaded_fields.arrays[name], tile_id)
    end

    threaded_zero = zero(threaded_fields)
    @test threaded_zero isa MultiHaloArray
    @test fill!(threaded_zero, -3) === threaded_zero
    for field in values(threaded_zero.arrays), tile_id in 1:tile_count(threaded_zero)
        @test all(==(-3), tile_parent(field, tile_id))
    end

    resized_threaded = similar(threaded_fields, Float32, (2, 8))
    @test resized_threaded isa MultiHaloArray
    @test eltype(resized_threaded) === Float32
    @test size(resized_threaded) == (2, 8)
    @test tile_size(resized_threaded) == (4,)
    @test_throws DimensionMismatch similar(threaded_fields, Float32, (3, 8))

    synchronize_halo!(threaded_fields)
    @test tile_parent(threaded_fields.arrays.rho, 1) == [1, 1, 2, 3, 4]
    @test tile_parent(threaded_fields.arrays.rho, 2) == [3, 4, 5, 6, 6]
    @test tile_parent(threaded_fields.arrays.mom, 1) == [10, 10, 20, 30, 40]
    @test tile_parent(threaded_fields.arrays.mom, 2) == [30, 40, 50, 60, 60]

    q_arrays = [HaloArray(Float64, (3, 2), 1, topology; boundary_condition=:repeating) for _ in 1:2]
    for c in eachindex(q_arrays)
        q_interior = interior_view(q_arrays[c])
        for i in 1:size(q_interior, 1), j in 1:size(q_interior, 2)
            q_interior[i, j] = 100 * c + 10 * i + j
        end
    end

    q = ArrayOfHaloArray(q_arrays)
    nested_fields = MultiHaloArray((; rho=u, q))

    @test nested_fields isa MultiHaloArray
    @test ndims(nested_fields) == 3
    @test size(nested_fields) == (2, 3, 2)
    @test size(nested_fields) == global_size(nested_fields)
    @test owned_size(nested_fields) == (2, 3, 2)
    @test interior_size(nested_fields) == (2, 3, 2)
    @test global_size(nested_fields) == (2, 3, 2)
    @test storage_size(nested_fields) == (2, 5, 4)
    @test halo_width(nested_fields) == 1
    @test nested_fields[:q] === q

    nested_shifted = nested_fields .+ 2
    @test nested_shifted isa MultiHaloArray
    @test nested_shifted.arrays.q isa ArrayOfHaloArray
    @test collect(interior_view(nested_shifted.arrays.rho)) == [i + j / 10 + 2 for i in 1:3, j in 1:2]
    @test collect(interior_view(nested_shifted.arrays.q[1])) == [100 + 10 * i + j + 2 for i in 1:3, j in 1:2]
    @test collect(interior_view(nested_shifted.arrays.q[2])) == [200 + 10 * i + j + 2 for i in 1:3, j in 1:2]

    nested_dest = similar(nested_fields)
    nested_dest .= 2 .* nested_fields .+ nested_shifted
    @test collect(interior_view(nested_dest.arrays.rho)) == [3 * (i + j / 10) + 2 for i in 1:3, j in 1:2]
    @test collect(interior_view(nested_dest.arrays.q[1])) == [3 * (100 + 10 * i + j) + 2 for i in 1:3, j in 1:2]
    @test collect(interior_view(nested_dest.arrays.q[2])) == [3 * (200 + 10 * i + j) + 2 for i in 1:3, j in 1:2]

    synchronize_halo!(nested_fields)
    @test parent(nested_fields.arrays.q[1])[1, 2] == first(interior_view(nested_fields.arrays.q[1]))
    @test parent(nested_fields.arrays.q[2])[end, 3] == last(interior_view(nested_fields.arrays.q[2]))
    @test all(x -> x > 0, nested_fields)
    @test any(x -> x == 111, nested_fields)

    copied = copy(fields)
    interior_view(copied.arrays.u)[1, 1] = -1
    @test interior_view(fields.arrays.u)[1, 1] != interior_view(copied.arrays.u)[1, 1]

    # fields + boundary_condition shorthand
    from_fields = MultiHaloArray(HaloArray, Float64, (3, 2), 1, topology;
        fields=(:a, :b, :c), boundary_condition=:repeating)
    @test from_fields isa MultiHaloArray
    @test keys(from_fields.arrays) == (:a, :b, :c)
    @test all(f -> f isa HaloArray, values(from_fields.arrays))
    @test size(from_fields) == (3, 3, 2)

    from_fields_default_type = MultiHaloArray(HaloArray, (3, 2), 1, topology;
        fields=(:x, :y), boundary_condition=:repeating)
    @test eltype(from_fields_default_type) === Float64

    bad = HaloArray(Float64, (4, 2), 1, topology; boundary_condition=:repeating)
    @test_throws DimensionMismatch MultiHaloArray((; u, bad))

    bad_q = ArrayOfHaloArray([HaloArray(Float64, (4, 2), 1, topology; boundary_condition=:repeating)])
    @test_throws DimensionMismatch MultiHaloArray((; u, q=bad_q))

    @test all(x -> x > 0, fields)
    @test any(x -> x == 22, fields)
end
