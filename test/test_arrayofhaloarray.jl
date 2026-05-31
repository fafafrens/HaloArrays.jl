using Test
using MPI
using HaloArrays

if !MPI.Initialized()
    MPI.Init()
end

function _arrayofhaloarray_topology(dims::NTuple{N,Int}) where {N}
    return CartesianTopology(MPI.COMM_SELF, dims; periodic=ntuple(_ -> false, Val(N)))
end

function _fill_arrayofhaloarray_fields!(arrays)
    for I in CartesianIndices(arrays)
        view = interior_view(arrays[I])
        for i in eachindex(view)
            view[i] = 100 * I[1] + 10 * I[2] + i
        end
    end
    return arrays
end

@testset "ArrayOfHaloArray" begin
    topology = _arrayofhaloarray_topology((1,))
    arrays = [HaloArray(Float64, (3,), 1, topology; boundary_condition=:repeating)
              for _ in 1:2, _ in 1:2]
    _fill_arrayofhaloarray_fields!(arrays)

    fields = ArrayOfHaloArray(arrays)

    @test fields isa ArrayOfHaloArray
    @test fields isa AbstractArray{Float64,3}
    @test eltype(fields) === Float64
    @test field_shape(fields) == (2, 2)
    @test HaloArrays.n_field(fields) == 4
    @test ndims(fields) == 3
    @test size(fields) == (2, 2, 3)
    @test size(fields) == global_size(fields)
    @test axes(fields) == map(Base.OneTo, global_size(fields))
    @test owned_axes(fields) == map(Base.OneTo, owned_size(fields))
    @test owned_size(fields) == (2, 2, 3)
    @test interior_size(fields) == (2, 2, 3)
    @test global_size(fields) == (2, 2, 3)
    @test storage_size(fields) == (2, 2, 5)
    @test halo_width(fields) == 1
    @test parent(fields) === arrays
    @test fields[1, 2] === arrays[1, 2]
    @test eltype(typeof(fields)) === Float64
    @test_logs (:warn, r"Global scalar getindex") begin
        @test fields[1, 2, 3] == 123
    end
    @test isactive(fields)

    interior_view(arrays[1, 2])[3] = 123
    @test_logs (:warn, r"Global scalar setindex!") begin
        fields[1, 2, 3] = -7
    end
    @test fields[1, 2, 3] == -7
    @test interior_view(arrays[1, 2])[3] == -7
    fields[1, 2, 3] = 123

    views = interior_view(fields)
    @test size(views) == (2, 2)
    @test views[2, 1] == interior_view(arrays[2, 1])

    synchronize_halo!(fields)
    for I in CartesianIndices(arrays)
        field = arrays[I]
        @test parent(field)[1] == first(interior_view(field))
        @test parent(field)[end] == last(interior_view(field))
    end

    shifted = fields .+ 5
    @test shifted isa ArrayOfHaloArray
    @test field_shape(shifted) == field_shape(fields)
    for I in CartesianIndices(arrays)
        @test collect(interior_view(shifted[I])) == collect(interior_view(fields[I]) .+ 5)
    end

    dest = similar(fields)
    dest .= 2 .* fields .+ shifted
    for I in CartesianIndices(arrays)
        expected = 2 .* interior_view(fields[I]) .+ interior_view(shifted[I])
        @test collect(interior_view(dest[I])) == collect(expected)
    end

    copied = copy(fields)
    interior_view(copied[1, 1])[1] = -100
    @test interior_view(fields[1, 1])[1] != interior_view(copied[1, 1])[1]

    copied_into = similar(fields)
    fill!(copied_into, -1)
    @test copyto!(copied_into, fields) === copied_into
    for I in CartesianIndices(arrays)
        @test collect(interior_view(copied_into[I])) == collect(interior_view(fields[I]))
    end

    zero_fields = zero(fields)
    @test zero_fields isa ArrayOfHaloArray
    @test all(==(0), zero_fields)
    @test fill!(zero_fields, 9) === zero_fields
    @test all(==(9), zero_fields)

    similar_fields = similar(fields, Int)
    @test similar_fields isa ArrayOfHaloArray
    @test eltype(similar_fields) === Int
    @test field_shape(similar_fields) == field_shape(fields)
    @test size(similar_fields) == size(fields)

    resized_fields = similar(fields, Float32, (2, 2, 4))
    @test resized_fields isa ArrayOfHaloArray
    @test eltype(resized_fields) === Float32
    @test field_shape(resized_fields) == (2, 2)
    @test size(resized_fields) == (2, 2, 4)
    @test owned_size(resized_fields) == (2, 2, 4)
    @test storage_size(resized_fields) == (2, 2, 6)

    reshaped_fields = similar(fields, Float32, (3, 2, 4))
    @test reshaped_fields isa ArrayOfHaloArray
    @test field_shape(reshaped_fields) == (3, 2)
    @test size(reshaped_fields) == (3, 2, 4)

    bad_arrays = copy(arrays)
    bad_arrays[2, 2] = HaloArray(Float64, (4,), 1, topology; boundary_condition=:repeating)
    @test_throws DimensionMismatch ArrayOfHaloArray(bad_arrays)

    bcs = fill(:repeating, 2, 2)
    from_bcs = ArrayOfHaloArray(Int, (3,), 1, topology; boundary_conditions=bcs)
    @test from_bcs isa ArrayOfHaloArray
    @test field_shape(from_bcs) == (2, 2)
    @test eltype(from_bcs) === Int

    local_bcs = [:repeating, :antireflecting]
    local_fields = ArrayOfHaloArray(LocalHaloArray, Int, (3,), 1; boundary_conditions=local_bcs)
    interior_view(local_fields[1]) .= [1, 2, 3]
    interior_view(local_fields[2]) .= [10, 20, 30]

    @test local_fields isa ArrayOfHaloArray
    @test local_fields isa AbstractArray{Int,2}
    @test local_fields[1] isa LocalHaloArray
    @test field_shape(local_fields) == (2,)
    @test size(local_fields) == (2, 3)
    @test size(local_fields) == global_size(local_fields)
    @test owned_axes(local_fields) == map(Base.OneTo, owned_size(local_fields))
    @test owned_size(local_fields) == (2, 3)
    @test local_fields[2, 3] == 30
    local_fields[2, 3] = 33
    @test local_fields[2, 3] == 33
    @test interior_view(local_fields[2])[3] == 33
    local_fields[2, 3] = 30

    shifted_local = local_fields .+ 4
    @test shifted_local isa ArrayOfHaloArray
    @test shifted_local[1] isa LocalHaloArray
    @test collect(interior_view(shifted_local[1])) == [5, 6, 7]
    @test collect(interior_view(shifted_local[2])) == [14, 24, 34]

    local_dest = similar(local_fields)
    local_dest .= 2 .* local_fields .+ shifted_local
    @test collect(interior_view(local_dest[1])) == [7, 10, 13]
    @test collect(interior_view(local_dest[2])) == [34, 64, 94]

    resized_local = similar(local_fields, Float32, (2, 5))
    @test resized_local isa ArrayOfHaloArray
    @test eltype(resized_local) === Float32
    @test field_shape(resized_local) == (2,)
    @test size(resized_local) == (2, 5)
    @test storage_size(resized_local) == (2, 7)

    reshaped_local = similar(local_fields, Float32, (3, 5))
    @test field_shape(reshaped_local) == (3,)
    @test size(reshaped_local) == (3, 5)

    synchronize_halo!(local_fields)
    @test parent(local_fields[1]) == [1, 1, 2, 3, 3]
    @test parent(local_fields[2]) == [-10, 10, 20, 30, -30]

    shaped_local = ArrayOfHaloArray(LocalHaloArray, Float32, (2, 3), (4,), 1;
        boundary_condition=:periodic)
    @test shaped_local isa ArrayOfHaloArray
    @test shaped_local isa AbstractArray{Float32,3}
    @test field_shape(shaped_local) == (2, 3)
    @test size(shaped_local) == (2, 3, 4)
    @test storage_size(shaped_local) == (2, 3, 6)
    @test all(field -> field isa LocalHaloArray, parent(shaped_local))

    custom_storage(T, dims...) = fill(T(7), dims...)
    custom_local = ArrayOfHaloArray(LocalHaloArray, Float32, (2,), (3,), 1;
        boundary_condition=:periodic, storage=custom_storage)
    @test parent(custom_local[1]) == fill(7.0f0, 5)
    @test parent(custom_local[2]) == fill(7.0f0, 5)

    shaped_bcs = fill(:repeating, 2, 2)
    shaped_bcs[2, 2] = :antireflecting
    shaped_from_bcs = ArrayOfHaloArray(LocalHaloArray, Float32, (2, 2), (3,), 1;
        boundary_conditions=shaped_bcs)
    @test field_shape(shaped_from_bcs) == (2, 2)
    @test_throws DimensionMismatch ArrayOfHaloArray(LocalHaloArray, Float32, (2, 2), (3,), 1;
        boundary_conditions=fill(:repeating, 3))

    threaded_bcs = [:repeating, :repeating]
    threaded_fields = ArrayOfHaloArray(ThreadedHaloArray, Int, (3,), 1;
        dims=(2,), boundary_conditions=threaded_bcs)
    interior_view(threaded_fields[1], 1) .= [1, 2, 3]
    interior_view(threaded_fields[1], 2) .= [4, 5, 6]
    interior_view(threaded_fields[2], 1) .= [10, 20, 30]
    interior_view(threaded_fields[2], 2) .= [40, 50, 60]

    @test threaded_fields isa ArrayOfHaloArray
    @test threaded_fields isa AbstractArray{Int,2}
    @test threaded_fields[1] isa ThreadedHaloArray
    @test field_shape(threaded_fields) == (2,)
    @test size(threaded_fields) == (2, 6)

    shifted_threaded = threaded_fields .+ 3
    @test shifted_threaded isa ArrayOfHaloArray
    @test shifted_threaded[1] isa ThreadedHaloArray
    @test collect(interior_view(shifted_threaded[1], 1)) == [4, 5, 6]
    @test collect(interior_view(shifted_threaded[1], 2)) == [7, 8, 9]
    @test collect(interior_view(shifted_threaded[2], 1)) == [13, 23, 33]
    @test collect(interior_view(shifted_threaded[2], 2)) == [43, 53, 63]

    threaded_dest = similar(threaded_fields)
    threaded_dest .= threaded_fields .+ shifted_threaded
    @test collect(interior_view(threaded_dest[1], 1)) == [5, 7, 9]
    @test collect(interior_view(threaded_dest[1], 2)) == [11, 13, 15]
    @test collect(interior_view(threaded_dest[2], 1)) == [23, 43, 63]
    @test collect(interior_view(threaded_dest[2], 2)) == [83, 103, 123]

    resized_threaded = similar(threaded_fields, Float32, (2, 8))
    @test resized_threaded isa ArrayOfHaloArray
    @test eltype(resized_threaded) === Float32
    @test field_shape(resized_threaded) == (2,)
    @test size(resized_threaded) == (2, 8)
    @test tile_size(resized_threaded[1]) == (4,)

    synchronize_halo!(threaded_fields)
    @test tile_parent(threaded_fields[1], 1) == [1, 1, 2, 3, 4]
    @test tile_parent(threaded_fields[1], 2) == [3, 4, 5, 6, 6]
    @test tile_parent(threaded_fields[2], 1) == [10, 10, 20, 30, 40]
    @test tile_parent(threaded_fields[2], 2) == [30, 40, 50, 60, 60]
end
