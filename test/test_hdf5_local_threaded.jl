using HDF5
using Test
using HaloArrays

function _serial_hdf5_base(name)
    return joinpath(tempdir(), "haloarrays_$(name)_$(getpid())")
end

function _read_dataset(path, dataset)
    h5open(path, "r") do fid
        return read(fid[dataset])
    end
end

@testset "Local and threaded HDF5 output" begin
    @testset "LocalHaloArray append and fixed-size writes" begin
        base = _serial_hdf5_base("local")
        path = base * ".h5"
        rm(path; force=true)

        halo = LocalHaloArray(Int, (2, 3), 1; boundary_condition=:repeating)
        interior_view(halo) .= reshape(collect(1:6), 2, 3)

        append_haloarray_to_file!(base, "field", halo)
        data = _read_dataset(path, "field")
        @test size(data) == (1, 2, 3)
        @test data[1, :, :] == interior_view(halo)

        save_base = _serial_hdf5_base("local_save")
        save_path = save_base * ".h5"
        rm(save_path; force=true)
        gather_and_save_haloarray(save_base, halo)
        saved = _read_dataset(save_path, "dataset")
        @test size(saved) == (2, 3)
        @test saved == interior_view(halo)
        rm(save_path; force=true)

        interior_view(halo) .+= 10
        fid, dset = create_haloarray_output_file(path, "fixed", halo, 2)
        write_haloarray_timestep!(dset, halo, 0)
        interior_view(halo) .+= 10
        write_haloarray_timestep!(dset, halo, 1)
        close(fid)

        fixed = _read_dataset(path, "fixed")
        @test size(fixed) == (2, 2, 3)
        @test fixed[1, :, :] == reshape(collect(11:16), 2, 3)
        @test fixed[2, :, :] == reshape(collect(21:26), 2, 3)

        # Reopening the file must validate the existing "fixed" dataset against the
        # halo array it will receive — a shape or eltype mismatch is refused loudly
        # rather than silently corrupting the dataset on the next write.
        wrong_shape = LocalHaloArray(Int, (2, 4), 1; boundary_condition=:repeating)
        @test_throws DimensionMismatch create_haloarray_output_file(path, "fixed", wrong_shape, 2)
        wrong_steps = LocalHaloArray(Int, (2, 3), 1; boundary_condition=:repeating)
        @test_throws DimensionMismatch create_haloarray_output_file(path, "fixed", wrong_steps, 5)
        wrong_eltype = LocalHaloArray(Float64, (2, 3), 1; boundary_condition=:repeating)
        @test_throws ArgumentError create_haloarray_output_file(path, "fixed", wrong_eltype, 2)
        # a matching reopen still succeeds and reuses the dataset
        fid2, dset2 = create_haloarray_output_file(path, "fixed", halo, 2)
        @test size(dset2) == (2, 2, 3)
        close(fid2)

        rm(path; force=true)
    end

    @testset "ThreadedHaloArray append" begin
        base = _serial_hdf5_base("threaded")
        path = base * ".h5"
        rm(path; force=true)

        halo = ThreadedHaloArray(Int, (2,), 1; dims=(3,), boundary_condition=:repeating)
        for tile_id in 1:tile_count(halo)
            interior_view(halo, tile_id) .= (2 * tile_id - 1):(2 * tile_id)
        end

        append_haloarray_to_file!(base, "field", halo)
        data = _read_dataset(path, "field")
        @test size(data) == (1, 6)
        @test vec(data[1, :]) == collect(1:6)

        save_base = _serial_hdf5_base("threaded_save")
        save_path = save_base * ".h5"
        rm(save_path; force=true)
        gather_and_save_haloarray(save_base, halo)
        saved = _read_dataset(save_path, "dataset")
        @test size(saved) == (6,)
        @test vec(saved) == collect(1:6)
        rm(save_path; force=true)

        rm(path; force=true)
    end

    @testset "ArrayOfHaloArray append" begin
        base = _serial_hdf5_base("arrayof")
        path = base * ".h5"
        rm(path; force=true)

        u = LocalHaloArray(Int, (2, 3), 1; boundary_condition=:repeating)
        v = similar(u)
        interior_view(u) .= reshape(collect(1:6), 2, 3)
        interior_view(v) .= reshape(collect(101:106), 2, 3)
        fields = ArrayOfHaloArray([u, v])

        append_haloarray_to_file!(base, "state", fields)
        data = _read_dataset(path, "state")
        @test size(data) == (1, 2, 2, 3)
        @test data[1, 1, :, :] == interior_view(u)
        @test data[1, 2, :, :] == interior_view(v)

        save_base = _serial_hdf5_base("arrayof_save")
        save_path = save_base * ".h5"
        rm(save_path; force=true)
        gather_and_save_haloarray(save_base, fields)
        saved = _read_dataset(save_path, "dataset")
        @test size(saved) == (2, 2, 3)
        @test saved[1, :, :] == interior_view(u)
        @test saved[2, :, :] == interior_view(v)
        rm(save_path; force=true)

        rm(path; force=true)
    end

    @testset "ArrayOfHaloArray with threaded fields" begin
        base = _serial_hdf5_base("arrayof_threaded")
        path = base * ".h5"
        rm(path; force=true)

        u = ThreadedHaloArray(Int, (2,), 1; dims=(2,), boundary_condition=:repeating)
        v = similar(u)
        interior_view(u, 1) .= [1, 2]
        interior_view(u, 2) .= [3, 4]
        interior_view(v, 1) .= [10, 20]
        interior_view(v, 2) .= [30, 40]
        fields = ArrayOfHaloArray([u, v])

        append_haloarray_to_file!(base, "state", fields)
        data = _read_dataset(path, "state")
        @test size(data) == (1, 2, 4)
        @test vec(data[1, 1, :]) == [1, 2, 3, 4]
        @test vec(data[1, 2, :]) == [10, 20, 30, 40]

        rm(path; force=true)
    end

    @testset "MultiHaloArray append and save" begin
        base = _serial_hdf5_base("multi")
        path = base * ".h5"
        rm(path; force=true)

        rho = LocalHaloArray(Int, (2, 3), 1; boundary_condition=:repeating)
        mom = similar(rho)
        interior_view(rho) .= reshape(collect(1:6), 2, 3)
        interior_view(mom) .= reshape(collect(101:106), 2, 3)
        fields = LocalMultiHaloArray((; rho, mom))

        append_haloarray_to_file!(base, "state", fields)
        rho_data = _read_dataset(path, "state/rho")
        mom_data = _read_dataset(path, "state/mom")
        @test size(rho_data) == (1, 2, 3)
        @test size(mom_data) == (1, 2, 3)
        @test rho_data[1, :, :] == interior_view(rho)
        @test mom_data[1, :, :] == interior_view(mom)

        save_base = _serial_hdf5_base("multi_save")
        save_path = save_base * ".h5"
        rm(save_path; force=true)
        gather_and_save_haloarray(save_base, fields)
        saved_rho = _read_dataset(save_path, "dataset/rho")
        saved_mom = _read_dataset(save_path, "dataset/mom")
        @test saved_rho == interior_view(rho)
        @test saved_mom == interior_view(mom)

        rm(path; force=true)
        rm(save_path; force=true)
    end

    @testset "MultiHaloArray with threaded and nested fields" begin
        base = _serial_hdf5_base("multi_threaded")
        path = base * ".h5"
        rm(path; force=true)

        rho = ThreadedHaloArray(Int, (2,), 1; dims=(2,), boundary_condition=:repeating)
        mom = similar(rho)
        interior_view(rho, 1) .= [1, 2]
        interior_view(rho, 2) .= [3, 4]
        interior_view(mom, 1) .= [10, 20]
        interior_view(mom, 2) .= [30, 40]
        fields = ThreadedMultiHaloArray((; rho, mom))

        append_haloarray_to_file!(base, "state", fields)
        rho_data = _read_dataset(path, "state/rho")
        mom_data = _read_dataset(path, "state/mom")
        @test size(rho_data) == (1, 4)
        @test vec(rho_data[1, :]) == [1, 2, 3, 4]
        @test vec(mom_data[1, :]) == [10, 20, 30, 40]
        rm(path; force=true)

        nested_base = _serial_hdf5_base("multi_nested")
        nested_path = nested_base * ".h5"
        rm(nested_path; force=true)

        scalar = LocalHaloArray(Int, (2,), 1; boundary_condition=:repeating)
        q1 = similar(scalar)
        q2 = similar(scalar)
        interior_view(scalar) .= [7, 8]
        interior_view(q1) .= [1, 2]
        interior_view(q2) .= [3, 4]
        nested = MultiHaloArray((; scalar, q=ArrayOfHaloArray([q1, q2])))

        append_haloarray_to_file!(nested_base, "state", nested)
        scalar_data = _read_dataset(nested_path, "state/scalar")
        q_data = _read_dataset(nested_path, "state/q")
        @test size(scalar_data) == (1, 2)
        @test vec(scalar_data[1, :]) == [7, 8]
        @test size(q_data) == (1, 2, 2)
        @test vec(q_data[1, 1, :]) == [1, 2]
        @test vec(q_data[1, 2, :]) == [3, 4]

        rm(nested_path; force=true)
    end
end
