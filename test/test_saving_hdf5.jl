using MPI
using HDF5
using Test
using HaloArrays

function _test_hdf5_path(name, comm)
    return joinpath(tempdir(), "haloarrays_$(name)_$(MPI.Comm_size(comm)).h5")
end

function _remove_on_root(path, comm)
    if MPI.Comm_rank(comm) == 0
        rm(path; force=true)
    end
    MPI.Barrier(comm)
    return nothing
end

function _owned_hdf5_slices(halo)
    owned_dims = HaloArrays.owned_size(halo)
    coords = halo.topology.cart_coords
    return ntuple(d -> (coords[d] * owned_dims[d] + 1):((coords[d] + 1) * owned_dims[d]), Val(ndims(halo)))
end

@testset "MPI HDF5 output" begin
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    owned_dims = (2, 3)
    halo_width_value = 1
    boundary = ntuple(_ -> (Periodic(), Periodic()), Val(2))

    @testset "append_haloarray!" begin
        topology = CartesianTopology(comm, (0, 0); periodic=(true, true))
        halo = HaloArray(Float64, owned_dims, halo_width_value, topology; boundary_condition=boundary)
        filename = _test_hdf5_path("append", comm)
        _remove_on_root(filename, comm)

        fid = h5open(filename, "w", comm, MPI.Info())
        dset = create_dataset_from_haloarray(fid, "field", halo)

        for step in 0:2
            fill_interior(halo, rank + step / 10)
            append_haloarray!(dset, halo)
        end

        close(fid)
        MPI.Barrier(comm)

        fid = h5open(filename, "r", comm, MPI.Info())
        dset = fid["field"]
        @test size(dset) == (3, global_size(halo)...)

        owned = _owned_hdf5_slices(halo)
        for step in 1:3
            slab = dset[step, owned...]
            @test all(slab .== rank + (step - 1) / 10)
        end
        close(fid)

        _remove_on_root(filename, comm)
    end

    @testset "write_haloarray_timestep!" begin
        topology = CartesianTopology(comm, (0, 0); periodic=(true, true))
        halo = HaloArray(Float64, owned_dims, halo_width_value, topology; boundary_condition=boundary)
        filename = _test_hdf5_path("fixed", comm)
        _remove_on_root(filename, comm)

        num_timesteps = 3
        fid = h5open(filename, "w", comm, MPI.Info())
        dset = create_fixedsize_dataset_from_haloarray(fid, "field", halo, num_timesteps)

        for step in 0:(num_timesteps - 1)
            fill_interior(halo, rank + 1 + step / 10)
            write_haloarray_timestep!(dset, halo, step)
        end

        close(fid)
        MPI.Barrier(comm)

        fid = h5open(filename, "r", comm, MPI.Info())
        dset = fid["field"]
        @test size(dset) == (num_timesteps, global_size(halo)...)

        owned = _owned_hdf5_slices(halo)
        for step in 1:num_timesteps
            slab = dset[step, owned...]
            @test all(slab .== rank + 1 + (step - 1) / 10)
        end
        close(fid)

        _remove_on_root(filename, comm)
    end

    @testset "ArrayOfHaloArray append" begin
        topology = CartesianTopology(comm, (0, 0); periodic=(true, true))
        u = HaloArray(Float64, owned_dims, halo_width_value, topology; boundary_condition=boundary)
        v = similar(u)
        filename = _test_hdf5_path("arrayof", comm)
        _remove_on_root(filename, comm)

        fill_interior(u, rank + 1)
        fill_interior(v, 100 + rank)
        fields = ArrayOfHaloArray([u, v])

        fid = h5open(filename, "w", comm, MPI.Info())
        dset = create_dataset_from_haloarray(fid, "state", fields)
        append_haloarray!(dset, fields)
        close(fid)
        MPI.Barrier(comm)

        fid = h5open(filename, "r", comm, MPI.Info())
        dset = fid["state"]
        @test size(dset) == (1, 2, global_size(u)...)

        owned = _owned_hdf5_slices(u)
        @test all(dset[1, 1, owned...] .== rank + 1)
        @test all(dset[1, 2, owned...] .== 100 + rank)
        close(fid)

        _remove_on_root(filename, comm)
    end

    @testset "ArrayOfHaloArray gather save" begin
        topology = CartesianTopology(comm, (0, 0); periodic=(true, true))
        u = HaloArray(Float64, owned_dims, halo_width_value, topology; boundary_condition=boundary)
        v = similar(u)
        filename_base = joinpath(tempdir(), "haloarrays_arrayof_gather_$(MPI.Comm_size(comm))")
        filename = filename_base * ".h5"
        _remove_on_root(filename, comm)

        fill_interior(u, rank + 10)
        fill_interior(v, rank + 110)
        fields = ArrayOfHaloArray([u, v])

        gather_and_save_haloarray(filename_base, fields)

        if rank == 0
            data = h5open(filename, "r") do fid
                read(fid["dataset"])
            end
            @test size(data) == (2, global_size(u)...)

            for r in 0:(MPI.Comm_size(comm) - 1)
                coords = Tuple(MPI.Cart_coords(topology.cart_comm, r))
                owned = ntuple(d -> (coords[d] * owned_dims[d] + 1):((coords[d] + 1) * owned_dims[d]), Val(2))
                @test all(data[1, owned...] .== r + 10)
                @test all(data[2, owned...] .== r + 110)
            end
        end

        _remove_on_root(filename, comm)
    end

    @testset "MultiHaloArray append" begin
        topology = CartesianTopology(comm, (0, 0); periodic=(true, true))
        rho = HaloArray(Float64, owned_dims, halo_width_value, topology; boundary_condition=boundary)
        mom = similar(rho)
        filename = _test_hdf5_path("multi", comm)
        _remove_on_root(filename, comm)

        fill_interior(rho, rank + 20)
        fill_interior(mom, rank + 120)
        fields = MultiHaloArray((; rho, mom))

        fid = h5open(filename, "w", comm, MPI.Info())
        group = create_dataset_from_haloarray(fid, "state", fields)
        append_haloarray!(group, fields)
        close(fid)
        MPI.Barrier(comm)

        fid = h5open(filename, "r", comm, MPI.Info())
        rho_dset = fid["state/rho"]
        mom_dset = fid["state/mom"]
        @test size(rho_dset) == (1, global_size(rho)...)
        @test size(mom_dset) == (1, global_size(mom)...)

        owned = _owned_hdf5_slices(rho)
        @test all(rho_dset[1, owned...] .== rank + 20)
        @test all(mom_dset[1, owned...] .== rank + 120)
        close(fid)

        _remove_on_root(filename, comm)
    end

    @testset "MultiHaloArray gather save" begin
        topology = CartesianTopology(comm, (0, 0); periodic=(true, true))
        rho = HaloArray(Float64, owned_dims, halo_width_value, topology; boundary_condition=boundary)
        mom = similar(rho)
        filename_base = joinpath(tempdir(), "haloarrays_multi_gather_$(MPI.Comm_size(comm))")
        filename = filename_base * ".h5"
        _remove_on_root(filename, comm)

        fill_interior(rho, rank + 30)
        fill_interior(mom, rank + 130)
        fields = MultiHaloArray((; rho, mom))

        gather_and_save_haloarray(filename_base, fields)

        if rank == 0
            rho_data = h5open(filename, "r") do fid
                read(fid["dataset/rho"])
            end
            mom_data = h5open(filename, "r") do fid
                read(fid["dataset/mom"])
            end

            @test size(rho_data) == global_size(rho)
            @test size(mom_data) == global_size(mom)

            for r in 0:(MPI.Comm_size(comm) - 1)
                coords = Tuple(MPI.Cart_coords(topology.cart_comm, r))
                owned = ntuple(d -> (coords[d] * owned_dims[d] + 1):((coords[d] + 1) * owned_dims[d]), Val(2))
                @test all(rho_data[owned...] .== r + 30)
                @test all(mom_data[owned...] .== r + 130)
            end
        end

        _remove_on_root(filename, comm)
    end

    @testset "MaybeHaloArray reduction append" begin
        topology = CartesianTopology(comm, (0, 0); periodic=(true, true))
        u = HaloArray(Int, owned_dims, halo_width_value, topology; boundary_condition=boundary)
        filename = _test_hdf5_path("maybe_reduce_append", comm)
        _remove_on_root(filename, comm)

        fill_interior(u, rank + 40)
        maybe_reduced = mapreduce_haloarray_dims(identity, +, u, (1,))
        append_haloarray_to_file!(filename[1:(end - 3)], "reduced", maybe_reduced)
        MPI.Barrier(comm)

        if rank == 0
            data = h5open(filename, "r") do fid
                read(fid["reduced"])
            end
            @test size(data) == (1, topology.dims[2] * owned_dims[2])

            for y in 0:(topology.dims[2] - 1)
                expected = sum(0:(topology.dims[1] - 1)) do x
                    source_rank = MPI.Cart_rank(topology.cart_comm, (x, y))
                    owned_dims[1] * (source_rank + 40)
                end
                y_range = (y * owned_dims[2] + 1):((y + 1) * owned_dims[2])
                @test all(data[1, y_range] .== expected)
            end
        end

        _remove_on_root(filename, comm)
    end

    @testset "MaybeHaloArray MultiHaloArray reduction save" begin
        topology = CartesianTopology(comm, (0, 0); periodic=(true, true))
        rho = HaloArray(Int, owned_dims, halo_width_value, topology; boundary_condition=boundary)
        mom = similar(rho)
        filename_base = joinpath(tempdir(), "haloarrays_maybe_multi_reduce_$(MPI.Comm_size(comm))")
        filename = filename_base * ".h5"
        _remove_on_root(filename, comm)

        fill_interior(rho, rank + 50)
        fill_interior(mom, rank + 150)
        maybe_fields = mapreduce_mhaloarray_dims(identity, +, MultiHaloArray((; rho, mom)), (1,))
        gather_and_save_haloarray(filename_base, maybe_fields)
        MPI.Barrier(comm)

        if rank == 0
            rho_data = h5open(filename, "r") do fid
                read(fid["dataset/rho"])
            end
            mom_data = h5open(filename, "r") do fid
                read(fid["dataset/mom"])
            end
            @test size(rho_data) == (topology.dims[2] * owned_dims[2],)
            @test size(mom_data) == (topology.dims[2] * owned_dims[2],)

            for y in 0:(topology.dims[2] - 1)
                expected_rho = sum(0:(topology.dims[1] - 1)) do x
                    source_rank = MPI.Cart_rank(topology.cart_comm, (x, y))
                    owned_dims[1] * (source_rank + 50)
                end
                expected_mom = sum(0:(topology.dims[1] - 1)) do x
                    source_rank = MPI.Cart_rank(topology.cart_comm, (x, y))
                    owned_dims[1] * (source_rank + 150)
                end
                y_range = (y * owned_dims[2] + 1):((y + 1) * owned_dims[2])
                @test all(rho_data[y_range] .== expected_rho)
                @test all(mom_data[y_range] .== expected_mom)
            end
        end

        _remove_on_root(filename, comm)
    end
end
