using HDF5
using Test
using HaloArrays

_h5(name) = joinpath(tempdir(), "haloarrays_$(name)_$(getpid()).h5")
_read(path, dset) = h5open(path, "r") do fid; read(fid[dset]); end

@testset "Local and threaded HDF5 output" begin
    @testset "LocalHaloArray: filename form appends and validates" begin
        path = _h5("local"); rm(path; force=true)
        halo = LocalHaloArray(Int, (2, 3), 1; boundary_condition=:repeating)
        interior_view(halo) .= reshape(collect(1:6), 2, 3)

        @test append_haloarray!(path, "field", halo) === nothing
        data = _read(path, "field")
        @test size(data) == (1, 2, 3)
        @test data[1, :, :] == interior_view(halo)

        # An existing dataset is validated against the array: a smaller array
        # would silently write a partial slab into each step.
        wrong_shape = LocalHaloArray(Int, (2, 4), 1; boundary_condition=:repeating)
        @test_throws DimensionMismatch append_haloarray!(path, "field", wrong_shape)
        wrong_eltype = LocalHaloArray(Float64, (2, 3), 1; boundary_condition=:repeating)
        @test_throws ArgumentError append_haloarray!(path, "field", wrong_eltype)
        interior_view(halo) .+= 10
        append_haloarray!(path, "field", halo)          # a matched append grows
        data = _read(path, "field")
        @test size(data) == (2, 2, 3)
        @test data[2, :, :] == reshape(collect(11:16), 2, 3)

        # Something else under that name is refused too.
        h5open(path, "r+") do fid; create_group(fid, "grp"); end
        @test_throws ArgumentError append_haloarray!(path, "grp", halo)
        # A fixed-time-axis dataset (not made for appending) is refused up front.
        h5open(path, "r+") do fid
            create_dataset(fid, "fixed", Int, dataspace((2, 2, 3)); chunk=(1, 2, 3))
        end
        @test_throws ArgumentError append_haloarray!(path, "fixed", halo)
        rm(path; force=true)
    end

    @testset "LocalHaloArray: handle form, returned dataset, attributes, groups" begin
        path = _h5("local_handle"); rm(path; force=true)
        halo = LocalHaloArray(Float64, (2, 3), 1; boundary_condition=:repeating)
        h5open(path, "w") do fid
            for step in 1:3
                fill!(halo, Float64(step))
                dset = append_haloarray!(fid, "field", halo)
                @test dset isa HDF5.Dataset
                @test size(dset) == (step, 2, 3)
                step == 3 && (attributes(dset)["t"] = 1.5)   # the caller keeps the handle
            end
            g = create_group(fid, "run1")
            @test append_haloarray!(g, "field", halo) isa HDF5.Dataset
        end
        data = _read(path, "field")
        @test all(all(data[s, :, :] .== s) for s in 1:3)
        @test h5open(path, "r") do fid; read(attributes(fid["field"])["t"]); end == 1.5
        @test size(_read(path, "run1/field")) == (1, 2, 3)
        rm(path; force=true)
    end

    @testset "snapshot: gather_haloarray + plain HDF5.jl" begin
        path = _h5("snapshot"); rm(path; force=true)
        halo = LocalHaloArray(Int, (2, 3), 1; boundary_condition=:repeating)
        interior_view(halo) .= reshape(collect(1:6), 2, 3)
        A = gather_haloarray(halo)
        is_root(halo) && h5write(path, "dataset", A)
        @test _read(path, "dataset") == interior_view(halo)
        rm(path; force=true)
    end

    @testset "ThreadedHaloArray append: tiles stitched in global order" begin
        path = _h5("threaded"); rm(path; force=true)
        halo = ThreadedHaloArray(Int, (2,), 1; dims=(3,), boundary_condition=:repeating)
        for tile_id in 1:tile_count(halo)
            interior_view(halo, tile_id) .= (2 * tile_id - 1):(2 * tile_id)
        end
        append_haloarray!(path, "field", halo)
        data = _read(path, "field")
        @test size(data) == (1, 6)
        @test vec(data[1, :]) == collect(1:6)
        rm(path; force=true)
    end

    @testset "ArrayOfHaloArray append: field axes first" begin
        path = _h5("arrayof"); rm(path; force=true)
        u = LocalHaloArray(Int, (2, 3), 1; boundary_condition=:repeating)
        v = similar(u)
        interior_view(u) .= reshape(collect(1:6), 2, 3)
        interior_view(v) .= reshape(collect(101:106), 2, 3)
        fields = ArrayOfHaloArray([u, v])
        append_haloarray!(path, "state", fields)
        data = _read(path, "state")
        @test size(data) == (1, 2, 2, 3)
        @test data[1, 1, :, :] == interior_view(u)
        @test data[1, 2, :, :] == interior_view(v)
        rm(path; force=true)

        # threaded fields
        path = _h5("arrayof_threaded"); rm(path; force=true)
        tu = ThreadedHaloArray(Int, (2,), 1; dims=(2,), boundary_condition=:repeating)
        tv = similar(tu)
        interior_view(tu, 1) .= [1, 2]; interior_view(tu, 2) .= [3, 4]
        interior_view(tv, 1) .= [10, 20]; interior_view(tv, 2) .= [30, 40]
        append_haloarray!(path, "state", ArrayOfHaloArray([tu, tv]))
        data = _read(path, "state")
        @test size(data) == (1, 2, 4)
        @test vec(data[1, 1, :]) == [1, 2, 3, 4]
        @test vec(data[1, 2, :]) == [10, 20, 30, 40]
        rm(path; force=true)
    end

    @testset "MultiHaloArray append: a group with one dataset per field" begin
        path = _h5("multi"); rm(path; force=true)
        mk(dims) = MultiHaloArray((;
            rho=LocalHaloArray(Int, dims, 1; boundary_condition=:repeating),
            mom=LocalHaloArray(Int, dims, 1; boundary_condition=:repeating)))
        state = mk((2, 3))
        interior_view(state.rho) .= reshape(collect(1:6), 2, 3)
        interior_view(state.mom) .= reshape(collect(101:106), 2, 3)

        h5open(path, "w") do fid
            g = append_haloarray!(fid, "state", state)
            @test g isa HDF5.Group
            @test keys(g) == ["mom", "rho"]
        end
        @test _read(path, "state/rho")[1, :, :] == interior_view(state.rho)
        @test _read(path, "state/mom")[1, :, :] == interior_view(state.mom)
        # each existing child dataset is validated
        @test_throws DimensionMismatch append_haloarray!(path, "state", mk((2, 4)))
        append_haloarray!(path, "state", state)          # matched append still grows
        @test size(_read(path, "state/mom")) == (2, 2, 3)
        rm(path; force=true)

        # threaded fields, and a nested ArrayOfHaloArray field
        path = _h5("multi_nested"); rm(path; force=true)
        rho = ThreadedHaloArray(Int, (2,), 1; dims=(2,), boundary_condition=:repeating)
        interior_view(rho, 1) .= [1, 2]; interior_view(rho, 2) .= [3, 4]
        q1 = LocalHaloArray(Int, (2,), 1; boundary_condition=:repeating)
        q2 = similar(q1)
        interior_view(q1) .= [1, 2]; interior_view(q2) .= [3, 4]
        scalar = similar(q1); interior_view(scalar) .= [7, 8]
        append_haloarray!(path, "t", MultiHaloArray((; rho, mom=copy(rho))))
        @test vec(_read(path, "t/rho")[1, :]) == [1, 2, 3, 4]
        # a leaf beside a collection (a record): one dataset per named field
        append_haloarray!(path, "r", MultiHaloArray((; scalar, q=ArrayOfHaloArray([q1, q2]))))
        @test vec(_read(path, "r/scalar")[1, :]) == [7, 8]
        @test size(_read(path, "r/q")) == (1, 2, 2)
        # a rectangular nesting is a group of one dataset per inner collection
        p = ArrayOfHaloArray([scalar, copy(scalar)])
        append_haloarray!(path, "n", MultiHaloArray((; p, q=ArrayOfHaloArray([q1, q2]))))
        pd = _read(path, "n/p")
        @test size(pd) == (1, 2, 2)
        @test vec(pd[1, 1, :]) == [7, 8] && vec(pd[1, 2, :]) == [7, 8]
        q = _read(path, "n/q")
        @test size(q) == (1, 2, 2)
        @test vec(q[1, 1, :]) == [1, 2]
        @test vec(q[1, 2, :]) == [3, 4]
        # an ArrayOfHaloArray of ArrayOfHaloArrays is one dataset, outer axis first
        aa = ArrayOfHaloArray([p, ArrayOfHaloArray([q1, q2])])
        append_haloarray!(path, "aa", aa)
        d = _read(path, "aa")
        @test size(d) == (1, 2, 2, 2)
        @test vec(d[1, 1, 2, :]) == [7, 8] && vec(d[1, 2, 2, :]) == [3, 4]
        # a MultiHaloArray inside an ArrayOfHaloArray has no single-dataset layout
        @test_throws ArgumentError append_haloarray!(path, "bad",
            ArrayOfHaloArray([MultiHaloArray((; a=q1, b=q2)), MultiHaloArray((; a=q1, b=q2))]))
        rm(path; force=true)
    end
end
