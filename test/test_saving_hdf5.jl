using MPI
using HDF5
using Test
using HaloArrays

_h5(name, comm) = joinpath(tempdir(), "haloarrays_$(name)_$(MPI.Comm_size(comm)).h5")
function _rm_on_root(path, comm)
    MPI.Comm_rank(comm) == 0 && rm(path; force=true)
    MPI.Barrier(comm)
end
function _owned(halo)
    dims = HaloArrays.interior_size(halo)
    coords = halo.topology.cart_coords
    return ntuple(d -> (coords[d] * dims[d] + 1):((coords[d] + 1) * dims[d]), Val(ndims(halo)))
end
_rank_owned(topology, r, owned_dims) = (coords = Tuple(MPI.Cart_coords(topology.cart_comm, r));
    ntuple(d -> (coords[d] * owned_dims[d] + 1):((coords[d] + 1) * owned_dims[d]), Val(2)))

@testset "MPI HDF5 output" begin
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    owned_dims = (2, 3)
    boundary = ntuple(_ -> (Periodic(), Periodic()), Val(2))
    topology = CartesianTopology(comm, (0, 0); periodic=(true, true))
    mk() = HaloArray(Float64, owned_dims, 1, topology; boundary_condition=boundary)

    @testset "handle form: collective appends, every rank writes its block" begin
        halo = mk()
        path = _h5("append", comm)
        _rm_on_root(path, comm)
        h5open(path, "w", comm, MPI.Info()) do fid
            for step in 0:2
                fill!(halo, rank + step / 10)
                dset = append_haloarray!(fid, "field", halo)
                @test size(dset) == (step + 1, size(halo)...)
            end
        end
        MPI.Barrier(comm)
        h5open(path, "r", comm, MPI.Info()) do fid
            dset = fid["field"]
            @test size(dset) == (3, size(halo)...)
            for step in 1:3
                @test all(dset[step, _owned(halo)...] .== rank + (step - 1) / 10)
            end
        end
        _rm_on_root(path, comm)
    end

    @testset "filename form: opens on the array's communicator" begin
        halo = mk()
        path = _h5("append_file", comm)
        _rm_on_root(path, comm)
        for step in 0:1
            fill!(halo, rank + 1 + step)
            @test append_haloarray!(path, "field", halo) === nothing
        end
        MPI.Barrier(comm)
        h5open(path, "r", comm, MPI.Info()) do fid
            dset = fid["field"]
            @test size(dset) == (2, size(halo)...)
            @test all(dset[2, _owned(halo)...] .== rank + 2)
        end
        _rm_on_root(path, comm)
    end

    @testset "ArrayOfHaloArray and MultiHaloArray appends" begin
        u = mk(); v = mk()
        fill!(u, rank + 1); fill!(v, 100 + rank)
        path = _h5("collections", comm)
        _rm_on_root(path, comm)
        h5open(path, "w", comm, MPI.Info()) do fid
            dset = append_haloarray!(fid, "state", ArrayOfHaloArray([u, v]))
            @test size(dset) == (1, 2, size(u)...)
            g = append_haloarray!(fid, "named", MultiHaloArray((; rho=u, mom=v)))
            @test g isa HDF5.Group
        end
        MPI.Barrier(comm)
        h5open(path, "r", comm, MPI.Info()) do fid
            dset = fid["state"]
            @test all(dset[1, 1, _owned(u)...] .== rank + 1)
            @test all(dset[1, 2, _owned(u)...] .== 100 + rank)
            @test size(fid["named/rho"]) == (1, size(u)...)
            @test all(fid["named/mom"][1, _owned(u)...] .== 100 + rank)
        end
        _rm_on_root(path, comm)
    end

    @testset "snapshot: gather_haloarray + plain HDF5.jl on root" begin
        u = mk(); v = mk()
        fill!(u, rank + 10); fill!(v, rank + 110)
        path = _h5("gather", comm)
        _rm_on_root(path, comm)
        A = gather_haloarray(ArrayOfHaloArray([u, v]))
        nt = gather_haloarray(MultiHaloArray((; rho=u, mom=v)))
        if is_root(u)
            h5write(path, "state", A)
            h5write(path, "rho", nt.rho)
        end
        MPI.Barrier(comm)
        if rank == 0
            data = h5read(path, "state")
            @test size(data) == (2, size(u)...)
            rho = h5read(path, "rho")
            for r in 0:(nranks - 1)
                o = _rank_owned(topology, r, owned_dims)
                @test all(data[1, o...] .== r + 10)
                @test all(data[2, o...] .== r + 110)
                @test all(rho[o...] .== r + 10)
            end
        else
            @test A === nothing && nt.rho === nothing
        end
        _rm_on_root(path, comm)
    end

    @testset "MaybeHaloArray (dims= reduction) appends and gathers" begin
        u = HaloArray(Int, owned_dims, 1, topology; boundary_condition=boundary)
        fill!(u, rank + 40)
        reduced = mapreduce(identity, +, u; dims=(1,))   # lives on one slice of the grid
        expected_col(y) = sum(0:(topology.dims[1] - 1)) do x
            owned_dims[1] * (MPI.Cart_rank(topology.cart_comm, (x, y)) + 40)
        end

        path = _h5("maybe_append", comm)
        _rm_on_root(path, comm)
        append_haloarray!(path, "reduced", reduced)           # inactive ranks: no-op
        append_haloarray!(path, "reduced", reduced)
        MPI.Barrier(comm)
        if rank == 0
            data = h5read(path, "reduced")
            @test size(data) == (2, topology.dims[2] * owned_dims[2])
            for y in 0:(topology.dims[2] - 1)
                yr = (y * owned_dims[2] + 1):((y + 1) * owned_dims[2])
                @test all(data[2, yr] .== expected_col(y))
            end
        end
        _rm_on_root(path, comm)

        # gather recipe on the reduced result
        A = gather_haloarray(reduced)
        if is_root(reduced)
            @test size(A) == (topology.dims[2] * owned_dims[2],)
            @test all(A[1:owned_dims[2]] .== expected_col(0))
        elseif !is_active(reduced)
            @test A === nothing
        end

        # a reduced MultiHaloArray: group per field, filename form
        rho = HaloArray(Int, owned_dims, 1, topology; boundary_condition=boundary)
        mom = similar(rho)
        fill!(rho, rank + 50); fill!(mom, rank + 150)
        # collection-global dims: field axis 1, spatial axes 2… → spatial dim 1 is (2,)
        reduced_fields = sum(MultiHaloArray((; rho, mom)); dims=2)
        path = _h5("maybe_multi", comm)
        _rm_on_root(path, comm)
        append_haloarray!(path, "reduced", reduced_fields)
        MPI.Barrier(comm)
        if rank == 0
            rho_data = h5read(path, "reduced/rho")
            mom_data = h5read(path, "reduced/mom")
            @test size(rho_data) == (1, topology.dims[2] * owned_dims[2])
            for y in 0:(topology.dims[2] - 1)
                yr = (y * owned_dims[2] + 1):((y + 1) * owned_dims[2])
                @test all(rho_data[1, yr] .== expected_col(y) + owned_dims[1] * 10 * topology.dims[1])
                @test all(mom_data[1, yr] .== expected_col(y) + owned_dims[1] * 110 * topology.dims[1])
            end
        end
        _rm_on_root(path, comm)
    end
end
