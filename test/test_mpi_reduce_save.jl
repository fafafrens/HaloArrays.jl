using MPI
using HDF5
using Test
using HaloArrays

# Dimensional reduction → MaybeHaloArray → save → read back, both ways:
#   (1) collective MPI write (append_haloarray_to_file!, no gather)
#   (2) gather-to-root then save (gather_and_save_haloarray)
# Both must reproduce the serial reference reduction exactly. The kept dimension is
# split across all ranks so the reduced result stays distributed (every rank
# active) — exercising the real parallel-write path, not a single-rank degenerate.
_rm_on_root(path, comm) = (MPI.Comm_rank(comm) == 0 && rm(path; force=true); MPI.Barrier(comm))

@testset "MPI dim-reduce → Maybe → save (collective vs gather)" begin
    comm  = MPI.COMM_WORLD
    rank  = MPI.Comm_rank(comm)
    nr    = MPI.Comm_size(comm)
    @test nr > 1

    f(I) = I[1] + 100 * I[2]
    GX = 8 * nr        # divisible by nr
    GY = 12
    topo  = CartesianTopology(comm, (nr, 1); periodic=(true, true))   # split kept dim 1
    bc    = ((Periodic(), Periodic()), (Periodic(), Periodic()))
    u = HaloArray(Float64, (GX ÷ nr, GY), 1, topo; boundary_condition=bc)
    fill_from_global_indices!(f, u)

    # reduce over dim 2 → MaybeHaloArray (global length GX, distributed over dim 1)
    r = mapreduce_haloarray_dims(identity, +, u, 2)
    @test r isa MaybeHaloArray

    ref = Float64[sum(f((i, j)) for j in 1:GY) for i in 1:GX]   # serial reference

    col = joinpath(tempdir(), "haloarrays_reduce_collective_$(nr)")
    gat = joinpath(tempdir(), "haloarrays_reduce_gather_$(nr)")
    for base in (col, gat); _rm_on_root(base * ".h5", comm); end

    append_haloarray_to_file!(col, "reduced", r)   # (1) collective, no gather
    MPI.Barrier(comm)
    gather_and_save_haloarray(gat, r)              # (2) gather then save
    MPI.Barrier(comm)

    if rank == 0
        d_col = vec(h5open(col * ".h5", "r") do fid; read(fid["reduced"]); end)  # (1,GX) extensible
        d_gat = vec(h5open(gat * ".h5", "r") do fid; read(fid["dataset"]); end)  # (GX,) snapshot
        @test length(d_col) == GX
        @test length(d_gat) == GX
        @test d_col ≈ ref          # collective write reproduces the serial reduction
        @test d_gat ≈ ref          # gather write reproduces the serial reduction
        @test d_col ≈ d_gat        # the two paths agree
    end

    for base in (col, gat); _rm_on_root(base * ".h5", comm); end
    MPI.Barrier(comm)
end
