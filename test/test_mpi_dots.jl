using Test
using HaloArrays
using MPI
using LinearAlgebra: dot

# `dots!` on a distributed HaloArray: the j inner products must equal j separate
# (global) `dot` calls, be identical on every rank, and cost ONE Allreduce.
@testset "MPI dots!: j inner products, one Allreduce" begin
    comm = MPI.COMM_WORLD
    nr = MPI.Comm_size(comm)
    GNX, GNY = 24, 10
    @test GNX % nr == 0
    topo = CartesianTopology(comm, (0, 1); periodic = (true, true))
    mk(T) = HaloArray(T, (GNX ÷ nr, GNY), 1, topo;
                      boundary_condition = ((Periodic(), Periodic()), (Periodic(), Periodic())))
    for T in (Float64, ComplexF64)
        j = 6
        V = [mk(T) for _ in 1:j+1]
        for (i, v) in enumerate(V)
            fill_from_global_indices!(I -> T(sin(0.3I[1] + 0.7I[2] + i)) + (T <: Complex ? im * T(cos(I[1] - i)) : zero(T)), v)
        end
        w = mk(T); fill_from_global_indices!(I -> T(cos(0.2I[1] * I[2])), w)
        h = fill(T(NaN), j + 2)
        @test dots!(h, V, w, j) === h
        ref = [dot(V[i], w) for i in 1:j]          # each a global reduction
        @test h[1:j] ≈ ref
        @test all(isnan, h[j+1:end])
        # every rank holds the same numbers
        root = MPI.bcast(h[1:j], 0, comm)
        @test h[1:j] == root
    end
    # Reduced exactly once: with every V[i] = 1 and w = 2 the global product is
    # 2·GNX·GNY; a second reduction would multiply it by the rank count, a
    # missing one would leave each rank's share.
    j = 4
    V = [mk(Float64) for _ in 1:j]; w = mk(Float64); h = zeros(j)
    foreach(v -> fill!(v, 1.0), V); fill!(w, 2.0)
    dots!(h, V, w, j)
    @test all(h .≈ 2.0 * GNX * GNY)
end
