using Test
using HaloArrays
using MPI
import LinearAlgebra
using LinearAlgebra: dot, norm, mul!
using HaloArrays.Krylov

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
    # A collection of distributed fields reduces once as well.
    m = MultiHaloArray((; a = mk(Float64), b = mk(Float64)))
    Vm = [copy(m) for _ in 1:3]; wm = copy(m)
    for (i, v) in enumerate(Vm), f in eachfield(v); fill_from_global_indices!(I -> sin(I[1] + i * I[2]), f); end
    for f in eachfield(wm); fill_from_global_indices!(I -> cos(I[1] * I[2] / 7), f); end
    hm = zeros(3)
    @test dots!(hm, Vm, wm, 3) ≈ [dot(Vm[i], wm) for i in 1:3]
    @test HaloArrays.communicator(m) == HaloArrays.communicator(first(eachfield(m)))

    # Reduced exactly once: with every V[i] = 1 and w = 2 the global product is
    # 2·GNX·GNY; a second reduction would multiply it by the rank count, a
    # missing one would leave each rank's share.
    j = 4
    V = [mk(Float64) for _ in 1:j]; w = mk(Float64); h = zeros(j)
    foreach(v -> fill!(v, 1.0), V); fill!(w, 2.0)
    dots!(h, V, w, j)
    @test all(h .≈ 2.0 * GNX * GNY)
end

# gmres!/fgmres! on a distributed HaloArray with a stencil operator (halo
# exchange inside), both orthogonalisations, against the serial solve of the
# same global problem. :auto must resolve to :cgs2 here.
struct Lap1D{T}
    σ::T
end
function LinearAlgebra.mul!(w, L::Lap1D, v)
    synchronize_halo!(v)
    E = CartesianIndex(1)
    for tile in 1:tile_count(v)
        s = tile_parent(v, tile); d = tile_parent(w, tile)
        @inbounds for I in CartesianIndices(interior_range(v))
            d[I] = (2 + L.σ) * s[I] - s[I - E] - s[I + E]
        end
    end
    return w
end

@testset "MPI gmres!/fgmres!: cgs2 vs mgs vs serial" begin
    comm = MPI.COMM_WORLD
    nr = MPI.Comm_size(comm)
    GNX = 96
    topo = CartesianTopology(comm, (0,); periodic = (true,))
    b = HaloArray(Float64, (GNX ÷ nr,), 1, topo; boundary_condition = ((Periodic(), Periodic()),))
    fill_from_global_indices!(I -> sin(2π * I[1] / GNX) + 0.3 * cos(6π * I[1] / GNX), b)
    @test Krylov.resolve_orth(:auto, b) === :cgs2
    @test Krylov.resolve_orth(:auto, MultiHaloArray((; a = copy(b)))) === :cgs2
    L = Lap1D(0.5)
    # serial reference on rank 0's gathered copy
    bl = LocalHaloArray(Float64, (GNX,), 1; boundary_condition = :periodic)
    fill_from_global_indices!(I -> sin(2π * I[1] / GNX) + 0.3 * cos(6π * I[1] / GNX), bl)
    xl = zero(bl)
    rl = gmres!(xl, L, bl, GMRESWorkspace(bl; restart = 20); abstol = 0.0, reltol = 1e-10, maxiter = 400)
    @test rl.converged
    ref = gather_haloarray(xl)
    for (f, mk) in ((gmres!, GMRESWorkspace), (fgmres!, FGMRESWorkspace)), orth in (:auto, :mgs, :cgs2)
        x = zero(b)
        r = f(x, L, b, mk(b; restart = 20); orth, abstol = 0.0, reltol = 1e-10, maxiter = 400)
        @test r.converged
        @test r.iters == rl.iters
        g = gather_haloarray(x)
        MPI.Comm_rank(comm) == 0 && @test g ≈ ref
    end
end
