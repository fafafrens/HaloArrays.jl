using Test
using HaloArrays
using MPI
using LinearAlgebra: norm, dot
using Adapt
using JLArrays   # GPUArrays' CPU reference backend: a distinct array type that
                 # errors on scalar indexing, so it exercises the GPU code paths.

MPI.Initialized() || MPI.Init()

@testset "Adapt to/from a device array type (JLArray)" begin
    @testset "LocalHaloArray" begin
        u = LocalHaloArray(Float64, (12, 9), 1; boundary_condition=:periodic)
        interior_view(u) .= rand(12, 9)

        ud = adapt(JLArray, u)
        @test parent(ud) isa JLArray                       # data moved to device
        @test eltype(ud) == Float64 && halo_width(ud) == 1
        # device-native ops: JLArray throws on scalar indexing, so these passing
        # proves the paths are scalar-free.
        synchronize_halo!(ud)
        @test norm(ud) ≈ norm(u)
        @test sum(ud)  ≈ sum(u)

        uc = adapt(Array, ud)                              # round-trip back to host
        @test parent(uc) isa Array
        @test interior_view(uc) ≈ interior_view(u)
    end

    @testset "HaloArray (MPI) — buffers must follow data to the device" begin
        topo = CartesianTopology(MPI.COMM_WORLD, (0, 0); periodic=(true, true))
        bc = ((Periodic(), Periodic()), (Periodic(), Periodic()))
        h = HaloArray(Float64, (10, 10), 1, topo; boundary_condition=bc)
        interior_view(h) .= rand(10, 10)

        hd = adapt(JLArray, h)
        @test parent(hd) isa JLArray
        # the critical invariant: send/recv buffers live on the SAME device as data,
        # else the exchange would stage host↔device and hand host pointers to MPI.
        @test hd.send_bufs[1][1] isa JLArray
        @test hd.receive_bufs[1][1] isa JLArray
        @test hd.send_bufs[2][2] isa JLArray
        # global reductions agree with the host array (scalar-free GPU fallback).
        @test norm(hd) ≈ norm(h)
        @test sum(hd)  ≈ sum(h)
    end

    @testset "ThreadedHaloArray — each tile's storage is adapted" begin
        t = ThreadedHaloArray(Float64, (8, 8), 1; dims=(2, 1), boundary_condition=:periodic)
        for k in 1:tile_count(t); interior_view(t, k) .= rand(8, 8); end
        td = adapt(JLArray, t)
        @test tile_parent(td, 1) isa JLArray
        @test tile_parent(td, 2) isa JLArray
        @test norm(td) ≈ norm(t)
    end
end
