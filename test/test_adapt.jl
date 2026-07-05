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

    @testset "BLAS-1 on a device parent (Array-gated kernels fall back correctly)" begin
        using LinearAlgebra: axpy!, axpby!, rmul!, lmul!, rotate!, reflect!
        mk(v) = (u = LocalHaloArray(Float64, (5,), 1; boundary_condition=:periodic);
                 interior_view(u) .= v; u)
        c, s = 0.6, 0.8
        # each op on JLArray must match the same op on the host array (the scalar
        # kernels are Array-gated; device parents take the broadcast fallback).
        for (host_op, dev_setup) in (
                (x -> rmul!(x, 2.0),                  () -> (mk([1.0,2,3,4,5]),)),
                (x -> lmul!(3.0, x),                  () -> (mk([1.0,2,3,4,5]),)),
            )
            hx, = dev_setup(); host_op(hx)
            gx, = dev_setup(); gx = adapt(JLArray, gx); host_op(gx)
            @test Array(interior_view(gx)) ≈ collect(interior_view(hx))
        end
        # two-array ops
        for op in (axpy!, axpby!)
            args = op === axpy! ? (2.0,) : (2.0, 0.5)
            hx, hy = mk([1.0,2,3,4,5]), mk([10.0,20,30,40,50])
            op === axpy! ? axpy!(2.0, hx, hy) : axpby!(2.0, hx, 0.5, hy)
            gx, gy = adapt(JLArray, mk([1.0,2,3,4,5])), adapt(JLArray, mk([10.0,20,30,40,50]))
            op === axpy! ? axpy!(2.0, gx, gy) : axpby!(2.0, gx, 0.5, gy)
            @test Array(interior_view(gy)) ≈ collect(interior_view(hy))
        end
        # swap! / rotate! / reflect! (in-place two-output; device path uses one temp)
        for op! in (swap!, (a,b)->rotate!(a,b,c,s), (a,b)->reflect!(a,b,c,s))
            hx, hy = mk([1.0,2,3,4,5]), mk([10.0,20,30,40,50]); op!(hx, hy)
            gx, gy = adapt(JLArray, mk([1.0,2,3,4,5])), adapt(JLArray, mk([10.0,20,30,40,50])); op!(gx, gy)
            @test Array(interior_view(gx)) ≈ collect(interior_view(hx))
            @test Array(interior_view(gy)) ≈ collect(interior_view(hy))
        end
    end

    @testset "reductions on a device parent (generic _interior_acc/_interior_dot fallbacks)" begin
        using LinearAlgebra: norm, dot
        mk(v) = (u = LocalHaloArray(Float64, (5,), 1; boundary_condition=:periodic);
                 interior_view(u) .= v; u)
        hx, hy = mk([1.0, 2, 3, 4, 5]), mk([10.0, 20, 30, 40, 50])
        gx, gy = adapt(JLArray, mk([1.0, 2, 3, 4, 5])), adapt(JLArray, mk([10.0, 20, 30, 40, 50]))
        # the Array-gated SIMD kernels must be bypassed; JLArray forbids scalar indexing
        @test sum(gx) ≈ sum(hx)
        @test norm(gx) ≈ norm(hx)
        @test dot(gx, gy) ≈ dot(hx, hy)
        @test mapreduce(abs2, +, gx) ≈ mapreduce(abs2, +, hx)
        @test fill!(gx, 7.0) === gx
        @test sum(gx) ≈ 5 * 7.0
        copyto!(gy, gx)
        @test Array(interior_view(gy)) ≈ fill(7.0, 5)
    end
end
