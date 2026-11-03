using Test
using HaloArrays
using JLArrays            # GPUArrays' CPU reference backend: a KernelAbstractions non-CPU backend
using KernelAbstractions
const KA = KernelAbstractions

@testset "KernelAbstractions extension: GPU face copies" begin
    # JLArray storage must take the kernel path for these tests to mean anything
    @test !(KA.get_backend(JLArray(zeros(2))) isa KA.CPU)

    # all-periodic single blocks: the fused fill equals the broadcast fill, 1–4D, halo 1–2
    for (dims, h) in (((6,), 1), ((5, 6), 1), ((4, 5, 6), 2), ((3, 4, 3, 4), 1))
        x = rand(dims .+ 2h...)
        ref = LocalHaloArray(copy(x), h, :periodic)
        synchronize_halo!(ref)
        dev = LocalHaloArray(JLArray(copy(x)), h, :periodic)
        @test HaloArrays._fused_periodic_fill!(dev)               # handled by the fused kernel
        @test Array(parent(dev)) == parent(ref)
        dev2 = LocalHaloArray(JLArray(copy(x)), h, :periodic)
        synchronize_halo!(dev2)                                    # the public entry point
        @test Array(parent(dev2)) == parent(ref)
    end

    # mixed conditions: the per-face path, periodic faces through the slab kernel
    x = rand(8, 9)
    bc = ((Periodic(), Periodic()), (Repeating(), Repeating()))
    ref = LocalHaloArray(copy(x), 1, bc)
    synchronize_halo!(ref)
    dev = LocalHaloArray(JLArray(copy(x)), 1, bc)
    @test !HaloArrays._fused_periodic_fill!(dev)
    synchronize_halo!(dev)
    @test Array(parent(dev)) == parent(ref)

    # slab copy between a view and a dense buffer, both directions, and the shape check
    a = JLArray(rand(6, 7))
    v = view(a, 2:5, 3:5)
    buf = JLArray(zeros(4, 3))
    HaloArrays._copy_slab!(buf, v)
    @test Array(buf) == Array(a)[2:5, 3:5]
    b = JLArray(zeros(6, 7))
    HaloArrays._copy_slab!(view(b, 2:5, 3:5), buf)
    @test Array(b)[2:5, 3:5] == Array(buf)
    @test_throws DimensionMismatch HaloArrays._copy_slab!(JLArray(zeros(3, 3)), v)

    # host storage keeps the broadcast path; the device synchronization is a no-op there
    @test HaloArrays._slab_backend(view(zeros(4, 4), 1:2, 1:2), zeros(2, 2)) === nothing
    @test HaloArrays._device_synchronize(zeros(2)) === nothing
    @test HaloArrays._device_synchronize(JLArray(zeros(2))) === nothing
end
