using Test
using HaloArrays
using JLArrays            # GPUArrays' CPU reference backend: a KernelAbstractions non-CPU backend
using KernelAbstractions
using MPI, Adapt
const KA = KernelAbstractions

struct DirichletExt{T} <: HaloArrays.AbstractBoundaryCondition; value::T; end
(bc::DirichletExt)(u, k) = 2bc.value - u(0)

@testset "KernelAbstractions extension: GPU face copies" begin
    # JLArray storage must take the kernel path for these tests to mean anything
    @test !(KA.get_backend(JLArray(zeros(2))) isa KA.CPU)

    # all-periodic single blocks: the fused fill equals the broadcast fill, 1–4D, halo 1–2
    for (dims, h) in (((6,), 1), ((5, 6), 1), ((4, 5, 6), 2), ((3, 4, 3, 4), 1))
        x = rand(dims .+ 2h...)
        ref = LocalHaloArray(copy(x), h, :periodic)
        synchronize_halo!(ref)
        dev = LocalHaloArray(JLArray(copy(x)), h, :periodic)
        @test HaloArrays._fused_fill!(dev)                        # handled by the fused kernel
        @test Array(parent(dev)) == parent(ref)
        dev2 = LocalHaloArray(JLArray(copy(x)), h, :periodic)
        synchronize_halo!(dev2)                                    # the public entry point
        @test Array(parent(dev2)) == parent(ref)
    end

    # `storage=` allocates on the device directly; the collection constructors pass it down
    jlzeros(T, d...) = JLArray(zeros(T, d...))
    g = LocalHaloArray(Float32, (5, 6), 1; boundary_condition=:periodic, storage=jlzeros)
    @test parent(g) isa JLArray{Float32,2} && size(parent(g)) == (7, 8)
    gref = LocalHaloArray(Float32, (5, 6), 1; boundary_condition=:periodic)
    interior_view(g) .= 1.0f0
    interior_view(gref) .= 1.0f0
    synchronize_halo!(g)
    synchronize_halo!(gref)
    @test Array(parent(g)) == parent(gref)
    gs = MultiHaloArray(LocalHaloArray, Float32, (5, 6), 1; fields=(:a, :b), boundary_condition=:periodic, storage=jlzeros)
    @test parent(gs.b) isa JLArray{Float32,2}
    gv = ArrayOfHaloArray(LocalHaloArray, Float32, (2,), (5, 6), 1; boundary_condition=:periodic, storage=jlzeros)
    @test parent(gv[2]) isa JLArray{Float32,2}

    # mixed conditions: one fused launch over all rule faces
    x = rand(8, 9)
    bc = ((Periodic(), Periodic()), (Repeating(), Repeating()))
    ref = LocalHaloArray(copy(x), 1, bc)
    synchronize_halo!(ref)
    dev = LocalHaloArray(JLArray(copy(x)), 1, bc)
    @test HaloArrays._fused_fill!(dev)
    @test Array(parent(dev)) == parent(ref)
    dev2 = LocalHaloArray(JLArray(copy(x)), 1, bc)
    synchronize_halo!(dev2)
    @test Array(parent(dev2)) == parent(ref)

    # NoBoundaryCondition faces are skipped by the fused launch: their ghosts keep the sentinel
    x = rand(7, 6, 5); x[1, :, :] .= -7.0; x[end, :, :] .= -7.0
    bcn = ((NoBoundaryCondition(), NoBoundaryCondition()), (Mirror(1), Extrapolation(1)), (Periodic(), Periodic()))
    ref = LocalHaloArray(copy(x), 1, bcn); synchronize_halo!(ref)
    dev = LocalHaloArray(JLArray(copy(x)), 1, bcn)
    @test HaloArrays._fused_fill!(dev)
    @test Array(parent(dev)) == parent(ref)
    @test all(==(-7.0), Array(parent(dev))[1, 2:end-1, 2:end-1]) && all(==(-7.0), Array(parent(dev))[end, 2:end-1, 2:end-1])
    # a block with no rule face at all is handled trivially
    @test HaloArrays._fused_fill!(LocalHaloArray(JLArray(zeros(4, 4)), 1, :noboundary))

    # the per-face kernel is still the path for a single face
    dev = LocalHaloArray(JLArray(copy(x)), 1, bcn)
    boundary_condition!(dev, Side(1), Dim(2))
    @test Array(parent(dev))[2:end-1, 1, 2:end-1] == parent(ref)[2:end-1, 1, 2:end-1]

    # MPI HaloArray on one rank, device storage: physical faces fused, periodic faces are exchange faces
    if MPI.Comm_size(MPI.COMM_WORLD) == 1
        topo = CartesianTopology(MPI.COMM_WORLD, (0, 0); periodic = (true, false))
        bch = ((Periodic(), Periodic()), (Mirror(-1), Extrapolation(2)))
        h = HaloArray(Float64, (6, 7), 1, topo; boundary_condition = bch)
        interior_view(h) .= rand(6, 7)
        hd = adapt(JLArray, h)
        @test !HaloArrays._rule_owns(hd, Side(1), Dim(1))                      # periodic: exchanged, not filled
        @test HaloArrays._rule_owns(hd, Side(2), Dim(2))
        @test HaloArrays._fused_fill!(hd)
        synchronize_halo!(h)
        @test Array(parent(hd))[2:end-1, 1] == parent(h)[2:end-1, 1]
        @test Array(parent(hd))[2:end-1, end] == parent(h)[2:end-1, end]
    end

    # rules on the device: built-ins, a custom type and a closure, fused, vs the CPU fill
    for (dims, h) in (((7,), 2), ((6, 5), 1), ((4, 5, 6), 2), ((3, 4, 3, 4), 1))
        x = rand(dims .+ 2h...)
        bcr = ntuple(d -> d == 1 ? (Mirror(-1), Extrapolation(2)) :
                          d == 2 ? (Repeating(), DirichletExt(0.5)) :
                          d == 3 ? ((u, k) -> u(0) + 0.1k, Periodic()) :
                                   (Periodic(), Mirror(1)), length(dims))
        ref = LocalHaloArray(copy(x), h, bcr); synchronize_halo!(ref)
        dev = LocalHaloArray(JLArray(copy(x)), h, bcr)
        @test HaloArrays._fused_fill!(dev)
        @test Array(parent(dev)) ≈ parent(ref)
    end

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

    # storage without a KernelAbstractions backend (BitArray) keeps working with the extension loaded
    bits = LocalHaloArray(falses(6, 6), 1, :periodic)
    interior_view(bits) .= true
    @test HaloArrays._device_synchronize(parent(bits)) === nothing
    synchronize_halo!(bits)
    @test all(parent(bits)[2:5, 1]) && all(parent(bits)[1, 2:5]) && !parent(bits)[1, 1]

    # stepped-view storage: the offset kernels do not apply, the broadcast path does
    base = rand(12, 9)
    ref = LocalHaloArray(copy(base)[1:2:12, :], 1, bc)
    synchronize_halo!(ref)
    dbase = JLArray(copy(base))
    strided = LocalHaloArray(view(dbase, 1:2:12, :), 1, bc)
    @test HaloArrays._slab_backend(ghost_view(strided, Side(1), Dim(1)), edge_view(strided, Side(2), Dim(1))) === nothing
    synchronize_halo!(strided)
    @test Array(dbase)[1:2:12, :] == parent(ref)

    # fewer interior cells than the halo width is not a periodic geometry
    @test_throws ArgumentError LocalHaloArray(zeros(5, 8), 2, :periodic)
    @test_throws ArgumentError LocalHaloArray(JLArray(zeros(5, 8)), 2, :periodic)
    @test LocalHaloArray(zeros(5, 8), 2, :repeating) isa LocalHaloArray

    # a rule capturing an array: the capture is adapted with the halo array
    # (host table → device table) and again at launch (device table → kernel form)
    table_rule(table) = FunctionBC((u, k) -> table[k] + u(0))
    x = rand(12); x[3:10] .= 1:8                      # 8 interior cells, halo 2
    ref = LocalHaloArray(copy(x), 2, ((table_rule([10.0, 20.0]), Repeating()),))
    synchronize_halo!(ref)
    @test parent(ref)[1:2] == [20.0 + 1, 10.0 + 1]      # storage index 1 is the ghost at distance 2
    host_captured = LocalHaloArray(JLArray(copy(x)), 2, ((table_rule([10.0, 20.0]), Repeating()),))
    moved = adapt(JLArray, host_captured)
    @test moved.boundary_condition[1][1].f.table isa JLArray
    @test moved.boundary_condition[1][2] === Repeating()
    synchronize_halo!(moved)
    @test Array(parent(moved)) == parent(ref)
    device_captured = LocalHaloArray(JLArray(copy(x)), 2, ((table_rule(JLArray([10.0, 20.0])), Repeating()),))
    synchronize_halo!(device_captured)
    @test Array(parent(device_captured)) == parent(ref)
    @test adapt(Array, moved).boundary_condition[1][1].f.table isa Vector{Float64}
    th = adapt(JLArray, ThreadedHaloArray(LocalHaloArray(copy(x), 2, ((table_rule([10.0, 20.0]), Repeating()),)); dims=(2,)))
    @test th.boundary_condition[1][1].f.table isa JLArray

    # BLAS-1 on device storage: the broadcast fallbacks of `_interior_map!`/`_interior_map2!`
    xa, ya = rand(8, 7), rand(8, 7)
    cx, cy = LocalHaloArray(copy(xa), 1, bc), LocalHaloArray(copy(ya), 1, bc)
    dx, dy = LocalHaloArray(JLArray(copy(xa)), 1, bc), LocalHaloArray(JLArray(copy(ya)), 1, bc)
    for (op!, args) in ((rmul!, (0.5,)), (lmul!, ()), (axpy!, ()), (axpby!, ()), (swap!, ()), (rotate!, (0.6, 0.8)), (reflect!, (0.6, 0.8)))
        if op! === rmul!
            rmul!(cx, 0.5); rmul!(dx, 0.5)
        elseif op! === lmul!
            lmul!(2.0, cy); lmul!(2.0, dy)
        elseif op! === axpy!
            axpy!(1.5, cx, cy); axpy!(1.5, dx, dy)
        elseif op! === axpby!
            axpby!(1.5, cx, -0.25, cy); axpby!(1.5, dx, -0.25, dy)
        elseif op! === swap!
            swap!(cx, cy); swap!(dx, dy)
        else
            op!(cx, cy, args...); op!(dx, dy, args...)
        end
        @test Array(parent(dx)) ≈ parent(cx) && Array(parent(dy)) ≈ parent(cy)
    end
    @test Array(parent(dx))[1, :] == xa[1, :]                         # ghosts untouched
end
