using Test
using HaloArrays
using KernelAbstractions
using Metal

const KA = KernelAbstractions

@kernel function _read_periodic_halo_kernel!(out, data)
    out[1] = data[1]
    out[2] = data[6]
end

@kernel function _colored_face_update_kernel!(
        du,
        u,
        touch_count,
        first_i::Int,
        stride_i::Int,
        offset_i::Int,
        lower_owned::Bool,
        upper_owned::Bool,
)
    i = @index(Global, Linear)

    IL = first_i + (i - 1) * stride_i
    IR = IL + offset_i
    flux = u[IR] - u[IL]

    if lower_owned
        du[IL] -= flux
        touch_count[IL] += 1
    end
    if upper_owned
        du[IR] += flux
        touch_count[IR] += 1
    end
end

function _launch_colored_face_update!(kernel!, du, u, touch_count, region::ColoredFaceKernelRegion)
    nface = region.size[1]
    nface == 0 && return nothing
    kernel!(
        du,
        u,
        touch_count,
        Tuple(region.first)[1],
        Tuple(region.stride)[1],
        Tuple(region.offset)[1],
        region.lower_owned,
        region.upper_owned;
        ndrange=nface,
    )
    return nothing
end

@kernel function _colored_cell_touch_kernel!(
        touch_count,
        region::ColoredCellKernelRegion{2},
)
    I = cell_index(region, @index(Global, NTuple))

    if is_cell_index_inbounds(region, I)
        i, j = I
        touch_count[i, j] += Int32(1)
    end
end

function _launch_colored_cell_touch!(kernel!, touch_count, region::ColoredCellKernelRegion)
    region.compressed_dim == 1 || throw(ArgumentError("Metal probe expects compressed_dim = 1"))
    any(==(0), region.size) && return nothing

    kernel!(
        touch_count,
        region;
        ndrange=region.size,
    )
    return nothing
end

@testset "Metal LocalHaloArray boundary ordering" begin
    u = LocalHaloArray(Metal.MtlArray(Float32[10, 1, 2, 3, 4, 20]), 1, :periodic)
    out = Metal.zeros(Float32, 2)
    backend = KA.get_backend(parent(u))
    read_halo! = _read_periodic_halo_kernel!(backend)

    boundary_condition!(u)
    read_halo!(out, parent(u); ndrange=1)
    KA.synchronize(backend)

    @test Array(out) == Float32[4, 1]
    @test Array(parent(u)) == Float32[4, 1, 2, 3, 4, 1]
end

@testset "Metal colored face regions" begin
    u = LocalHaloArray(Metal.MtlArray(Float32[100, 1, 2, 3, 4, 200]), 1, :repeating)
    du = similar(u)
    touch_count = Metal.zeros(Int32, length(parent(u)))
    backend = KA.get_backend(parent(u))
    update! = _colored_face_update_kernel!(backend)
    ranges = FaceRanges(u)

    fill!(parent(du), 0.0f0)
    for color in 0:1
        fill!(touch_count, Int32(0))
        for region in (
                get_colored_left_face_region(ranges, 1, color),
                get_colored_internal_face_region(ranges, 1, color),
                get_colored_right_face_region(ranges, 1, color),
        )
            _launch_colored_face_update!(update!, parent(du), parent(u), touch_count, region)
        end
        KA.synchronize(backend)
        @test maximum(Array(touch_count)) <= 1
    end

    @test Array(parent(du)) == Float32[0, -100, 0, 0, -195, 0]
end

@testset "Metal colored cell regions" begin
    u = LocalHaloArray(Metal.zeros(Float32, 6, 7), 1, :repeating)
    touch_count = Metal.zeros(Int32, size(parent(u))...)
    backend = KA.get_backend(parent(u))
    touch! = _colored_cell_touch_kernel!(backend)
    ranges = CellRanges(u)

    for color in 0:1
        region = get_colored_owned_cell_region(ranges, color, Dim(1))
        _launch_colored_cell_touch!(touch!, touch_count, region)
        KA.synchronize(backend)
    end

    touches = Array(touch_count)
    @test all(touches[2:5, 2:6] .== 1)
    @test sum(touches) == 20
end
