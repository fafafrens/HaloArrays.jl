using HaloArrays
using KernelAbstractions
using Metal
using Printf

const KA = KernelAbstractions

function stable_heat_dt(alpha, cfl, dx)
    return cfl / (alpha * (inv(abs2(dx[1])) + inv(abs2(dx[2]))))
end

function fill_gaussian!(u)
    nx, ny = global_size(u)
    cx = (nx + 1) / 2
    cy = (ny + 1) / 2
    width_x = nx / 10
    width_y = ny / 10

    fill_from_global_indices!(u) do I
        x, y = I
        r2 = ((x - cx) / width_x)^2 + ((y - cy) / width_y)^2
        return 1.0f0 + exp(-r2)
    end
    synchronize_halo!(u)
    return u
end

function heat_step_cpu!(u_next, du, u, alpha, dt, dx)
    zero_owned_cpu!(du)
    accumulate_heat_fluxes_cpu!(du, u, alpha, dx)
    apply_heat_update_cpu!(u_next, u, du, dt)
    return u_next
end

function zero_owned_cpu!(u)
    data = parent(u)
    @inbounds for I in get_owned_cells(CellRanges(u))
        data[I] = zero(eltype(data))
    end
    return u
end

function accumulate_heat_fluxes_cpu!(du, u, alpha, dx)
    ranges = FaceRanges(u)

    for dim in 1:2
        accumulate_heat_fluxes_cpu!(du, u, ranges, dim, alpha / dx[dim]^2)
    end

    return du
end

function accumulate_heat_fluxes_cpu!(du, u, ranges::FaceRanges, dim, scale)
    du_data = parent(du)
    data = parent(u)
    offset = get_unit_vector(ranges, dim)

    @inbounds for IL in get_left_face(ranges, dim)
        IR = IL + offset
        flux = scale * (data[IR] - data[IL])
        du_data[IR] -= flux
    end

    @inbounds for IL in get_internal_face(ranges)
        IR = IL + offset
        flux = scale * (data[IR] - data[IL])
        du_data[IL] += flux
        du_data[IR] -= flux
    end

    @inbounds for IL in get_right_face(ranges, dim)
        IR = IL + offset
        flux = scale * (data[IR] - data[IL])
        du_data[IL] += flux
    end

    return du
end

function apply_heat_update_cpu!(u_next, u, du, dt)
    out = parent(u_next)
    data = parent(u)
    du_data = parent(du)

    @inbounds for I in get_owned_cells(CellRanges(u))
        out[I] = data[I] + dt * du_data[I]
    end

    return u_next
end

@kernel function zero_owned_gpu_kernel!(data, region::CellKernelRegion{2})
    I = cell_index(region, @index(Global, NTuple))
    @inbounds data[I...] = zero(eltype(data))
end

@kernel function heat_flux_gpu_kernel!(du, data, region::ColoredFaceKernelRegion{2}, scale)
    J = @index(Global, NTuple)
    first = Tuple(region.first)
    stride = Tuple(region.stride)
    offset = Tuple(region.offset)

    IL = (
        first[1] + (J[1] - 1) * stride[1],
        first[2] + (J[2] - 1) * stride[2],
    )
    IR = (IL[1] + offset[1], IL[2] + offset[2])

    @inbounds begin
        flux = scale * (data[IR...] - data[IL...])
        if region.lower_owned
            du[IL...] += flux
        end
        if region.upper_owned
            du[IR...] -= flux
        end
    end
end

@kernel function apply_heat_update_gpu_kernel!(out, data, du, dt, region::CellKernelRegion{2})
    I = cell_index(region, @index(Global, NTuple))
    @inbounds out[I...] = data[I...] + dt * du[I...]
end

function launch_cell_kernel!(kernel!, args...)
    region = args[end]
    any(==(0), region.size) && return nothing
    kernel!(args...; ndrange=region.size)
    return nothing
end

function launch_face_kernel!(kernel!, args...)
    region = args[end - 1]
    any(==(0), region.size) && return nothing
    kernel!(args...; ndrange=region.size)
    return nothing
end

function heat_step_gpu!(kernels, u_next, du, u, alpha, dt, dx)
    backend, zero!, flux!, update! = kernels
    cell_region = get_owned_cell_region(CellRanges(u))
    ranges = FaceRanges(u)

    launch_cell_kernel!(zero!, parent(du), cell_region)
    KA.synchronize(backend)

    for dim in 1:2
        scale = Float32(alpha / dx[dim]^2)
        for color in 0:1
            for region in (
                    get_colored_left_face_region(ranges, dim, color),
                    get_colored_internal_face_region(ranges, dim, color),
                    get_colored_right_face_region(ranges, dim, color),
            )
                launch_face_kernel!(flux!, parent(du), parent(u), region, scale)
            end
            KA.synchronize(backend)
        end
    end

    launch_cell_kernel!(update!, parent(u_next), parent(u), parent(du), Float32(dt), cell_region)
    KA.synchronize(backend)
    return u_next
end

function make_gpu_kernels(u)
    backend = KA.get_backend(parent(u))
    return (
        backend,
        zero_owned_gpu_kernel!(backend),
        heat_flux_gpu_kernel!(backend),
        apply_heat_update_gpu_kernel!(backend),
    )
end

function solve_cpu!(u; alpha, dt, dx, steps)
    current = u
    next = similar(u)
    du = similar(u)

    for _ in 1:steps
        synchronize_halo!(current)
        heat_step_cpu!(next, du, current, alpha, dt, dx)
        current, next = next, current
    end

    synchronize_halo!(current)
    return current
end

function solve_gpu!(u; alpha, dt, dx, steps)
    kernels = make_gpu_kernels(u)
    backend = first(kernels)
    current = u
    next = similar(u)
    du = similar(u)

    for _ in 1:steps
        synchronize_halo!(current)
        KA.synchronize(backend)
        heat_step_gpu!(kernels, next, du, current, alpha, dt, dx)
        current, next = next, current
    end

    synchronize_halo!(current)
    KA.synchronize(backend)
    return current
end

function run_local_cpu_gpu_heat_2d(; n=(128, 128), alpha=1.0f0, cfl=0.25f0, steps=100)
    halo = 1
    dx = (1.0f0 / n[1], 1.0f0 / n[2])
    dt = stable_heat_dt(alpha, cfl, dx)

    cpu0 = LocalHaloArray(Float32, n, halo; boundary_condition=:periodic)
    fill_gaussian!(cpu0)

    gpu0 = LocalHaloArray(Metal.MtlArray(parent(cpu0)), halo, :periodic)

    cpu = solve_cpu!(copy(cpu0); alpha, dt, dx, steps)
    gpu = solve_gpu!(gpu0; alpha, dt, dx, steps)

    cpu_interior = Array(interior_view(cpu))
    gpu_interior = Array(interior_view(gpu))
    error = maximum(abs.(Float64.(cpu_interior) .- Float64.(gpu_interior)))

    return (; cpu, gpu, dt, error)
end

host_interior_mean(u) = sum(Float64, Array(interior_view(u))) / length(u)

function main()
    result = run_local_cpu_gpu_heat_2d()
    @printf("Local CPU/GPU heat solve completed\n")
    @printf("  size:      %s\n", string(size(result.cpu)))
    @printf("  dt:        %.6e\n", result.dt)
    @printf("  CPU mean:  %.8f\n", host_interior_mean(result.cpu))
    @printf("  GPU mean:  %.8f\n", host_interior_mean(result.gpu))
    @printf("  max error: %.6e\n", result.error)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
