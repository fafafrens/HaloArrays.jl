using HaloArrays
using KernelAbstractions
using Metal
using Printf

const KA = KernelAbstractions

# One KernelAbstractions implementation, run on both the CPU and the Metal GPU
# backend (chosen from the array type via `KA.get_backend`), then compared. The
# kernels do all the work through HaloArrays' CellWindow/FaceCheckerboard
# helpers, so the same code is correct on either device — no separate hand-written
# CPU loops.

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

# --- portable kernels (NB: @index hoisted to its own line so the CPU backend's
# index-injection transform fires; nesting it inside cell_index(...) compiles for
# GPU but errors on CPU) ---

@kernel function zero_owned_kernel!(data, region::CellWindow{2})
    J = @index(Global, NTuple)
    I = cell_index(region, J)
    @inbounds data[I...] = zero(eltype(data))
end

@kernel function heat_flux_kernel!(du, data, region::FaceCheckerboard{2}, scale)
    J = @index(Global, NTuple)
    IL = cell_index(region, J)          # lower cell of this face
    offset = Tuple(region.offset)
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

@kernel function apply_heat_update_kernel!(out, data, du, dt, region::CellWindow{2})
    J = @index(Global, NTuple)
    I = cell_index(region, J)
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

function heat_step!(kernels, u_next, du, u, alpha, dt, dx)
    backend, zero!, flux!, update! = kernels
    cell_region = interior_cell_window(CellRanges(u))
    ranges = FaceRanges(u)

    launch_cell_kernel!(zero!, parent(du), cell_region)
    KA.synchronize(backend)

    for dim in 1:2
        scale = Float32(alpha / dx[dim]^2)
        for color in 0:1
            for region in (
                    left_face_window(ranges, dim, color),
                    internal_face_window(ranges, dim, color),
                    right_face_window(ranges, dim, color),
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

# Instantiate the kernels for the array's backend (CPU() for a plain Array,
# MetalBackend() for an MtlArray, etc.).
function make_kernels(u)
    backend = KA.get_backend(parent(u))
    return (
        backend,
        zero_owned_kernel!(backend),
        heat_flux_kernel!(backend),
        apply_heat_update_kernel!(backend),
    )
end

# Backend-agnostic solve: works for a CPU- or GPU-resident HaloArray alike.
function solve!(u; alpha, dt, dx, steps)
    kernels = make_kernels(u)
    backend = first(kernels)
    current = u
    next = similar(u)
    du = similar(u)

    for _ in 1:steps
        synchronize_halo!(current)
        KA.synchronize(backend)
        heat_step!(kernels, next, du, current, alpha, dt, dx)
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

    # same `solve!` on both — CPU backend vs Metal backend
    cpu = solve!(copy(cpu0); alpha, dt, dx, steps)
    gpu = solve!(gpu0; alpha, dt, dx, steps)

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
