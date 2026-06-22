# ============================================================
# HaloArrays.jl — GPU tutorial  (Metal / KernelAbstractions)
#
# Requirements:
#   import Pkg; Pkg.add(["Metal", "KernelAbstractions"])
#
# Run with:
#   julia --project=examples -t 4 examples/tutorials/gpu.jl
#
# The approach transfers to any KernelAbstractions-compatible
# backend (CUDA, ROCm, oneAPI) by swapping Metal for the
# corresponding package and replacing Metal.MtlArray with
# CuArray / ROCArray / etc.
#
# Sections:
#   1. Moving a HaloArray to the GPU
#   2. Writing a KernelAbstractions kernel on halo storage
#   3. CellWindow — mapping launch indices to storage
#   4. CellCheckerboard — checkerboard sweeps
#   5. FaceWindow — finite-volume flux kernels
#   6. Heat equation on the GPU (2-D)
# ============================================================

using HaloArrays
using KernelAbstractions
using Metal              # swap for CUDA / AMDGPU / oneAPI as needed
using Printf

const KA = KernelAbstractions

# ============================================================
# 1. MOVING A HaloArray TO THE GPU
# ============================================================
#
# LocalHaloArray stores data in any AbstractArray.  To put it on
# the GPU, create the array on CPU, fill initial data, then wrap
# the GPU-transferred storage in a new LocalHaloArray:
#
#   LocalHaloArray(gpu_storage, halo_width, boundary_condition)
#
# The halo width and boundary condition carry over; only the
# backing storage changes.

println("=" ^ 60)
println("Section 1 — Moving a HaloArray to the GPU")
println("=" ^ 60)

nx, ny = 64, 64
halo   = 1

# 1. Create on CPU and fill initial data
phi_cpu = LocalHaloArray(Float32, (nx, ny), halo; boundary_condition=:periodic)
interior_view(phi_cpu) .= 1.0f0
synchronize_halo!(phi_cpu)

# 2. Transfer full halo-padded storage to Metal
phi_gpu = LocalHaloArray(Metal.MtlArray(parent(phi_cpu)), halo, :periodic)
synchronize_halo!(phi_gpu)

println("CPU backend : ", typeof(parent(phi_cpu)))
println("GPU backend : ", typeof(parent(phi_gpu)))
println("storage_size: ", storage_size(phi_gpu))    # same as CPU version
println("halo_width  : ", halo_width(phi_gpu))

# synchronize_halo! on a GPU-backed LocalHaloArray fills ghost cells
# entirely on the GPU — no CPU round-trip.

# ============================================================
# 2. WRITING A KernelAbstractions KERNEL ON HALO STORAGE
# ============================================================
#
# GPU kernels receive parent(u) — the raw halo-padded storage.
# interior_range(u) gives the CartesianIndices of interior cells;
# ghost cells live outside that range but are still valid memory,
# readable for stencil access.
#
# Launch size is the interior-cell size, not the storage size, so
# the kernel maps its global thread index I directly to an interior
# storage index by offsetting by halo_width.

println()
println("=" ^ 60)
println("Section 2 — Basic kernel on halo storage")
println("=" ^ 60)

@kernel function laplacian_kernel!(du, u, inv_dx2, inv_dy2)
    i, j = @index(Global, NTuple)
    # Shift into storage coordinates (halo offset = 1)
    h = 1
    si = i + h
    sj = j + h
    @inbounds du[si, sj] =
        (u[si+1, sj] - 2*u[si, sj] + u[si-1, sj]) * inv_dx2 +
        (u[si, sj+1] - 2*u[si, sj] + u[si, sj-1]) * inv_dy2
end

u_gpu  = phi_gpu
du_gpu = similar(u_gpu)
synchronize_halo!(u_gpu)

backend    = KA.get_backend(parent(u_gpu))
kernel!    = laplacian_kernel!(backend, (16, 16))
launch_size = interior_size(u_gpu)         # (64, 64)
inv_dx2    = Float32((nx)^2)
inv_dy2    = Float32((ny)^2)

kernel!(parent(du_gpu), parent(u_gpu), inv_dx2, inv_dy2; ndrange=launch_size)
KA.synchronize(backend)

du_cpu = Array(interior_view(du_gpu))
println("max |Δu|  : ", maximum(abs, du_cpu))   # near 0 for constant field

# ============================================================
# 3. CellWindow — MAPPING LAUNCH INDICES TO STORAGE
# ============================================================
#
# Manually adding the halo offset in every kernel is error-prone.
# CellWindow encapsulates the mapping from compact launch
# coordinates (1-based, interior cells only) to storage coordinates
# (1-based, includes ghost padding).
#
# Workflow:
#   1.  region = interior_cell_window(CellRanges(u))
#   2.  Launch with ndrange = region.size
#   3.  Inside kernel: I = cell_index(region, J)
#                      where J = @index(Global, NTuple)
#   4.  is_cell_index_inbounds(region, I) guards boundary threads.

println()
println("=" ^ 60)
println("Section 3 — CellWindow")
println("=" ^ 60)

@kernel function fill_index_kernel!(data, region::CellWindow{2})
    J = @index(Global, NTuple)
    I = cell_index(region, J)
    if is_cell_index_inbounds(region, I)
        @inbounds data[I[1], I[2]] = Float32(I[1] * 100 + I[2])
    end
end

region = interior_cell_window(CellRanges(u_gpu))
println("launch size (interior)  : ", region.size)
println("first interior cell     : ", region.first)   # storage coords, = (halo+1, halo+1)

kernel2! = fill_index_kernel!(backend, (16, 16))
kernel2!(parent(u_gpu), region; ndrange=region.size)
KA.synchronize(backend)

corner_val = Array(parent(u_gpu))[halo+1, halo+1]
println("storage[2,2] (J=1,1) : ", corner_val)   # I=(2,2) → 2*100+2 = 202.0

# ============================================================
# 4. CellCheckerboard — CHECKERBOARD SWEEPS
# ============================================================
#
# For in-place Gauss-Seidel or Metropolis updates where a cell
# reads its own updated neighbours, you must split the interior cells
# into two non-adjacent subsets (checkerboard coloring).
# Cells of the same color are independent and can be updated in
# one parallel kernel launch.
#
# interior_cell_window(ranges, color; compressed_dim)
#   compressed_dim — the spatial dimension along which the launch
#                    grid is compressed.  Choosing the fastest-
#                    varying memory dimension gives coalesced access.

println()
println("=" ^ 60)
println("Section 4 — CellCheckerboard (checkerboard)")
println("=" ^ 60)

@kernel function checkerboard_kernel!(data, region::CellCheckerboard{2})
    J = @index(Global, NTuple)
    I = cell_index(region, J)
    if is_cell_index_inbounds(region, I)
        # Read 4 neighbours (already updated opposite-color cells
        # are safe because they are not in this launch)
        @inbounds begin
            nb_sum = data[I[1]+1, I[2]] + data[I[1]-1, I[2]] +
                     data[I[1], I[2]+1] + data[I[1], I[2]-1]
            data[I[1], I[2]] = nb_sum / 4.0f0
        end
    end
end

ranges    = CellRanges(u_gpu)
kern_cb!  = checkerboard_kernel!(backend, (16, 16))

for color in 0:1
    synchronize_halo!(u_gpu)
    KA.synchronize(backend)

    region_c = interior_cell_window(ranges, color; compressed_dim=2)
    println("color=$color  launch size : ", region_c.size)

    any(==(0), region_c.size) && continue

    kern_cb!(parent(u_gpu), region_c; ndrange=region_c.size)
    KA.synchronize(backend)
end

# ============================================================
# 5. FaceWindow — FINITE-VOLUME FLUX KERNELS
# ============================================================
#
# Finite-volume loops iterate over faces.  FaceWindow maps
# a compact 1-D (per face type) launch index to a storage index.
#
# Three region types per dimension:
#   left_face_window(fr, dim)     — ghost | interior boundary faces
#   internal_face_window(fr, dim) — interior | interior internal faces
#   right_face_window(fr, dim)    — interior | ghost boundary faces
#
# Inside the kernel use cell_index(region, J) to get IL (lower cell).
# The upper cell is IL + region.offset.

println()
println("=" ^ 60)
println("Section 5 — FaceWindow")
println("=" ^ 60)

@kernel function flux_accumulate_kernel!(du, u, region::FaceWindow{2}, inv_dx)
    J  = @index(Global, NTuple)
    IL = cell_index(region, J)
    if is_cell_index_inbounds(region, IL)
        IR = IL + region.offset
        @inbounds begin
            ul = u[IL[1], IL[2]]
            ur = u[IR[1], IR[2]]
            flux = 0.5f0 * (ul + ur) * inv_dx   # simple central flux
            # face contributes to both adjacent cells
            # (boundary face kernels only update the interior side — see below)
            du[IL[1], IL[2]] -= flux
            du[IR[1], IR[2]] += flux
        end
    end
end

fr = FaceRanges(u_gpu)
dim = 1
left_region     = left_face_window(fr, dim)
internal_region = internal_face_window(fr, dim)
right_region    = right_face_window(fr, dim)

println("dim=1  left    launch size : ", left_region.size)
println("dim=1  internal launch size: ", internal_region.size)
println("dim=1  right   launch size : ", right_region.size)

# ============================================================
# 6. HEAT EQUATION ON THE GPU (2-D)
# ============================================================
#
# We put everything together: GPU-backed LocalHaloArray, a
# KernelAbstractions kernel using CellWindow, and explicit
# Euler time integration.

println()
println("=" ^ 60)
println("Section 6 — Heat equation on the GPU (2-D)")
println("=" ^ 60)

@kernel function heat_rhs_kernel!(du, u, region::CellWindow{2}, inv_dx2, inv_dy2)
    J = @index(Global, NTuple)
    I = cell_index(region, J)
    if is_cell_index_inbounds(region, I)
        @inbounds du[I[1], I[2]] =
            (u[I[1]+1, I[2]] - 2*u[I[1], I[2]] + u[I[1]-1, I[2]]) * inv_dx2 +
            (u[I[1], I[2]+1] - 2*u[I[1], I[2]] + u[I[1], I[2]-1]) * inv_dy2
    end
end

@kernel function euler_update_kernel!(u_next, u, du, region::CellWindow{2}, dt)
    J = @index(Global, NTuple)
    I = cell_index(region, J)
    if is_cell_index_inbounds(region, I)
        @inbounds u_next[I[1], I[2]] = u[I[1], I[2]] + dt * du[I[1], I[2]]
    end
end

function run_heat_gpu(; n=(128,128), alpha=1.0f0, nt=200, cfl=0.4f0, groupsize=(16,16))
    h = 1

    # Build on CPU, transfer to GPU
    cpu = LocalHaloArray(Float32, n, h; boundary_condition=:periodic)
    fill_from_global_indices!(cpu) do I
        cx, cy = (n[1]+1)/2f0, (n[2]+1)/2f0
        r2 = ((I[1]-cx)/(n[1]/8f0))^2 + ((I[2]-cy)/(n[2]/8f0))^2
        return 1.0f0 + exp(-r2)
    end
    synchronize_halo!(cpu)

    u     = LocalHaloArray(Metal.MtlArray(parent(cpu)), h, :periodic)
    u_nxt = LocalHaloArray(Metal.MtlArray(similar(parent(cpu))), h, :periodic)
    du    = LocalHaloArray(Metal.MtlArray(similar(parent(cpu))), h, :periodic)

    synchronize_halo!(u)

    bk      = KA.get_backend(parent(u))
    dx      = 1.0f0 / n[1]
    dy      = 1.0f0 / n[2]
    dt      = cfl / (alpha * (inv(dx^2) + inv(dy^2)))
    inv_dx2 = alpha / dx^2
    inv_dy2 = alpha / dy^2

    cr      = CellRanges(u)
    region  = interior_cell_window(cr)

    rhs_k!  = heat_rhs_kernel!(bk, groupsize)
    step_k! = euler_update_kernel!(bk, groupsize)

    u0max = maximum(Array(interior_view(u)))

    cur, nxt = u, u_nxt
    for _ in 1:nt
        synchronize_halo!(cur)
        KA.synchronize(bk)

        rhs_k!(parent(du), parent(cur), region, inv_dx2, inv_dy2; ndrange=region.size)
        step_k!(parent(nxt), parent(cur), parent(du), region, dt; ndrange=region.size)
        KA.synchronize(bk)

        cur, nxt = nxt, cur
    end
    synchronize_halo!(cur)

    # Bring a slice back to CPU for inspection
    result_cpu = Array(interior_view(cur))
    umax = maximum(result_cpu)

    @printf("  GPU heat: n=%dx%d  dt=%.2e  nt=%d  max(u₀)=%.4f  max(u)=%.4f\n",
        n..., dt, nt, u0max, umax)
    return cur
end

run_heat_gpu()

println()
println("GPU tutorial complete.")
