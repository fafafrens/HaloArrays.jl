# ============================================================
# HaloArrays.jl — Broadcast tutorial
#
# Run with:
#   julia --project -t 4 tutorial_broadcast.jl
#
# Sections:
#   1. What broadcast does (and what it does NOT touch)
#   2. In-place broadcast (.=, .+=, etc.)
#   3. Allocating broadcast (out-of-place)
#   4. Mixing halo arrays with scalars and plain arrays
#   5. MultiHaloArray broadcast (per-field dispatch)
#   6. ThreadedHaloArray broadcast (per-tile dispatch)
#   7. What is intentionally NOT supported
# ============================================================

using HaloArrays
using OhMyThreads: tforeach

println("=" ^ 60)
println("HaloArrays.jl — Broadcast tutorial")
println("=" ^ 60)

# ============================================================
# 1. WHAT BROADCAST DOES (AND WHAT IT DOES NOT TOUCH)
# ============================================================
#
# The fundamental rule:
#
#   broadcast on a HaloArray operates ONLY on interior (owned)
#   cells.  Ghost cells are NEVER written by broadcast.
#
# This is intentional: ghost cells have a well-defined owner
# (a neighbouring rank or a boundary condition).  Writing them
# through broadcast would violate that contract.
#
# The invariant after any broadcast expression:
#   • interior cells hold the new computed value
#   • ghost cells are unchanged (stale until the next synchronize_halo!)
#
# Illustration:

println()
println("Section 1 — Interior-only semantics")
println("-" ^ 40)

u = LocalHaloArray(Float64, (4,), 1; boundary_condition=:periodic)
interior_view(u) .= [1.0, 2.0, 3.0, 4.0]
synchronize_halo!(u)   # ghost cells now hold 4.0 (left) and 1.0 (right)

println("before broadcast — parent : ", collect(parent(u)))
# [4.0 | 1.0 | 2.0 | 3.0 | 4.0 | 1.0]  (ghosts = wrapped values)

u .= 0.0               # broadcast: zeros ONLY the interior

println("after  u .= 0   — parent : ", collect(parent(u)))
# ghost cells are untouched: [4.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0]

println("interior         : ", collect(interior_view(u)))
# [0.0, 0.0, 0.0, 0.0]   (clean)

# ============================================================
# 2. IN-PLACE BROADCAST
# ============================================================
#
# In-place operators (.=  .+=  .-=  .*=  …) write into the
# destination's interior, never allocating a new halo array.
#
# Both sides of the expression are "unpacked" to interior_view
# before the operation is dispatched to Julia's ordinary broadcast
# machinery.

println()
println("Section 2 — In-place broadcast")
println("-" ^ 40)

u = LocalHaloArray(Float64, (4,), 1; boundary_condition=:periodic)
v = LocalHaloArray(Float64, (4,), 1; boundary_condition=:periodic)

interior_view(u) .= [1.0, 2.0, 3.0, 4.0]
interior_view(v) .= [10.0, 20.0, 30.0, 40.0]

# Element-wise addition into u
u .+= v
println("u .+= v → ", collect(interior_view(u)))   # [11, 22, 33, 44]

# Scale and shift
u .= 2.0 .* u .- 1.0
println("2u - 1  → ", collect(interior_view(u)))   # [21, 43, 65, 87]

# Any Julia function works element-wise
u .= sqrt.(abs.(u))
println("sqrt|u| → ", collect(interior_view(u)))

# ============================================================
# 3. ALLOCATING BROADCAST (OUT-OF-PLACE)
# ============================================================
#
# Out-of-place broadcast creates a NEW halo array whose type,
# halo width, and boundary condition are copied from the first
# halo array found in the expression (via Base.similar).
#
# The result holds the broadcast output in its interior; ghost
# cells are uninitialised and should be filled with
# synchronize_halo! before any stencil read.

println()
println("Section 3 — Allocating broadcast")
println("-" ^ 40)

a = LocalHaloArray(Float64, (4,), 1; boundary_condition=:periodic)
b = LocalHaloArray(Float64, (4,), 1; boundary_condition=:periodic)

interior_view(a) .= [1.0, 2.0, 3.0, 4.0]
interior_view(b) .= [10.0, 10.0, 10.0, 10.0]

c = a .+ b                     # allocates a new LocalHaloArray
println("typeof(c)       : ", typeof(c))
println("a .+ b interior : ", collect(interior_view(c)))   # [11, 12, 13, 14]
println("halo_width(c)   : ", halo_width(c))               # same as a (= 1)

# Ghost cells of c are uninitialised — fill them before any stencil:
synchronize_halo!(c)

# Chained out-of-place expression
d = a .* b .+ c
println("a*b+c interior  : ", collect(interior_view(d)))

# ============================================================
# 4. MIXING HALO ARRAYS WITH SCALARS AND PLAIN ARRAYS
# ============================================================
#
# Scalars broadcast freely alongside halo arrays — they are
# treated as zero-dimensional arrays by Julia's broadcast.
#
# Plain Julia Arrays of the same size as interior_view(u) can
# be mixed in when the broadcast is IN-PLACE (the destination
# is a halo array).  The plain array is used as-is without
# ghost-cell padding.

println()
println("Section 4 — Mixing with scalars and plain arrays")
println("-" ^ 40)

u = LocalHaloArray(Float64, (4,), 1; boundary_condition=:periodic)
interior_view(u) .= [1.0, 2.0, 3.0, 4.0]

# Scalar on the right
u .= u .* 3.0
println("u * 3.0    : ", collect(interior_view(u)))   # [3, 6, 9, 12]

# Plain vector the same size as the interior
mask = [1.0, 0.0, 1.0, 0.0]
u .= u .* mask
println("u .* mask  : ", collect(interior_view(u)))   # [3, 0, 9, 0]

# Out-of-place with a scalar produces a new halo array
w = u .+ 100.0
println("u .+ 100   : ", collect(interior_view(w)))   # [103, 100, 109, 100]

# ============================================================
# 5. MultiHaloArray BROADCAST
# ============================================================
#
# Broadcast on a MultiHaloArray (or LocalMultiHaloArray) applies
# the operation field-by-field.  Each field's interior is updated
# independently.  The field structure is preserved in the result.
#
# Rules:
#   • A scalar broadcasts to every field.
#   • A MultiHaloArray on the right must have the same field names.
#   • You cannot mix a MultiHaloArray with a plain HaloArray in a
#     single broadcast — use per-field access for mixed updates.

println()
println("Section 5 — MultiHaloArray broadcast")
println("-" ^ 40)

state = LocalMultiHaloArray(Float64, (4,), 1;
    boundary_conditions=(
        rho = ((Repeating(), Repeating()),),
        vel = ((Reflecting(), Reflecting()),),
    ))

interior_view(state.rho) .= [1.0, 2.0, 3.0, 4.0]
interior_view(state.vel) .= [0.1, 0.2, 0.3, 0.4]

# Scale all fields at once
state .*= 2.0
println("rho after *=2 : ", collect(interior_view(state.rho)))   # [2, 4, 6, 8]
println("vel after *=2 : ", collect(interior_view(state.vel)))   # [0.2, 0.4, 0.6, 0.8]

# Add two MultiHaloArrays (same field names)
state2 = LocalMultiHaloArray(Float64, (4,), 1;
    boundary_conditions=(
        rho = ((Repeating(), Repeating()),),
        vel = ((Reflecting(), Reflecting()),),
    ))
interior_view(state2.rho) .= 10.0
interior_view(state2.vel) .= -1.0

state .+= state2
println("rho after +state2 : ", collect(interior_view(state.rho)))
println("vel after +state2 : ", collect(interior_view(state.vel)))

# Out-of-place creates a new MultiHaloArray preserving field structure
result = state .* 0.5
println("typeof(result) : ", typeof(result))

# ============================================================
# 6. ThreadedHaloArray BROADCAST
# ============================================================
#
# ThreadedHaloArray uses a separate broadcast style
# (ThreadedHaloArrayStyle).  The broadcast is dispatched tile by
# tile, each tile's interior_view being updated concurrently via
# OhMyThreads.tforeach.
#
# From the user's perspective the behaviour is identical to
# LocalHaloArray — only the execution is parallel.

println()
println("Section 6 — ThreadedHaloArray broadcast")
println("-" ^ 40)

nthreads  = max(1, Threads.nthreads())
tile_size = (8,)
tile_dims = (nthreads,)

tu = ThreadedHaloArray(Float64, tile_size, 1; dims=tile_dims, boundary_condition=:periodic)
tv = ThreadedHaloArray(Float64, tile_size, 1; dims=tile_dims, boundary_condition=:periodic)

# Fill via broadcast (operates on every tile's interior in parallel)
tu .= 1.0
tv .= 2.0

tu .+= tv                      # parallel in-place addition
println("tu .+= tv → max : ", maximum(tu))   # should be 3.0

result_t = tu .* tv .+ 1.0    # parallel out-of-place
println("tu*tv+1 → max   : ", maximum(result_t))   # should be 7.0

# ThreadedMultiHaloArray broadcast behaves the same — per-field,
# per-tile in parallel:
tstate = ThreadedMultiHaloArray(Float64, tile_size, 1;
    dims=tile_dims,
    boundary_conditions=(
        a=((Repeating(), Repeating()),),
        b=((Repeating(), Repeating()),),
    ))
tstate.a .= 3.0
tstate.b .= 4.0
tstate .*= 2.0
println("tstate.a *=2 : ", maximum(tstate.a))   # 6.0
println("tstate.b *=2 : ", maximum(tstate.b))   # 8.0

# ============================================================
# 7. WHAT IS INTENTIONALLY NOT SUPPORTED
# ============================================================

println()
println("Section 7 — Unsupported patterns")
println("-" ^ 40)

# a) Mixing ThreadedHaloArray and (Local/MPI)HaloArray in one broadcast
#    → throws ArgumentError at style-resolution time (before any work)
println("Mixed-backend broadcast throws at style resolution:")
u_loc = LocalHaloArray(Float64, tile_size, 1; boundary_condition=:periodic)
u_thr = ThreadedHaloArray(Float64, tile_size, 1; dims=(1,), boundary_condition=:periodic)
try
    u_loc .= u_thr .+ 1.0
catch e
    println("  caught: ", e.msg)
end

# b) Broadcast does NOT refresh halos — you must call synchronize_halo!
#    yourself before any stencil that reads ghost cells.
println()
println("Correct stencil pattern (halo not implicit in broadcast):")
println("  1. interior_view(u) .= ...   # or u .= ...  (sets interior)")
println("  2. synchronize_halo!(u)       # fill ghost cells")
println("  3. stencil on parent(u)       # now safe to read ghosts")

println()
println("Broadcast tutorial complete.")
