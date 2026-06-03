using StaticArrays

@inline _slice_index(::Val{N}, dim, i) where {N} =
    ntuple(j -> j == dim ? (i:i) : Colon(), Val(N))

# ============================================================
# HaloArray top-level dispatcher
# Only applies BC on faces with no MPI neighbour (physical boundary).
# ============================================================

function boundary_condition!(halo::HaloArray{T,N,A,Halo,B,BCondition},
        s::Side{side}, dim::Dim{d}) where {T,N,A,Halo,B,BCondition,side,d}
    mode   = halo.boundary_condition[d][side]
    nbrank = halo.topology.neighbors[d][side]
    if nbrank < 0   # MPI.PROC_NULL == -1 → physical boundary
        boundary_condition!(halo, s, dim, mode)
    end
    return nothing
end

function boundary_condition!(halo::HaloArray{T,N,A,Halo,B,BCondition}) where {
        T,N,A,Halo,B,BCondition}
    ntuple(Val(N)) do D
        ntuple(Val(2)) do S
            boundary_condition!(halo, Side(S), Dim(D))
        end
    end
    return nothing
end

# ============================================================
# LocalHaloArray top-level dispatcher
# ============================================================

function boundary_condition!(halo::LocalHaloArray, s::Side{side}, dim::Dim{d}) where {side,d}
    mode = halo.boundary_condition[d][side]
    boundary_condition!(halo, s, dim, mode)
    return nothing
end

function boundary_condition!(halo::LocalHaloArray{T,N}) where {T,N}
    ntuple(Val(N)) do D
        ntuple(Val(2)) do S
            boundary_condition!(halo, Side(S), Dim(D))
        end
    end
    return nothing
end

# ============================================================
# Shared mode implementations on AbstractSingleHaloArray
#
# ThreadedHaloArray dispatches all take tile_id as the second
# argument and will win over these fallbacks — no conflict.
# HaloArray and LocalHaloArray both have a 3-arg get_recv_view
# dispatch, so the body is identical for both types.
# ============================================================

# ---- Reflecting -----------------------------------------------

function boundary_condition!(halo::AbstractSingleHaloArray,
        s::Side{1}, d::Dim{dim}, ::Reflecting) where {dim}
    h = halo_width(halo)
    N = ndims(halo)
    interior_region = interior_view(halo)
    halo_region     = get_recv_view(s, d, halo)
    for i in 1:size(halo_region, dim)
        src_i = h - i + 1
        @views halo_region[_slice_index(Val(N), dim, i)...] .=
               interior_region[_slice_index(Val(N), dim, src_i)...]
    end
    return nothing
end

function boundary_condition!(halo::AbstractSingleHaloArray,
        s::Side{2}, d::Dim{dim}, ::Reflecting) where {dim}
    N = ndims(halo)
    n = size(interior_view(halo), dim)
    interior_region = interior_view(halo)
    halo_region     = get_recv_view(s, d, halo)
    for i in 1:size(halo_region, dim)
        src_i = n - (i - 1)
        @views halo_region[_slice_index(Val(N), dim, i)...] .=
               interior_region[_slice_index(Val(N), dim, src_i)...]
    end
    return nothing
end

# ---- Antireflecting -------------------------------------------

function boundary_condition!(halo::AbstractSingleHaloArray,
        s::Side{1}, d::Dim{dim}, ::Antireflecting) where {dim}
    h = halo_width(halo)
    N = ndims(halo)
    interior_region = interior_view(halo)
    halo_region     = get_recv_view(s, d, halo)
    for i in 1:size(halo_region, dim)
        src_i = h - i + 1
        @views halo_region[_slice_index(Val(N), dim, i)...] .=
               .- interior_region[_slice_index(Val(N), dim, src_i)...]
    end
    return nothing
end

function boundary_condition!(halo::AbstractSingleHaloArray,
        s::Side{2}, d::Dim{dim}, ::Antireflecting) where {dim}
    N = ndims(halo)
    n = size(interior_view(halo), dim)
    interior_region = interior_view(halo)
    halo_region     = get_recv_view(s, d, halo)
    for i in 1:size(halo_region, dim)
        src_i = n - (i - 1)
        @views halo_region[_slice_index(Val(N), dim, i)...] .=
               .- interior_region[_slice_index(Val(N), dim, src_i)...]
    end
    return nothing
end

# ---- Repeating ------------------------------------------------

function boundary_condition!(halo::AbstractSingleHaloArray,
        s::Side{1}, d::Dim{dim}, ::Repeating) where {dim}
    N = ndims(halo)
    interior_region = interior_view(halo)
    halo_region     = get_recv_view(s, d, halo)
    edge = @view interior_region[_slice_index(Val(N), dim, 1)...]
    for i in 1:size(halo_region, dim)
        @views halo_region[_slice_index(Val(N), dim, i)...] .= edge
    end
    return nothing
end

function boundary_condition!(halo::AbstractSingleHaloArray,
        s::Side{2}, d::Dim{dim}, ::Repeating) where {dim}
    N = ndims(halo)
    n = size(interior_view(halo), dim)
    interior_region = interior_view(halo)
    halo_region     = get_recv_view(s, d, halo)
    edge = @view interior_region[_slice_index(Val(N), dim, n)...]
    for i in 1:size(halo_region, dim)
        @views halo_region[_slice_index(Val(N), dim, i)...] .= edge
    end
    return nothing
end

# ---- Periodic -------------------------------------------------
# HaloArray: MPI exchange fills halos → no-op here.
# LocalHaloArray: must physically wrap the interior data.

boundary_condition!(::HaloArray, ::Side, ::Dim, ::Periodic) = nothing

function boundary_condition!(halo::LocalHaloArray, s::Side{1}, d::Dim{dim}, ::Periodic) where {dim}
    N = ndims(halo)
    h = halo_width(halo)
    interior_region = interior_view(halo)
    halo_region     = get_recv_view(s, d, halo)
    n = size(interior_region, dim)
    for i in 1:size(halo_region, dim)
        src_i = n - h + i
        @views halo_region[_slice_index(Val(N), dim, i)...] .=
               interior_region[_slice_index(Val(N), dim, src_i)...]
    end
    return nothing
end

function boundary_condition!(halo::LocalHaloArray, s::Side{2}, d::Dim{dim}, ::Periodic) where {dim}
    N = ndims(halo)
    interior_region = interior_view(halo)
    halo_region     = get_recv_view(s, d, halo)
    for i in 1:size(halo_region, dim)
        @views halo_region[_slice_index(Val(N), dim, i)...] .=
               interior_region[_slice_index(Val(N), dim, i)...]
    end
    return nothing
end

# ---- NoBoundaryCondition ----------------------------------------
# Ghost cells are left unchanged; the user fills them via a custom function.

boundary_condition!(::AbstractSingleHaloArray, ::Side, ::Dim, ::NoBoundaryCondition) = nothing
boundary_condition!(::ThreadedHaloArray, ::Integer, ::Side, ::Dim, ::NoBoundaryCondition) = nothing

# ============================================================
# Collection delegators
# ============================================================

function boundary_condition!(mha::MultiHaloArray{T,N,A}) where {T,N,A}
    foreach_field!(boundary_condition!, mha)
    return nothing
end

function boundary_condition!(mha::ArrayOfHaloArray)
    foreach_field!(boundary_condition!, mha)
    return nothing
end

# ============================================================
# Helpers: symbol/type → BC instance, normalization
# ============================================================

function to_bc(x)
    if x isa Symbol
        x == :reflecting       && return Reflecting()
        x == :antireflecting   && return Antireflecting()
        x == :repeating        && return Repeating()
        x == :periodic         && return Periodic()
        x == :noboundary       && return NoBoundaryCondition()
        throw(ArgumentError("Unknown boundary condition symbol: $x"))
    elseif x isa DataType && x <: AbstractBoundaryCondition
        return x()
    elseif x isa AbstractBoundaryCondition
        return x
    else
        throw(ArgumentError("Invalid boundary_condition: $x"))
    end
end

function normalize_one_dim(bc_dim)
    if bc_dim isa Tuple && length(bc_dim) == 2
        return (to_bc(bc_dim[1]), to_bc(bc_dim[2]))
    else
        return (to_bc(bc_dim), to_bc(bc_dim))
    end
end

function normalize_boundary_condition(bc, N::Int)
    if bc isa Tuple
        length(bc) == N ||
            throw(ArgumentError("Boundary condition tuple length $(length(bc)) != $N"))
        return ntuple(i -> normalize_one_dim(bc[i]), N)
    else
        bc_concrete = to_bc(bc)
        return ntuple(_ -> (bc_concrete, bc_concrete), N)
    end
end

function infer_periodicity(boundary_condition::NTuple{N,NTuple{2,AbstractBoundaryCondition}}) where {N}
    ntuple(i -> boundary_condition[i][1] isa Periodic && boundary_condition[i][2] isa Periodic, Val(N))
end
