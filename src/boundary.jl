using StaticArrays

@inline _slice_index(::Val{N}, dim, i) where {N} =
    ntuple(j -> j == dim ? (i:i) : Colon(), Val(N))

function boundary_condition!(halo::HaloArray{T,N,A,Halo,B,BCondition}, s::Side{side},dim::Dim{d}) where {
    T,N,A,Halo,B,BCondition,side, d}
    
    mode = halo.boundary_condition[d][side]
    nbrank = halo.topology.neighbors[d][side]
    if nbrank < 0  # MPI.PROC_NULL == -1: no neighbour → physical boundary
        boundary_condition!(halo, s, dim, mode)
    end
    return nothing
end


function boundary_condition!(halo::HaloArray{T,N,A,Halo,B,BCondition},s::Side{1},d::Dim{dim}, mode::Reflecting) where {T,N,A,Halo,B,BCondition,dim}
    h = halo_width(halo)
    full = parent(halo)
    interior_region = interior_view(halo)
    halo_region = get_recv_view(s, d, full, h)

    for i in 1:size(halo_region, dim)

        src_i = h-i + 1
        mirror_idx = _slice_index(Val(N), dim, src_i)
        dst_idx = _slice_index(Val(N), dim, i)
    
        @views halo_region[dst_idx...] .= interior_region[mirror_idx...]
    end
    return nothing
end

function boundary_condition!(halo::HaloArray{T,N,A,Halo,B,BCondition},s::Side{2}, d::Dim{dim}, mode::Reflecting) where {T,N,A,Halo,B,BCondition,dim}
    h = halo_width(halo)
    full = parent(halo)
    interior_region = interior_view(halo)
    halo_region = get_recv_view(s, d, full, h)
    n = size(interior_region, dim)
    for i in 1:size(halo_region, dim)
        src_i = n - (i - 1)
        mirror_idx = _slice_index(Val(N), dim, src_i)
        dst_idx = _slice_index(Val(N), dim, i)
        @views halo_region[dst_idx...] .= interior_region[mirror_idx...]
    end
    return nothing
end



function boundary_condition!(halo::HaloArray{T,N,A,Halo,B,BCondition}, s::Side{1},d::Dim{dim}, mode::Antireflecting) where {T,N,A,Halo,B,BCondition,dim}
    h = halo_width(halo)
    full = parent(halo)
    interior_region = interior_view(halo)
    halo_region = get_recv_view(s, d, full, h)
    for i in 1:size(halo_region, dim)
        src_i = h - i + 1
        mirror_idx = _slice_index(Val(N), dim, src_i)
        dst_idx = _slice_index(Val(N), dim, i)
        @views halo_region[dst_idx...] .= .- interior_region[mirror_idx...]
    end
    return nothing
end

function boundary_condition!(halo::HaloArray{T,N,A,Halo,B,BCondition},s::Side{2}, d::Dim{dim}, mode::Antireflecting) where {T,N,A,Halo,B,BCondition,dim}
    h = halo_width(halo)
    full = parent(halo)
    interior_region = interior_view(halo)
    halo_region = get_recv_view(s, d, full, h)
    n = size(interior_region, dim)
    for i in 1:size(halo_region, dim)
        src_i = n - (i - 1)
        mirror_idx = _slice_index(Val(N), dim, src_i)
        dst_idx = _slice_index(Val(N), dim, i)
        @views halo_region[dst_idx...] .= .- interior_region[mirror_idx...]
    end
    return nothing
end



function boundary_condition!(halo::HaloArray{T,N,A,Halo,B,BCondition},s::Side{1}, d::Dim{dim}, mode::Repeating) where {T,N,A,Halo,B,BCondition,dim}
    h = halo_width(halo)
    full = parent(halo)
    interior_region = interior_view(halo)
    halo_region = get_recv_view(s, d, full, h)
    edge_idx = _slice_index(Val(N), dim, 1)
    boundary_slice = @view interior_region[edge_idx...]
    for i in 1:size(halo_region, dim)
        halo_idx = _slice_index(Val(N), dim, i)
        @views halo_region[halo_idx...] .= boundary_slice
    end
    return nothing
end

function boundary_condition!(halo::HaloArray{T,N,A,Halo,B,BCondition},s::Side{2}, d::Dim{dim}, mode::Repeating) where {T,N,A,Halo,B,BCondition,dim}
    h = halo_width(halo)
    full = parent(halo)
    interior_region = interior_view(halo)
    halo_region = get_recv_view(s, d, full, h)
    edge_idx = _slice_index(Val(N), dim, size(interior_region, dim))
    boundary_slice = @view interior_region[edge_idx...]
    for i in 1:size(halo_region, dim)
        halo_idx = _slice_index(Val(N), dim, i)
        @views halo_region[halo_idx...] .= boundary_slice
    end
    return nothing
end

function boundary_condition!(halo::HaloArray{T,N,A,Halo,B,BCondition}, s::Side{1},d::Dim{dim}, mode::Periodic) where {T,N,A,Halo,B,BCondition,dim}
    return nothing
end

function boundary_condition!(halo::HaloArray{T,N,A,Halo,B,BCondition}, s::Side{2},d::Dim{dim}, mode::Periodic) where {T,N,A,Halo,B,BCondition,dim}
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

function boundary_condition!(halo::LocalHaloArray, s::Side{side}, dim::Dim{d}) where {side,d}
    mode = halo.boundary_condition[d][side]
    boundary_condition!(halo, s, dim, mode)
    return nothing
end

function boundary_condition!(halo::LocalHaloArray, s::Side{1}, d::Dim{dim}, mode::Reflecting) where {dim}
    h = halo_width(halo)
    interior_region = interior_view(halo)
    halo_region = get_recv_view(s, d, halo)

    for i in 1:size(halo_region, dim)
        src_i = h - i + 1
        mirror_idx = _slice_index(Val(ndims(halo)), dim, src_i)
        dst_idx = _slice_index(Val(ndims(halo)), dim, i)
        @views halo_region[dst_idx...] .= interior_region[mirror_idx...]
    end
    return nothing
end

function boundary_condition!(halo::LocalHaloArray, s::Side{2}, d::Dim{dim}, mode::Reflecting) where {dim}
    interior_region = interior_view(halo)
    halo_region = get_recv_view(s, d, halo)
    n = size(interior_region, dim)

    for i in 1:size(halo_region, dim)
        src_i = n - (i - 1)
        mirror_idx = _slice_index(Val(ndims(halo)), dim, src_i)
        dst_idx = _slice_index(Val(ndims(halo)), dim, i)
        @views halo_region[dst_idx...] .= interior_region[mirror_idx...]
    end
    return nothing
end

function boundary_condition!(halo::LocalHaloArray, s::Side{1}, d::Dim{dim}, mode::Antireflecting) where {dim}
    h = halo_width(halo)
    interior_region = interior_view(halo)
    halo_region = get_recv_view(s, d, halo)

    for i in 1:size(halo_region, dim)
        src_i = h - i + 1
        mirror_idx = _slice_index(Val(ndims(halo)), dim, src_i)
        dst_idx = _slice_index(Val(ndims(halo)), dim, i)
        @views halo_region[dst_idx...] .= .- interior_region[mirror_idx...]
    end
    return nothing
end

function boundary_condition!(halo::LocalHaloArray, s::Side{2}, d::Dim{dim}, mode::Antireflecting) where {dim}
    interior_region = interior_view(halo)
    halo_region = get_recv_view(s, d, halo)
    n = size(interior_region, dim)

    for i in 1:size(halo_region, dim)
        src_i = n - (i - 1)
        mirror_idx = _slice_index(Val(ndims(halo)), dim, src_i)
        dst_idx = _slice_index(Val(ndims(halo)), dim, i)
        @views halo_region[dst_idx...] .= .- interior_region[mirror_idx...]
    end
    return nothing
end

function boundary_condition!(halo::LocalHaloArray, s::Side{1}, d::Dim{dim}, mode::Repeating) where {dim}
    interior_region = interior_view(halo)
    halo_region = get_recv_view(s, d, halo)
    edge_idx = _slice_index(Val(ndims(halo)), dim, 1)
    boundary_slice = @view interior_region[edge_idx...]

    for i in 1:size(halo_region, dim)
        halo_idx = _slice_index(Val(ndims(halo)), dim, i)
        @views halo_region[halo_idx...] .= boundary_slice
    end
    return nothing
end

function boundary_condition!(halo::LocalHaloArray, s::Side{2}, d::Dim{dim}, mode::Repeating) where {dim}
    interior_region = interior_view(halo)
    halo_region = get_recv_view(s, d, halo)
    edge_idx = _slice_index(Val(ndims(halo)), dim, size(interior_region, dim))
    boundary_slice = @view interior_region[edge_idx...]

    for i in 1:size(halo_region, dim)
        halo_idx = _slice_index(Val(ndims(halo)), dim, i)
        @views halo_region[halo_idx...] .= boundary_slice
    end
    return nothing
end

function boundary_condition!(halo::LocalHaloArray, s::Side{1}, d::Dim{dim}, mode::Periodic) where {dim}
    interior_region = interior_view(halo)
    halo_region = get_recv_view(s, d, halo)
    n = size(interior_region, dim)
    h = halo_width(halo)

    for i in 1:size(halo_region, dim)
        src_i = n - h + i
        src_idx = _slice_index(Val(ndims(halo)), dim, src_i)
        dst_idx = _slice_index(Val(ndims(halo)), dim, i)
        @views halo_region[dst_idx...] .= interior_region[src_idx...]
    end
    return nothing
end

function boundary_condition!(halo::LocalHaloArray, s::Side{2}, d::Dim{dim}, mode::Periodic) where {dim}
    interior_region = interior_view(halo)
    halo_region = get_recv_view(s, d, halo)

    for i in 1:size(halo_region, dim)
        src_idx = _slice_index(Val(ndims(halo)), dim, i)
        dst_idx = _slice_index(Val(ndims(halo)), dim, i)
        @views halo_region[dst_idx...] .= interior_region[src_idx...]
    end
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

function boundary_condition!(mha::MultiHaloArray{T,N,A}) where {T,N,A}
    foreach_field!(boundary_condition!,mha)
    return nothing
    
end

function boundary_condition!(mha::ArrayOfHaloArray)
    foreach_field!(boundary_condition!, mha)
    return nothing
end

function to_bc(x)
    if x isa Symbol
        if x == :reflecting
            return Reflecting()
        elseif x == :antireflecting
            return Antireflecting()
        elseif x == :repeating
            return Repeating()
        elseif x == :periodic
            return Periodic()
        else
            throw(ArgumentError("Unknown boundary condition symbol: $x"))
        end
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
        if length(bc) == N
            return ntuple(i -> normalize_one_dim(bc[i]), N)
        else
            throw(ArgumentError("Boundary condition tuple length $(length(bc)) does not match expected dimension $N"))
        end
    else
        bc_concrete = to_bc(bc)
        return ntuple(_ -> (bc_concrete, bc_concrete), N)
    end
end

function isperiodic(bc::AbstractBoundaryCondition)
    return bc isa Periodic
end

function infer_periodicity(boundary_condition::NTuple{N,NTuple{2,AbstractBoundaryCondition}}) where {N}
    ntuple(i -> isperiodic(boundary_condition[i][1]) && isperiodic(boundary_condition[i][2]), Val(N))
end
