

using StaticArrays

function boundary_condition!(halo::HaloArray{T,N,A,Halo,Size,B,C,BCondition}, s::Side{side},dim::Dim{d}) where {
    T,N,A,Halo,Size,B,C,BCondition,side, d}
    
    mode = halo.boundary_condition[d][side]
    # Check if this is a boundary
    nbrank = halo.topology.neighbors[d][side]
    if nbrank == MPI.PROC_NULL
        # Extract halo region and its mirror interior slice
        #halo_region = get_recv_view(s, dim, full, h)
        #interior_region = get_send_view(s, dim, full, h)

        # Dispatch based on boundary mode
        #@show "Applying boundary condition for side $s, dim $dim, mode $mode"
        boundary_condition!(halo, s, dim, mode)
    end
    return nothing
end


function boundary_condition!(halo::HaloArray{T,N,A,Halo,Size,B,C,BCondition},s::Side{1},d::Dim{dim}, mode::Reflecting) where {T,N,A,Halo,Size,B,C,BCondition,dim}
    h = halo_width(halo)
    full = parent(halo)
    interior_region = interior_view(halo)
    halo_region = get_recv_view(s, d, full, h)

    for i in 1:size(halo_region, dim)

        src_i = h-i + 1                        # reflect index: 1 → 1, 2 → 2, ...
        mirror_idx = ntuple(j -> j == dim ? src_i : Colon(), Val(N))
        dst_idx = ntuple(j -> j == dim ?  i  : Colon(), Val(N))
    
        @views halo_region[dst_idx...] .= interior_region[mirror_idx...]
    end
    return nothing
end

function boundary_condition!(halo::HaloArray{T,N,A,Halo,Size,B,C,BCondition},s::Side{2}, d::Dim{dim}, mode::Reflecting) where {T,N,A,Halo,Size,B,C,BCondition,dim}
    h = halo_width(halo)
    full = parent(halo)
    interior_region = interior_view(halo)
    halo_region = get_recv_view(s, d, full, h)
    n = size(interior_region, dim)
    for i in 1:size(halo_region, dim)
        src_i = n - (i - 1)
        mirror_idx = ntuple(j -> j == dim ? src_i : Colon(), Val(N))
        dst_idx = ntuple(j -> j == dim ? i : Colon(), Val(N))
        @views halo_region[dst_idx...] .= interior_region[mirror_idx...]
    end
    return nothing
end



function boundary_condition!(halo::HaloArray{T,N,A,Halo,Size,B,C,BCondition}, s::Side{1},d::Dim{dim}, mode::Antireflecting) where {T,N,A,Halo,Size,B,C,BCondition,dim}
    h = halo_width(halo)
    full = parent(halo)
    interior_region = interior_view(halo)
    halo_region = get_recv_view(s, d, full, h)
    for i in 1:size(halo_region, dim)
        src_i = h - i + 1                        # reflect index: 1 → 1, 2 → 2, ...
        mirror_idx = ntuple(j -> j == dim ? src_i : Colon(), Val(N))
        dst_idx = ntuple(j -> j == dim ? i : Colon(), Val(N))
        @views halo_region[dst_idx...] .= .- interior_region[mirror_idx...]
    end
    return nothing
end

function boundary_condition!(halo::HaloArray{T,N,A,Halo,Size,B,C,BCondition},s::Side{2}, d::Dim{dim}, mode::Antireflecting) where {T,N,A,Halo,Size,B,C,BCondition,dim}
    h = halo_width(halo)
    full = parent(halo)
    interior_region = interior_view(halo)
    halo_region = get_recv_view(s, d, full, h)
    n = size(interior_region, dim)
    for i in 1:size(halo_region, dim)
        src_i = n - h + i
        mirror_idx = ntuple(j -> j == dim ? src_i : Colon(), Val(N))
        dst_idx = ntuple(j -> j == dim ? i : Colon(), Val(N))
        @views halo_region[dst_idx...] .= .- interior_region[mirror_idx...]
    end
    return nothing
end



function boundary_condition!(halo::HaloArray{T,N,A,Halo,Size,B,C,BCondition},s::Side{1}, d::Dim{dim}, mode::Repeating) where {T,N,A,Halo,Size,B,C,BCondition,dim}
    h = halo_width(halo)
    full = parent(halo)
    interior_region = interior_view(halo)
    halo_region = get_recv_view(s, d, full, h)
    edge_idx = ntuple(j -> j == dim ? 1 : Colon(), Val(N))
    boundary_slice = @view interior_region[edge_idx...]
    for i in 1:size(halo_region, dim)
        halo_idx = ntuple(j -> j == dim ? i : Colon(), Val(N))
        @views halo_region[halo_idx...] .= boundary_slice
    end
    return nothing
end

function boundary_condition!(halo::HaloArray{T,N,A,Halo,Size,B,C,BCondition},s::Side{2}, d::Dim{dim}, mode::Repeating) where {T,N,A,Halo,Size,B,C,BCondition,dim}
    h = halo_width(halo)
    full = parent(halo)
    interior_region = interior_view(halo)
    halo_region = get_recv_view(s, d, full, h)
    edge_idx = ntuple(j -> j == dim ? size(interior_region, dim) : Colon(), Val(N))
    boundary_slice = @view interior_region[edge_idx...]
    for i in 1:size(halo_region, dim)
        halo_idx = ntuple(j -> j == dim ? i : Colon(), Val(N))
        @views halo_region[halo_idx...] .= boundary_slice
    end
    return nothing
end

function boundary_condition!(halo::HaloArray{T,N,A,Halo,Size,B,C,BCondition}, s::Side{1},d::Dim{dim}, mode::Periodic) where {T,N,A,Halo,Size,B,C,BCondition,dim}
    return nothing
end

function boundary_condition!(halo::HaloArray{T,N,A,Halo,Size,B,C,BCondition}, s::Side{2},d::Dim{dim}, mode::Periodic) where {T,N,A,Halo,Size,B,C,BCondition,dim}
    return nothing
end



function boundary_condition!(halo::HaloArray{T,N,A,Halo,Size,B,C,BCondition}) where {
    T,N,A,Halo,Size,B,C,BCondition}
    
    ntuple(Val(N)) do D
        ntuple(Val(2)) do S
            boundary_condition!(halo, Side(S), Dim(D))
        end
    end
    return nothing
end

function boundary_condition!(mha::MultiHaloArray{T,N,A,Len}) where {T,N,A,Len}
    # Dispatch to each field's boundary condition
    foreach_field!(boundary_condition!,mha)
    return nothing
    
end



function to_bc(symbol::Symbol)
    if symbol == :reflecting
        return Reflecting()
    elseif symbol == :antireflecting
        return Antireflecting()
    elseif symbol == :repeating
        return Repeating()
    elseif symbol == :periodic
        return Periodic()
    else
        throw(ArgumentError("""Unknown boundary condition symbol: $symbol you can implement your own boundary condition with the following interface:

        boundary_condition!(halo::HaloArray, d::Dim{dim}, s::Side{1}, mode::YourBoundaryConditionType)   
        boundary_condition!(halo::HaloArray, d::Dim{dim}, s::Side{2}, mode::YourBoundaryConditionType)
        
        where YourBoundaryConditionType is a subtype of AbstractBoundaryCondition.
        """ ))
    end
end
to_bc(bc::AbstractBoundaryCondition) = bc


function normalize_boundary_condition(bc, N::Int)
    # Helper to normalize one dimension's BC input to (left, right)
    normalize_one_dim(bc_dim) = 
        bc_dim isa AbstractBoundaryCondition ? (bc_dim, bc_dim) :
        bc_dim isa Tuple && length(bc_dim) == 2 ? bc_dim :
        throw(ArgumentError("Boundary condition per dimension must be either a single AbstractBoundaryCondition or a tuple of two"))

    if bc isa AbstractBoundaryCondition
        # Single BC for all dims, both sides
        return ntuple(_ -> (bc, bc), N)
    elseif bc isa Tuple
        if length(bc) == N
            # Tuple with one entry per dimension
            # Each entry can be a single BC or (left,right) tuple
            return ntuple(i -> normalize_one_dim(bc[i]), N)
        else
            throw(ArgumentError("Boundary condition tuple length $(length(bc)) does not match expected dimension $N"))
        end
    else
        throw(ArgumentError("Invalid boundary_condition specification: $bc"))
    end
end

# Helper: convert symbol or concrete BC to concrete BC
#function to_bc(x)
#    if x isa AbstractBoundaryCondition
#        return x
#    elseif x isa Symbol
#        bc = get(BC_SYMBOL_MAP, x, nothing)
#        if bc === nothing
#            throw(ArgumentError("Unknown boundary condition symbol: $x"))
#        end
#        return bc
#    else
#        throw(ArgumentError("Invalid boundary condition: $x"))
#    end
#end

# Normalize one dimension BC spec to (left, right)
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
        # single BC or symbol for all dims
        bc_concrete = to_bc(bc)
        return ntuple(_ -> (bc_concrete, bc_concrete), N)
    end
end


# ------------------------------------------------------------------------------
# Periodicity check
# ------------------------------------------------------------------------------

function isperiodic(bc::AbstractBoundaryCondition) 
    return bc isa Periodic 
end


function infer_periodicity(boundary_condition::NTuple{N,NTuple{2,AbstractBoundaryCondition}}) where {N}
    ntuple(i -> isperiodic(boundary_condition[i][1]) && isperiodic(boundary_condition[i][2]), Val(N))
end



