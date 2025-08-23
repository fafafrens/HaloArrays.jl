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



# Global mapping symbol -> BC type/instance (modificabile tramite register_bc)
const BC_SYMBOL_MAP = Dict{Symbol, Union{DataType, AbstractBoundaryCondition}}(
    :reflecting     => Reflecting,
    :antireflecting => Antireflecting,
    :repeating      => Repeating,
    :periodic       => Periodic
)

"""
    register_bc(sym::Symbol, ctor_or_instance)

Registra una nuova mappatura per `to_bc(:sym)`. `ctor_or_instance` può essere:
- un Type <: AbstractBoundaryCondition (viene istanziato quando usato),
- oppure una istanza di AbstractBoundaryCondition.
Esempio: register_bc(:mybc, MyBC) o register_bc(:mybc, MyBC()).
"""
function register_bc(sym::Symbol, ctor_or_instance)
    if !(ctor_or_instance isa DataType || ctor_or_instance isa AbstractBoundaryCondition)
        throw(ArgumentError("ctor_or_instance must be a Type <: AbstractBoundaryCondition or an instance"))
    end
    BC_SYMBOL_MAP[sym] = ctor_or_instance
    return nothing
end

# Unified converter: Symbol | Type | instance -> concrete BC instance
function to_bc(x)
    if x isa Symbol
        entry = get(BC_SYMBOL_MAP, x, nothing)
        entry === nothing && throw(ArgumentError("Unknown boundary condition symbol: $x. Register it with register_bc(:$x, MyBC)."))
        return entry isa DataType ? entry() : entry
    elseif x isa DataType && x <: AbstractBoundaryCondition
        return x()
    elseif x isa AbstractBoundaryCondition
        return x
    else
        throw(ArgumentError("Invalid boundary_condition: $x"))
    end
end

# Normalize one-dimension BC spec to (left, right)
function normalize_one_dim(bc_dim)
    if bc_dim isa Tuple && length(bc_dim) == 2
        return (to_bc(bc_dim[1]), to_bc(bc_dim[2]))
    else
        return (to_bc(bc_dim), to_bc(bc_dim))
    end
end

# Normalize full spec to NTuple{N,NTuple{2,AbstractBoundaryCondition}}
function normalize_boundary_condition(bc, N::Int)
    if bc isa Tuple
        if length(bc) == N
            return ntuple(i -> normalize_one_dim(bc[i]), N)
        else
            throw(ArgumentError("Boundary condition tuple length $(length(bc)) does not match expected dimension $N"))
        end
    else
        # single BC (Symbol/Type/instance) for all dims
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



