"""
    gather_haloarray(u; root=0)

Assemble the global interior (ghost-free) data of `u` into an ordinary `Array`.

- MPI [`HaloArray`](@ref): every rank sends its subdomain; `root` returns the
  assembled global array and every other rank returns `nothing`. Collective.
- [`LocalHaloArray`](@ref) / [`ThreadedHaloArray`](@ref): returns the assembled
  interior directly (tiles are stitched in global order); `root` is ignored.
- [`ArrayOfHaloArray`](@ref): an array with the field axes first, then the
  spatial axes (`field_shape(u)..., global_size...`). [`MultiHaloArray`](@ref):
  a `NamedTuple` with one assembled array per field. Distributed collections
  follow the `HaloArray` rule per field.
- [`MaybeHaloArray`](@ref) (a `dims=` reduction result): active ranks gather on
  its sub-communicator, inactive ranks return `nothing`.

The same code therefore works on every backend, e.g. for output:
```julia
A = gather_haloarray(u)
is_root(u) && h5write("snapshot.h5", "rho", A)   # with HDF5.jl
```
"""
function gather_haloarray(halo::HaloArray; root::Int=0)
    comm = halo.topology.cart_comm
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)
    N = ndims(halo)
    T = eltype(halo)

    coords = halo.topology.cart_coords
    dims = halo.topology.dims

    local_data = interior_view(halo)

    owned_shape = size(local_data)
    local_len = prod(owned_shape)

    # Gather all buffers as flat arrays
    sendbuf = collect(vec(local_data))
    recvbuf = if rank == root
        Array{T}(undef, local_len * nproc)
    else
        nothing
    end

    MPI.Gather!(sendbuf, recvbuf, comm; root=root)

    # Reconstruct the full array at the root
    if rank == root
        global_size = ntuple(i -> dims[i] * owned_shape[i], Val(N))
        global_array = Array{T}(undef, global_size)

        for r in 0:nproc-1
            coords_r = MPI.Cart_coords(comm, r) |> Tuple
            offset = ntuple(i -> coords_r[i] * owned_shape[i], Val(N))
            inds = ntuple(i -> (offset[i]+1):(offset[i]+owned_shape[i]), Val(N))

            flat_offset = r * local_len + 1
            subarray = reshape(view(recvbuf, flat_offset:flat_offset + local_len - 1), owned_shape...)
            @views global_array[inds...] .= subarray
        end

        return global_array
    else
        return nothing
    end
end

gather_haloarray(halo::LocalHaloArray; root::Int=0) = Array(interior_view(halo))

function gather_haloarray(halo::ThreadedHaloArray{T,N}; root::Int=0) where {T,N}
    data = Array{T}(undef, global_size(halo))
    owned = tile_size(halo)
    for tile_id in 1:tile_count(halo)
        coords = tile_coordinates(halo, tile_id)
        inds = ntuple(Val(N)) do d
            ((coords[d] - 1) * owned[d] + 1):(coords[d] * owned[d])
        end
        data[inds...] .= interior_view(halo, tile_id)
    end
    return data
end

function gather_haloarray(halo::ArrayOfHaloArray; root::Int=0)
    fields = parent(halo)
    data = nothing
    # Gather every field on every rank (each is a collective on MPI); assemble
    # whatever comes back, which is `nothing` only on non-root MPI ranks.
    for I in CartesianIndices(fields)
        field_data = gather_haloarray(fields[I]; root=root)
        field_data === nothing && continue
        data === nothing &&
            (data = Array{eltype(halo)}(undef, (field_shape(halo)..., size(field_data)...)))
        data[Tuple(I)..., ntuple(_ -> Colon(), ndims(field_data))...] .= field_data
    end
    return data
end

# A dims-reduction result lives on a sub-communicator: active ranks gather on
# it, inactive ranks return nothing (so `is_root(m) && h5write(...)` just works).
gather_haloarray(m::MaybeHaloArray; root::Int=0) =
    is_active(m) ? gather_haloarray(getdata(m); root=root) : nothing

function gather_haloarray(halo::MultiHaloArray; root::Int=0)
    fields = map(f -> gather_haloarray(f; root=root), values(halo.arrays))
    return NamedTuple{keys(halo.arrays)}(fields)
end
