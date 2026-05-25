using HaloArrays
using MPI
using OhMyThreads: tforeach
using Printf
using Random

if !MPI.Initialized()
    MPI.Init()
end

const SerialField2D = Union{HaloArray,LocalHaloArray}

heatbath_parameters(; mass2=1.0, kappa=1.0) = (;
    mass2,
    kappa,
    denom=mass2 + 4 * kappa,
    sigma=sqrt(inv(mass2 + 4 * kappa)),
)

checkerboard_color(i, j) = isodd(i + j)

function neighbor_sum_2d(data, I)
    i, j = Tuple(I)
    return data[i - 1, j] + data[i + 1, j] + data[i, j - 1] + data[i, j + 1]
end

function update_color!(phi::SerialField2D, rng, color::Bool, p)
    data = parent(phi)
    h = halo_width(phi)

    @inbounds for I in CartesianIndices(interior_range(phi))
        storage_i, storage_j = Tuple(I)
        global_i, global_j = owned_to_global_index(phi, (storage_i - h, storage_j - h))

        if checkerboard_color(global_i, global_j) == color
            mean = p.kappa * neighbor_sum_2d(data, I) / p.denom
            data[I] = mean + p.sigma * randn(rng)
        end
    end

    return phi
end

function update_color!(phi::ThreadedHaloArray, rngs, color::Bool, p)
    h = halo_width(phi)
    tile_cells = tile_size(phi)

    tforeach(1:tile_count(phi); scheduler=:static) do tile_id
        data = tile_parent(phi, tile_id)
        coord = tile_coordinates(phi, tile_id)
        rng = rngs[tile_id]

        @inbounds for I in CartesianIndices(interior_range(phi))
            storage_i, storage_j = Tuple(I)
            owned_i = storage_i - h
            owned_j = storage_j - h
            global_i = (coord[1] - 1) * tile_cells[1] + owned_i
            global_j = (coord[2] - 1) * tile_cells[2] + owned_j

            if checkerboard_color(global_i, global_j) == color
                mean = p.kappa * neighbor_sum_2d(data, I) / p.denom
                data[I] = mean + p.sigma * randn(rng)
            end
        end
    end

    return phi
end

function update_color!(fields::MultiHaloArray, rngs::NamedTuple, color::Bool, p)
    for name in keys(fields.arrays)
        update_color!(fields[name], rngs[name], color, p)
    end
    return fields
end

function update_color!(fields::ArrayOfHaloArray, rngs, color::Bool, p)
    for I in eachindex(parent(fields))
        update_color!(parent(fields)[I], rngs[I], color, p)
    end
    return fields
end

function heatbath_sweep!(phi, rng, p)
    synchronize_halo!(phi)
    update_color!(phi, rng, false, p)
    synchronize_halo!(phi)
    update_color!(phi, rng, true, p)
    synchronize_halo!(phi)
    return phi
end

function run_heatbath!(phi, rng, p; sweeps=50)
    fill!(phi, zero(eltype(phi)))
    synchronize_halo!(phi)

    for _ in 1:sweeps
        heatbath_sweep!(phi, rng, p)
    end

    return phi
end

observables(phi) = (; mean=sum(phi) / length(phi), phi2=sum(abs2, phi) / length(phi))

heatbath_rng(phi::LocalHaloArray, seed) = MersenneTwister(seed)
heatbath_rng(phi::HaloArray, seed) = MersenneTwister(seed + MPI.Comm_rank(get_comm(phi)))
heatbath_rng(phi::ThreadedHaloArray, seed) =
    [MersenneTwister(seed + tile_id) for tile_id in 1:tile_count(phi)]

function heatbath_rng(fields::MultiHaloArray, seed)
    names = keys(fields.arrays)
    rngs = ntuple(i -> heatbath_rng(fields.arrays[names[i]], seed + 1000 * i), Val(length(names)))
    return NamedTuple{names}(rngs)
end

function heatbath_rng(fields::ArrayOfHaloArray, seed)
    linear = LinearIndices(parent(fields))
    return map(CartesianIndices(parent(fields))) do I
        heatbath_rng(parent(fields)[I], seed + 1000 * linear[I])
    end
end

function print_observables(label, phi, obs)
    @printf("%-22s size=%s mean=% .6e phi2=%.6e\n", label, string(global_size(phi)), obs.mean, obs.phi2)
    return nothing
end
