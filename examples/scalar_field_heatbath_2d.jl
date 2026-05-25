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

function zero_field!(phi)
    fill!(phi, zero(eltype(phi)))
    synchronize_halo!(phi)
    return phi
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
    zero_field!(phi)
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

function run_local_heatbath(; n=(32, 32), sweeps=50, seed=1234, mass2=1.0, kappa=1.0)
    phi = LocalHaloArray(Float64, n, 1; boundary_condition=:periodic)
    run_heatbath!(phi, heatbath_rng(phi, seed), heatbath_parameters(; mass2, kappa); sweeps)
    return phi, observables(phi)
end

function run_threaded_heatbath(;
        n=(32, 32),
        tile_dims=(2, 2),
        sweeps=50,
        seed=1234,
        mass2=1.0,
        kappa=1.0,
)
    all(d -> n[d] % tile_dims[d] == 0, 1:2) ||
        throw(ArgumentError("n=$n must be divisible by tile_dims=$tile_dims"))

    tile_cells = ntuple(d -> n[d] ÷ tile_dims[d], Val(2))
    phi = ThreadedHaloArray(Float64, tile_cells, 1; dims=tile_dims, boundary_condition=:periodic)
    run_heatbath!(phi, heatbath_rng(phi, seed), heatbath_parameters(; mass2, kappa); sweeps)
    return phi, observables(phi)
end

function mpi_lattice(n)
    comm = MPI.COMM_WORLD
    topology = CartesianTopology(comm, (0, 0); periodic=(true, true))
    all(d -> n[d] % topology.dims[d] == 0, 1:2) ||
        throw(ArgumentError("n=$n must be divisible by MPI topology dims=$(topology.dims)"))

    owned_cells = ntuple(d -> n[d] ÷ topology.dims[d], Val(2))
    return topology, owned_cells
end

function run_mpi_heatbath(; n=(32, 32), sweeps=50, seed=1234, mass2=1.0, kappa=1.0)
    topology, owned_cells = mpi_lattice(n)
    phi = HaloArray(Float64, owned_cells, 1, topology; boundary_condition=:periodic)
    run_heatbath!(phi, heatbath_rng(phi, seed), heatbath_parameters(; mass2, kappa); sweeps)
    return phi, observables(phi)
end

function run_local_multifield_heatbath(; n=(32, 32), sweeps=50, seed=4321, mass2=1.0, kappa=1.0)
    fields = MultiHaloArray(LocalHaloArray, Float64, n, 1;
        boundary_conditions=(phi=:periodic, chi=:periodic))
    run_heatbath!(fields, heatbath_rng(fields, seed), heatbath_parameters(; mass2, kappa); sweeps)
    return fields, observables(fields)
end

function run_local_replicas_heatbath(;
        n=(32, 32),
        nreplicas=2,
        sweeps=50,
        seed=5678,
        mass2=1.0,
        kappa=1.0,
)
    replicas = ArrayOfHaloArray(LocalHaloArray, Float64, n, 1;
        boundary_conditions=fill(:periodic, nreplicas))
    run_heatbath!(replicas, heatbath_rng(replicas, seed), heatbath_parameters(; mass2, kappa); sweeps)
    return replicas, observables(replicas)
end

function run_mpi_multifield_heatbath(; n=(32, 32), sweeps=50, seed=4321, mass2=1.0, kappa=1.0)
    topology, owned_cells = mpi_lattice(n)
    fields = MultiHaloArray(Float64, owned_cells, 1, topology;
        boundary_conditions=(phi=:periodic, chi=:periodic))
    run_heatbath!(fields, heatbath_rng(fields, seed), heatbath_parameters(; mass2, kappa); sweeps)
    return fields, observables(fields)
end

function run_mpi_replicas_heatbath(;
        n=(32, 32),
        nreplicas=2,
        sweeps=50,
        seed=5678,
        mass2=1.0,
        kappa=1.0,
)
    topology, owned_cells = mpi_lattice(n)
    replicas = ArrayOfHaloArray(Float64, owned_cells, 1, topology;
        boundary_conditions=fill(:periodic, nreplicas))
    run_heatbath!(replicas, heatbath_rng(replicas, seed), heatbath_parameters(; mass2, kappa); sweeps)
    return replicas, observables(replicas)
end

function print_observables(label, phi, obs)
    @printf("%-22s size=%s mean=% .6e phi2=%.6e\n", label, string(global_size(phi)), obs.mean, obs.phi2)
    return nothing
end

function main()
    n = (32, 32)
    sweeps = 50
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    nranks = MPI.Comm_size(MPI.COMM_WORLD)

    if nranks == 1
        local_phi, local_obs = run_local_heatbath(; n, sweeps)
        print_observables("LocalHaloArray", local_phi, local_obs)

        threaded_phi, threaded_obs = run_threaded_heatbath(; n, tile_dims=(2, 2), sweeps)
        print_observables("ThreadedHaloArray", threaded_phi, threaded_obs)

        local_fields, local_fields_obs = run_local_multifield_heatbath(; n, sweeps)
        print_observables("Local MultiHaloArray", local_fields, local_fields_obs)

        local_replicas, local_replicas_obs = run_local_replicas_heatbath(; n, nreplicas=2, sweeps)
        print_observables("Local ArrayOfHaloArray", local_replicas, local_replicas_obs)
    end

    mpi_phi, mpi_obs = run_mpi_heatbath(; n, sweeps)
    rank == 0 && print_observables("HaloArray MPI", mpi_phi, mpi_obs)

    mpi_fields, mpi_fields_obs = run_mpi_multifield_heatbath(; n, sweeps)
    rank == 0 && print_observables("MPI MultiHaloArray", mpi_fields, mpi_fields_obs)

    mpi_replicas, mpi_replicas_obs = run_mpi_replicas_heatbath(; n, nreplicas=2, sweeps)
    rank == 0 && print_observables("MPI ArrayOfHaloArray", mpi_replicas, mpi_replicas_obs)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
