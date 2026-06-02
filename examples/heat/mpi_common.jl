using MPI

include("common.jl")

function _mpi_domain_lengths(domain_length, ::Val{N}) where {N}
    return domain_length isa Number ? ntuple(_ -> domain_length, Val(N)) : Tuple(domain_length)
end

function _mpi_global_mean(u::HaloArray)
    comm = get_comm(u)
    local_total = sum(interior_view(u))
    total = MPI.Allreduce(local_total, +, comm)
    return total / prod(global_size(u))
end

function run_mpi_heat(::Val{N};
        owned_dims,
        alpha=1.0,
        cfl=0.4,
        domain_length=1.0,
        nt=50,
        save_hdf5=false,
        output_name="heat_diffusion_mpi_$(N)d") where {N}
    if !MPI.Initialized()
        MPI.Init()
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    topology = CartesianTopology(comm, ntuple(_ -> 0, Val(N)); periodic=ntuple(_ -> true, Val(N)))
    u = HaloArray(Float64, owned_dims, 1, topology; boundary_condition=:periodic)

    lengths = _mpi_domain_lengths(domain_length, Val(N))
    dx = ntuple(d -> lengths[d] / global_size(u)[d], Val(N))
    dt = stable_heat_dt(alpha, cfl, dx)

    fill_centered_gaussian!(u; baseline=1.0, amplitude=1.0)
    solve_heat!(u; alpha, dt, dx, nt)

    if save_hdf5
        gather_and_save_haloarray(output_name, u; root=0)
    end

    final_mean = _mpi_global_mean(u)
    if rank == 0
        println("MPI heat diffusion $(N)D completed.")
        println("  ranks = ", MPI.Comm_size(comm))
        println("  topology dims = ", topology.dims)
        println("  global size = ", global_size(u))
        println("  dt = ", dt)
        println("  final mean = ", final_mean)
    end

    return u, final_mean
end
