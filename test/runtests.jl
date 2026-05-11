using Test
using MPI
using HaloArrays

if !MPI.Initialized()
    MPI.Init()
end

_env_true(name) = lowercase(get(ENV, name, "")) in ("1", "true", "yes", "on")
_env_false(name) = lowercase(get(ENV, name, "")) in ("0", "false", "no", "off")

const MPI_COMM = MPI.COMM_WORLD
const MPI_SIZE = MPI.Comm_size(MPI_COMM)
const RUN_UNIT_TESTS = !_env_false("HALOARRAYS_RUN_UNIT_TESTS")
const MPI_TESTS_REQUESTED = _env_true("HALOARRAYS_RUN_MPI_TESTS")
const RUN_MPI_TESTS = !_env_false("HALOARRAYS_RUN_MPI_TESTS") && (MPI_TESTS_REQUESTED || MPI_SIZE > 1)

# Helper to include test files relative to this directory
function include_test(name)
    include(joinpath(@__DIR__, name))
end

@testset "HaloArrays" begin
    # Always verify the package loads
    @test true

    if RUN_UNIT_TESTS
        include_test("test_public_api.jl")
        include_test("test_haloarray_helpers.jl")
        include_test("test_boundary.jl")
        include_test("test_cartesian_topology.jl")
        include_test("test_mharray.jl")
        include_test("test_maybe_broadcast.jl")
        include_test("test_ode.jl")
    else
        @info "Skipping unit tests (set HALOARRAYS_RUN_UNIT_TESTS=true to enable)"
    end

    if RUN_MPI_TESTS
        if MPI_SIZE <= 1
            @test MPI_SIZE > 1
        else
            include_test("test_mpi_cartesian_topology.jl")
            include_test("test_mpi_halo_exchange.jl")
            include_test("test_reduce.jl")
            include_test("test_gather.jl")
            if !RUN_UNIT_TESTS
                include_test("test_ode.jl")
            end
        end
    else
        @info "Skipping MPI tests (run with mpiexec -n 2 or set HALOARRAYS_RUN_MPI_TESTS=true)"
    end

    MPI.Barrier(MPI_COMM)
end
