using Test
using MPI
using HaloArrays

if !MPI.Initialized()
    MPI.Init()
end

_env_true(name)  = lowercase(get(ENV, name, "")) in ("1", "true", "yes", "on")
_env_false(name) = lowercase(get(ENV, name, "")) in ("0", "false", "no", "off")

const MPI_COMM = MPI.COMM_WORLD
const MPI_SIZE = MPI.Comm_size(MPI_COMM)
const RUN_UNIT_TESTS = !_env_false("HALOARRAYS_RUN_UNIT_TESTS")
const RUN_MPI_TESTS  = !_env_false("HALOARRAYS_RUN_MPI_TESTS") &&
    (_env_true("HALOARRAYS_RUN_MPI_TESTS") || MPI_SIZE > 1)
# Example smoke tests are opt-in (they run the full demo drivers); off by default
# so the normal unit suite stays fast. Enabled by the dedicated CI examples job.
const RUN_EXAMPLE_TESTS = _env_true("HALOARRAYS_RUN_EXAMPLE_TESTS")

include_test(name) = include(joinpath(@__DIR__, name))

@testset "HaloArrays" begin
    @test true

    if RUN_UNIT_TESTS
        # Aqua lives in the test extras; available under Pkg.test but not when
        # running this file directly with --project=. — skip gracefully then.
        if Base.find_package("Aqua") !== nothing
            include_test("test_aqua.jl")
        else
            @info "Skipping Aqua tests (Aqua not in this environment; run via Pkg.test)"
        end
        include_test("test_public_api.jl")
        include_test("test_haloarray_helpers.jl")
        include_test("test_boundary.jl")
        include_test("test_function_bc.jl")
        include_test("test_coupled_boundary.jl")
        include_test("test_cartesian_topology.jl")
        include_test("test_local_haloarray.jl")
        include_test("test_threaded_haloarray.jl")
        include_test("test_thread_backend.jl")
        include_test("test_heat_diffusion_examples.jl")
        include_test("test_relativistic.jl")
        include_test("test_mharray.jl")
        include_test("test_arrayofhaloarray.jl")
        include_test("test_fallbacks.jl")
        include_test("test_maybe_broadcast.jl")
        include_test("test_local_threaded_reduction.jl")
        include_test("test_dims_reduction.jl")
        include_test("test_svector_eltype.jl")
        include_test("test_hdf5_local_threaded.jl")
        # Adapt integration: JLArrays (GPUArrays' CPU backend) is a test extra —
        # available under Pkg.test, skipped on a bare --project=. run.
        if Base.find_package("JLArrays") !== nothing
            include_test("test_adapt.jl")
        else
            @info "Skipping Adapt tests (JLArrays not in this environment; run via Pkg.test)"
        end
        # LinearSolve/Krylov extension: available under Pkg.test, skipped on a
        # bare --project=. run (same as Aqua).
        if all(p -> Base.find_package(p) !== nothing, ("LinearSolve", "Krylov", "OrdinaryDiffEq"))
            include_test("test_linearsolve_ext.jl")
        else
            @info "Skipping LinearSolve extension tests (LinearSolve/Krylov/OrdinaryDiffEq not in this environment; run via Pkg.test)"
        end
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
            include_test("test_reduce_marray.jl")
            include_test("test_gather.jl")
            include_test("test_saving_hdf5.jl")
            include_test("test_mpi_reduce_save.jl")
            include_test("test_mpi_reduction_plan.jl")
            # Distributed implicit solve (HaloKrylov) — available under Pkg.test.
            if all(p -> Base.find_package(p) !== nothing, ("LinearSolve", "Krylov", "OrdinaryDiffEq"))
                include_test("test_mpi_implicit.jl")
            else
                @info "Skipping MPI implicit test (LinearSolve/Krylov/OrdinaryDiffEq not in this environment)"
            end
        end
    else
        @info "Skipping MPI tests (run with mpiexec -n 2 or set HALOARRAYS_RUN_MPI_TESTS=true)"
    end

    if RUN_EXAMPLE_TESTS
        include_test("test_examples.jl")
    else
        @info "Skipping example smoke tests (set HALOARRAYS_RUN_EXAMPLE_TESTS=true to enable)"
    end

    MPI.Barrier(MPI_COMM)
end
