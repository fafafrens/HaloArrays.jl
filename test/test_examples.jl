using Test
using HaloArrays

# ============================================================
# Example smoke tests
#
# Each script is run end-to-end (including its trailing auto-run driver) in a
# throwaway module, asserting it completes without error. This guards against
# silent example rot when the library API changes — a runtime regression in an
# example (e.g. a BoundsError from an indexing-contract change) fails here.
#
# Only CPU examples whose dependencies are in the package test environment are
# listed. Deliberately excluded:
#   - MPI scripts          (need `mpiexec -n …`; covered by the MPI CI job)
#   - Metal / KernelAbstractions GPU scripts
#   - poisson/*            (needs SciMLOperators, not a test dependency)
#
# Gated by HALOARRAYS_RUN_EXAMPLE_TESTS (see runtests.jl); the dedicated CI job
# runs it with JULIA_NUM_THREADS=2 so the threaded examples exercise >1 tile.
# ============================================================

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")

# Need only HaloArrays + StaticArrays / Printf / OhMyThreads / LinearAlgebra /
# HDF5 / Random — all available in the package test environment.
const SMOKE_EXAMPLES = [
    "finite_volume/acoustics_characteristic_1d.jl",
    "finite_volume/burgers_1d.jl",
    "finite_volume/relativistic_hydro_1d.jl",
    "finite_volume/relativistic_hydro_repeating_1d.jl",
    "finite_volume/relativistic_hydro_mu0_1d.jl",
    "finite_volume/relativistic_hydro_mu0_2d.jl",
    "finite_volume/relativistic_hydro_mu0_3d.jl",
    "finite_volume/relativistic_hydro_Tmu_1d.jl",
    "finite_volume/relativistic_hydro_Tmu_2d.jl",
    "finite_volume/relativistic_hydro_Tmu_3d.jl",
    "finite_volume/relativistic_hydro_cylindrical_1d.jl",
    "finite_volume/relativistic_hydro_cylindrical_threaded_1d.jl",
    "heat/local.jl",
    "hydro/local_2d.jl",
    "hydro/threaded_2d.jl",
    "lattice/scalar_local_threaded_2d.jl",
    "lattice/su2_wilson_local_threaded_2d.jl",
    "tutorials/local.jl",
    "tutorials/broadcast.jl",
    "tutorials/threaded.jl",
]

# Additionally need DiffEqBase + OrdinaryDiffEq (skipped if DiffEqBase is not in
# the active environment, e.g. when this file is run via `--project=.` directly).
const DIFFEQ_EXAMPLES = [
    "finite_volume/advection_diffeq_1d.jl",
    "finite_volume/burgers_diffeq_1d.jl",
    "finite_volume/relativistic_hydro_Tmu_diffeq_1d.jl",
    "heat/local_vs_threaded.jl",
    "tutorials/diffeq.jl",
]

function _smoke_run(rel)
    path = joinpath(EXAMPLES_DIR, rel)
    # A module created with `module … end` syntax gets its own `include`/`eval`,
    # which a bare `Module()` does not — scripts that `include("common.jl")` need
    # it. Name it after the file so each script gets a fresh namespace.
    modname = Symbol("Example_", replace(rel, r"[^A-Za-z0-9]" => "_"))
    sandbox = Core.eval(Main, :(module $modname end))
    return redirect_stdout(devnull) do
        try
            Base.include(sandbox, path)
            # Scripts that gate their driver behind
            # `if abspath(PROGRAM_FILE) == @__FILE__` define `main` but do not run
            # it on include; the rest auto-run a `run_*()` at top level. Evaluate
            # the `main` call inside the sandbox (in the post-include world) so the
            # simulation runs either way, without a world-age binding warning.
            Core.eval(sandbox, :(isdefined(@__MODULE__, :main) && main()))
            true
        catch err
            @error "example script failed" example = rel exception = (err, catch_backtrace())
            false
        end
    end
end

@testset "Example scripts (smoke)" begin
    for rel in SMOKE_EXAMPLES
        @testset "$rel" begin
            @test _smoke_run(rel)
        end
    end

    if Base.find_package("DiffEqBase") !== nothing
        for rel in DIFFEQ_EXAMPLES
            @testset "$rel" begin
                @test _smoke_run(rel)
            end
        end
    else
        @info "Skipping DiffEq example smoke tests (DiffEqBase not in this environment; run via Pkg.test)"
    end
end
