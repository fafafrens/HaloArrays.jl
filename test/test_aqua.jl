using Test
using Aqua
using HaloArrays

@testset "Aqua quality assurance" begin
    # MPIPreferences is a deliberate direct dependency (it configures the MPI
    # binary via LocalPreferences) even though src/ never calls it.
    #
    # ambiguities = false: Aqua reports ~71 method ambiguities, almost all
    # between the generic element-type collection constructors
    # (ArrayOfHaloArray(::Type{T}, ...)) and the specialized field-type ones
    # (ArrayOfHaloArray(::Type{ThreadedHaloArray}, ...)), plus similar().
    # Resolving them is a constructor-API redesign tracked as follow-up work.
    Aqua.test_all(HaloArrays;
        ambiguities = false,
        stale_deps = (ignore = [:MPIPreferences],))
end
