using Test
using Aqua
using HaloArrays

@testset "Aqua quality assurance" begin
    # MPIPreferences is a deliberate direct dependency (it configures the MPI
    # binary via LocalPreferences) even though src/ never calls it.
    # Ambiguities are checked: the collection constructors are field-type-first
    # only (the element-type-first forms that caused ambiguities were removed).
    Aqua.test_all(HaloArrays; stale_deps = (ignore = [:MPIPreferences],))
end
