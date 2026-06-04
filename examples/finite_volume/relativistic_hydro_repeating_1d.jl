# ============================================================
# 1-D Special Relativistic Hydrodynamics — :repeating outflow BC
#
#   julia --project=. examples/finite_volume/relativistic_hydro_repeating_1d.jl
#
# Same relativistic Sod shock tube as relativistic_hydro_1d.jl, but with the
# simplest possible outflow boundary: each conserved field's ghost cell is set
# to its edge value (zeroth-order extrapolation).  Because that is an ordinary
# *per-field* boundary condition, `synchronize_halo!` fills the ghosts and there
# is no coupled BC to write — the per-step boundary hook is a no-op.
#
# Note the uniform-BC shorthand: `fields=(...)` + `boundary_condition=:repeating`
# builds all three fields with the same BC, instead of a per-field NamedTuple.
#
# Physics/numerics are shared in relativistic_common.jl.
# ============================================================

include("relativistic_common.jl")

# Every field gets :repeating; synchronize_halo! fills the ghosts for us.
make_state(nx) = LocalMultiHaloArray(Float64, (nx,), 1;
    fields=(:D, :S, :tau), boundary_condition=:repeating)

run_relativistic_sod(make_state, u -> nothing;
    eos=IdealGas(5.0 / 3.0), label="repeating (zeroth-order) outflow")
