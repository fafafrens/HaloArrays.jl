# ============================================================
# 1-D Special Relativistic Hydrodynamics — coupled outflow BC
#
#   julia --project=. examples/finite_volume/relativistic_hydro_1d.jl
#
# Relativistic Sod shock tube with a *coupled* characteristic outflow
# boundary: the x-ghosts are filled by recovering the primitive state at the
# edge, clamping the velocity against inflow, and converting back to conserved
# form — all three fields written together because the primitive recovery
# couples them.  This is the use case for AbstractCoupledBoundaryCondition.
#
# The physics/numerics (EOS, Rusanov flux, FaceRanges FV update, SSP-RK2) live
# in relativistic_common.jl.  See relativistic_hydro_repeating_1d.jl for the
# simpler per-field :repeating outflow.
# ============================================================

include("relativistic_common.jl")

# Coupled outflow: recover primitives at the boundary cell, clamp the velocity
# so there is no inflow, convert back to conserved form, fill all three ghosts.
# The BC carries the EOS, since the recovery/conversion are thermodynamic.
struct RelativisticOutflow{E<:AbstractEOS} <: AbstractCoupledBoundaryCondition
    eos::E
end

function HaloArrays.apply_coupled_bc!(bc::RelativisticOutflow, state, ::Side{Sd}, ::Dim{1}) where {Sd}
    Df, Sf, Tf = eachfield(state)
    U = SVector(get_send_view(Side(Sd), Dim(1), Df)[1],
                get_send_view(Side(Sd), Dim(1), Sf)[1],
                get_send_view(Side(Sd), Dim(1), Tf)[1])
    ρ, v, p = prim_from_cons(bc.eos, U)
    v = Sd == 1 ? max(v, 0.0) : min(v, 0.0)   # no supersonic inflow
    G = cons_from_prim(bc.eos, ρ, v, p)
    fill!(get_recv_view(Side(Sd), Dim(1), Df), G[1])
    fill!(get_recv_view(Side(Sd), Dim(1), Sf), G[2])
    fill!(get_recv_view(Side(Sd), Dim(1), Tf), G[3])
    return nothing
end

# x-boundaries opt out of the per-field BC so the coupled hook can fill them.
make_state(nx) = LocalMultiHaloArray(Float64, (nx,), 1;
    boundary_conditions=(D=:noboundary, S=:noboundary, tau=:noboundary))

eos = IdealGas(5.0 / 3.0)
run_relativistic_sod(make_state, u -> apply_coupled_bc!(RelativisticOutflow(eos), u);
    eos=eos, label="coupled characteristic outflow")
