module HaloArraysDiffEqBaseExt

using DiffEqBase
using HaloArrays

function DiffEqBase.recursive_length(halo::AbstractSingleHaloArray)
    return prod(global_size(halo))
end

# One method covers both collection flavors (the per-field sum used previously
# for MultiHaloArray equals prod(global_size) since all fields share one geometry).
function DiffEqBase.recursive_length(halo::HaloArrays.FieldCollection)
    return prod(global_size(halo))
end

function DiffEqBase.recursive_length(halo::MaybeHaloArray)
    return DiffEqBase.recursive_length(getdata(halo))
end

function DiffEqBase.NAN_CHECK(halo::AbstractSingleHaloArray)
    return any(DiffEqBase.NAN_CHECK, halo)
end

function DiffEqBase.NAN_CHECK(halo::HaloArrays.FieldCollection)
    return any(DiffEqBase.NAN_CHECK, halo)
end

function DiffEqBase.NAN_CHECK(halo::MaybeHaloArray)
    return is_active(halo) && DiffEqBase.NAN_CHECK(getdata(halo))
end

function DiffEqBase.ODE_DEFAULT_UNSTABLE_CHECK(dt, halo::AbstractHaloArray, p, t)
    return DiffEqBase.NAN_CHECK(halo)
end

end
