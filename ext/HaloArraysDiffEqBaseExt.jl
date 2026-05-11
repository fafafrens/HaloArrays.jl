module HaloArraysDiffEqBaseExt

using DiffEqBase
using HaloArrays

function DiffEqBase.NAN_CHECK(halo::HaloArray)
    return any(DiffEqBase.NAN_CHECK, halo)
end

function DiffEqBase.ODE_DEFAULT_UNSTABLE_CHECK(dt, halo::HaloArray, p, t)
    return DiffEqBase.NAN_CHECK(halo)
end

end
