module HaloArraysDiffEqBaseExt

using DiffEqBase
using HaloArrays

function DiffEqBase.recursive_length(halo::AbstractSingleHaloArray)
    return prod(global_size(halo))
end

function DiffEqBase.recursive_length(halo::ArrayOfHaloArray)
    return prod(global_size(halo))
end

function DiffEqBase.recursive_length(halo::MultiHaloArray)
    return sum(DiffEqBase.recursive_length, values(halo.arrays))
end

function DiffEqBase.recursive_length(halo::MaybeHaloArray)
    return DiffEqBase.recursive_length(getdata(halo))
end

function DiffEqBase.NAN_CHECK(halo::AbstractSingleHaloArray)
    return any(DiffEqBase.NAN_CHECK, halo)
end

function DiffEqBase.NAN_CHECK(halo::ArrayOfHaloArray)
    return any(DiffEqBase.NAN_CHECK, halo)
end

function DiffEqBase.NAN_CHECK(halo::MultiHaloArray)
    return any(DiffEqBase.NAN_CHECK, halo)
end

function DiffEqBase.NAN_CHECK(halo::MaybeHaloArray)
    return isactive(halo) && DiffEqBase.NAN_CHECK(getdata(halo))
end

function DiffEqBase.ODE_DEFAULT_UNSTABLE_CHECK(dt, halo::AbstractHaloArray, p, t)
    return DiffEqBase.NAN_CHECK(halo)
end

end
