module HaloArraysDiffEqBaseExt

using DiffEqBase
using HaloArrays

function DiffEqBase.recursive_length(halo::Union{HaloArray,LocalHaloArray})
    return prod(global_size(halo))
end

function DiffEqBase.recursive_length(halo::MultiHaloArray)
    return sum(DiffEqBase.recursive_length, values(halo.arrays))
end

function DiffEqBase.NAN_CHECK(halo::Union{HaloArray,LocalHaloArray})
    return any(DiffEqBase.NAN_CHECK, halo)
end

function DiffEqBase.NAN_CHECK(halo::MultiHaloArray)
    return any(DiffEqBase.NAN_CHECK, halo)
end

function DiffEqBase.ODE_DEFAULT_UNSTABLE_CHECK(dt, halo::Union{HaloArray,LocalHaloArray,MultiHaloArray}, p, t)
    return DiffEqBase.NAN_CHECK(halo)
end

end
