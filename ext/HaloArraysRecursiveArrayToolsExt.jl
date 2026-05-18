module HaloArraysRecursiveArrayToolsExt

using HaloArrays
using RecursiveArrayTools

function RecursiveArrayTools.recursivefill!(halo::Union{HaloArray,LocalHaloArray}, value)
    fill!(halo, value)
    return halo
end

function RecursiveArrayTools.recursivefill!(halo::MultiHaloArray, value)
    foreach_field!(field -> RecursiveArrayTools.recursivefill!(field, value), halo)
    return halo
end

end
