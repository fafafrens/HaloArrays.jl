module HaloArraysRecursiveArrayToolsExt

using HaloArrays
using RecursiveArrayTools

function RecursiveArrayTools.recursivefill!(halo::HaloArray, value)
    fill!(halo, value)
    return halo
end

end
