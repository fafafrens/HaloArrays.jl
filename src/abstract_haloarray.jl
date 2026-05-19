abstract type AbstractHaloArray end

abstract type AbstractSingleHaloArray <: AbstractHaloArray end
abstract type AbstractDistributedHaloArray <: AbstractSingleHaloArray end
abstract type AbstractSerialHaloArray <: AbstractSingleHaloArray end

abstract type AbstractHaloCollection <: AbstractHaloArray end
