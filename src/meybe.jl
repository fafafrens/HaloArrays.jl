# Struct definition
struct MaybeActive{T}
    value::T
    active::Bool
end

# Constructors
active(x) = MaybeActive(x, true)
inactive(x) = MaybeActive(x, false)

# Query functions
isactive(m::MaybeActive) = m.active

# Get with default
"""
    get(m::MaybeActive, default)

Return `m.value` if active, otherwise return `default`.
"""
get(m::MaybeActive, default) = m.active ? m.value : default

unsafe_get(m::MaybeActive) = m.value

