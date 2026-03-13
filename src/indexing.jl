Base.pairs(hv::AbstractHeterogeneousVector{T, S}) where {S, T} = pairs(NamedTuple(hv))

function Base.getindex(hv::AbstractHeterogeneousVector{T, S}, idx::Int) where {T, S}
    current_idx = 1
    for (name, field) in pairs(hv)
        unwrapped_field = _unwrap(field)
        if unwrapped_field isa AbstractArray
            field_length = length(unwrapped_field)
            if current_idx <= idx < current_idx + field_length
                return unwrapped_field[idx - current_idx + 1]
            end
            current_idx += field_length
        else
            if idx == current_idx
                return unwrapped_field
            end
            current_idx += 1
        end
    end
    throw(BoundsError(hv, idx))
end

function Base.setindex!(hv::AbstractHeterogeneousVector{T, S}, val, idx::Int) where {T, S}
    current_idx = 1
    for (name, field) in pairs(hv)
        if field isa AbstractArray
            field_length = length(field)
            if current_idx <= idx < current_idx + field_length
                field[idx - current_idx + 1] = val
                return val
            end
            current_idx += field_length
        else
            if idx == current_idx
                _set_value!(field, val)
                return val
            end
            current_idx += 1
        end
    end
    throw(BoundsError(hv, idx))
end

_field_length(field::Ref) = 1
_field_length(field::AbstractArray) = length(field)
# Update length calculation
Base.length(hv::AbstractHeterogeneousVector) = sum(_field_length, NamedTuple(hv))
Base.size(hv::AbstractHeterogeneousVector) = (length(hv),)
Base.firstindex(hv::AbstractHeterogeneousVector) = 1
Base.lastindex(hv::AbstractHeterogeneousVector) = length(hv)

# Flat Iteration Support
struct Chain{T <: Tuple}
    xss::T
end
chain(xss...) = Chain(xss)
Base.length(it::Chain{Tuple{}}) = 0
Base.length(it::Chain) = sum(length, it.xss)
Base.eltype(::Type{Chain{T}}) where {T} = typejoin([eltype(t) for t in T.parameters]...)

function Base.iterate(it::Chain)
    i = 1
    xs_state = nothing
    while i <= length(it.xss)
        xs_state = iterate(it.xss[i])
        xs_state !== nothing && return xs_state[1], (i, xs_state[2])
        i += 1
    end
    return nothing
end

function Base.iterate(it::Chain, state)
    i, xs_state = state
    xs_state = iterate(it.xss[i], xs_state)
    while xs_state == nothing
        i += 1
        i > length(it.xss) && return nothing
        xs_state = iterate(it.xss[i])
    end
    return xs_state[1], (i, xs_state[2])
end

Base.iterate(x::AbstractHeterogeneousVector) = iterate(Chain(values(NamedTuple(x))))
function Base.iterate(x::AbstractHeterogeneousVector, state)
    iterate(Chain(values(NamedTuple(x))), state)
end
