Base.pairs(hv::AbstractHeterogeneousVector{T, S}) where {S, T} = pairs(NamedTuple(hv))

"""
    Base.getindex(hv::AbstractHeterogeneousVector, idx::Int)

Index into the flattened view of the HeterogeneousVector.

The vector presents a flat 1-based indexed interface where indices are mapped sequentially 
across all fields in order. Scalar fields count as a single element, and array fields contribute 
their length to the total.

# Arguments
- `hv`: The HeterogeneousVector to index
- `idx::Int`: The 1-based index into the flattened view

# Returns
The element at the given flattened index

# Errors
- Throws `BoundsError` if `idx` is outside the range `[1, length(hv)]`

# Performance Warning

This method is **not type-stable**. The return type depends on which field contains the 
requested index, and the compiler cannot determine this at compile time. This forces the 
return type to be a union of all possible field element types, preventing optimization.

**For performance-critical code, use named field access instead of integer indexing:**

- `v[1]` — Not type-stable (avoid in loops)
- `v.field[1]` — Type-stable (preferred for performance)

# Examples
```jldoctest
julia> using HeterogeneousArrays

julia> v = HeterogeneousVector(a = [1, 2, 3], b = 4.5);

julia> v[1]  # First element of field 'a'
1

julia> v[4]  # Field 'b' (scalar)
4.5

julia> v[5]  # Out of bounds
ERROR: BoundsError
```
"""
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

"""
    Base.setindex!(hv::AbstractHeterogeneousVector, val, idx::Int)

Assign a value at the flattened index in a HeterogeneousVector.

Index mapping follows the same flattened convention as `getindex`. For scalar fields, 
the new value replaces the wrapped value. For array fields, the element is updated in place.

# Arguments
- `hv`: The HeterogeneousVector to modify
- `val`: The new value to assign
- `idx::Int`: The 1-based index into the flattened view

# Returns
The value that was assigned

# Errors
- Throws `BoundsError` if `idx` is outside the range `[1, length(hv)]`

# Performance Warning

Like `getindex`, this method is **not type-stable** and should be avoided in 
performance-critical code. Use **named field assignment** instead:

- `v[1] = x` — Not type-stable (avoid in loops)
- `v.field[1] = x` — Type-stable (preferred for performance)

# Examples
```jldoctest
julia> using HeterogeneousArrays

julia> v = HeterogeneousVector(a = [1, 2, 3], b = 4.5);

julia> v[2] = 99
99

julia> v.a
3-element Vector{Int64}:
  1
 99
  3

julia> v[4] = 10.0
10.0

julia> v.b
10.0
```
"""
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

"""
    Base.length(hv::AbstractHeterogeneousVector) -> Int

Return the total length of the HeterogeneousVector as the sum of all field lengths.

Scalar fields (wrapped in `Ref`) contribute 1 to the total, and array fields contribute 
their full length. This is the length of the flattened view used for indexing.

# Returns
The total number of elements in the flattened representation

# Examples
```jldoctest
julia> using HeterogeneousArrays

julia> v = HeterogeneousVector(a = [1, 2, 3], b = 4.5, c = [10, 20]);

julia> length(v)  # 3 (from 'a') + 1 (from 'b') + 2 (from 'c')
6
```
"""
Base.length(hv::AbstractHeterogeneousVector) = sum(_field_length, NamedTuple(hv); init = 0)

"""
    Base.size(hv::AbstractHeterogeneousVector) -> Tuple

Return the size of the HeterogeneousVector as a 1-tuple of its total length.

This satisfies the `AbstractArray` interface by returning `(length(hv),)`, 
representing a 1-dimensional array.

# Returns
A tuple `(n,)` where `n = length(hv)`

# Examples
```jldoctest
julia> using HeterogeneousArrays

julia> v = HeterogeneousVector(x = [1, 2], y = 3.0);

julia> size(v)
(3,)
```
"""
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

"""
    Base.iterate(hv::AbstractHeterogeneousVector) -> Union{Tuple, Nothing}
    Base.iterate(hv::AbstractHeterogeneousVector, state) -> Union{Tuple, Nothing}

Iterate over all elements in the HeterogeneousVector using the flattened view.

The vector is traversed field-by-field in the order they are stored in the internal 
NamedTuple. Scalar fields yield a single value, and array fields yield each of their 
elements in sequence.

# Returns
- On first call: `(element, state)` or `nothing` if the vector is empty
- On subsequent calls with state: next `(element, state)` or `nothing` when exhausted

# Examples
```jldoctest
julia> using HeterogeneousArrays

julia> v = HeterogeneousVector(a = [1, 2], b = 3.0);

julia> for (i, element) in enumerate(v)
           println(i, ": ", element)
       end
1: 1
2: 2
3: 3.0
```
"""
function Base.iterate(x::AbstractHeterogeneousVector, state)
    iterate(Chain(values(NamedTuple(x))), state)
end
