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

# Performance

The field lookup is unrolled at compile time and does not allocate. The return type depends
on which field contains the requested index, so it is concrete only when all fields share
the same element type; otherwise it is a union of the field element types (e.g. quantities
with different units). Small unions are handled efficiently, but with many distinct element
types the result has to be boxed.

**For performance-critical code, prefer named field access over integer indexing:**

- `v[1]` — Scans the fields to find the index; type-stable only for a uniform element type
- `v.field[1]` — Always type-stable and direct

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
function Base.getindex(hv::AbstractHeterogeneousVector, idx::Int)
    return _getindex_fields(_fields(hv), hv, idx, 0)
end

# The flattened-index lookups below recurse over the tuple of fields, which the compiler
# unrolls: each field gets a branch specialized to its concrete type, and only the choice of
# branch happens at runtime. `offset` is the number of elements in the preceding fields.
@inline _getindex_fields(::Tuple{}, hv, idx, offset) = throw(BoundsError(hv, idx))
@inline function _getindex_fields(fields::Tuple, hv, idx, offset)
    field = first(fields)
    n = _field_length(field)
    1 <= idx - offset <= n && return _field_element(field, idx - offset)
    return _getindex_fields(Base.tail(fields), hv, idx, offset + n)
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

# Performance

Like `getindex`, the field lookup is unrolled at compile time and does not allocate, but it
has to scan the fields to find the index. Prefer **named field assignment** in
performance-critical code:

- `v[1] = x` — Scans the fields to find the index
- `v.field[1] = x` — Direct (preferred for performance)

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
function Base.setindex!(hv::AbstractHeterogeneousVector, val, idx::Int)
    _setindex_fields!(_fields(hv), hv, val, idx, 0)
    return val
end

@inline _setindex_fields!(::Tuple{}, hv, val, idx, offset) = throw(BoundsError(hv, idx))
@inline function _setindex_fields!(fields::Tuple, hv, val, idx, offset)
    field = first(fields)
    n = _field_length(field)
    1 <= idx - offset <= n && return _set_field_element!(field, val, idx - offset)
    return _setindex_fields!(Base.tail(fields), hv, val, idx, offset + n)
end

_fields(hv::AbstractHeterogeneousVector) = values(NamedTuple(hv))

_field_length(field::Ref) = 1
_field_length(field::AbstractArray) = length(field)

# Element `j` (1-based, in linear order) of a field; callers guarantee `1 <= j <= _field_length(field)`.
# Offsetting from `firstindex` also supports arrays with non-standard indices.
@inline _field_element(field::Ref, j) = field[]
@inline _field_element(field::AbstractArray, j) = @inbounds field[firstindex(field) + j - 1]
@inline _set_field_element!(field::Ref, val, j) = _set_value!(field, val)
@inline _set_field_element!(field::AbstractArray, val, j) = (@inbounds field[firstindex(field) + j - 1] = val)

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
#
# The state `(field_index, element_index)` is always a `Tuple{Int, Int}`, and the fields are
# visited by compile-time recursion (see `_getindex_fields`), so iteration does not allocate.
# `k` is the index of the first field in `fields`.
@inline _iterate_fields(::Tuple{}, fi, j, k) = nothing
@inline function _iterate_fields(fields::Tuple, fi, j, k)
    if fi == k
        field = first(fields)
        j <= _field_length(field) && return _field_element(field, j), (fi, j + 1)
        fi, j = fi + 1, 1  # field exhausted (or empty): continue with the next one
    end
    return _iterate_fields(Base.tail(fields), fi, j, k + 1)
end

Base.iterate(hv::AbstractHeterogeneousVector) = iterate(hv, (1, 1))

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

# Performance
Iteration does not allocate. The element type is concrete when all fields share the same
element type, and otherwise a union of the field element types (e.g. quantities with
different units). Small unions are handled efficiently, but with many distinct element types
the elements have to be boxed; field-wise operations (broadcasting, `sum`, `mapreduce`,
`any`, `all`) avoid this entirely.

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
function Base.iterate(hv::AbstractHeterogeneousVector, state)
    fi, j = state
    return _iterate_fields(_fields(hv), fi, j, 1)
end
