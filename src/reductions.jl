# Field-wise reductions: recurse over the fields at compile time instead of iterating
# element-wise, so each field is reduced with a call specialized to its type.

# Accumulator without a value yet (no `init` and only empty fields so far)
struct _NoValue end

@inline _combine(op, ::_NoValue, x) = x
@inline _combine(op, acc, x) = op(acc, x)

@inline _mapreduce_field(f, op, acc, field::Ref) = _combine(op, acc, f(field[]))
@inline function _mapreduce_field(f::F, op::OP, acc, field::AbstractArray) where {F, OP}
    isempty(field) ? acc : _combine(op, acc, mapreduce(f, op, field))
end

@inline _mapreduce_fields(f, op, acc, ::Tuple{}) = acc
@inline function _mapreduce_fields(f::F, op::OP, acc, fields::Tuple) where {F, OP}
    _mapreduce_fields(f, op, _mapreduce_field(f, op, acc, first(fields)), Base.tail(fields))
end

"""
    Base.mapreduce(f, op, hv::AbstractHeterogeneousVector; dims = :, init)

Apply `f` to every element of `hv` and reduce the results with `op`.

The reduction is performed field by field, so `sum`, `maximum` etc. do not allocate.
If given, `init` is used once as the starting value.

# Examples
```jldoctest
julia> using HeterogeneousArrays

julia> v = HeterogeneousVector(a = [1.0, 2.0], b = 3.0);

julia> mapreduce(abs2, +, v)
14.0

julia> sum(v)
6.0
```
"""
function Base.mapreduce(
        f::F, op::OP, hv::AbstractHeterogeneousVector;
        dims::D = :, init = _NoValue()
) where {F, OP, D}
    # The type parameters force specialization on `f`, `op` and `dims`
    dims === (:) || return _generic_mapreduce(f, op, hv, init; dims)
    acc = _mapreduce_fields(f, op, init, _fields(hv))
    # All fields empty: use Base's behavior for empty collections
    acc isa _NoValue && return _generic_mapreduce(f, op, hv, init)
    return acc
end

function _generic_mapreduce(f, op, hv::AbstractHeterogeneousVector, ::_NoValue; kw...)
    invoke(mapreduce, Tuple{Any, Any, AbstractArray}, f, op, hv; kw...)
end
function _generic_mapreduce(f, op, hv::AbstractHeterogeneousVector, init; kw...)
    invoke(mapreduce, Tuple{Any, Any, AbstractArray}, f, op, hv; init, kw...)
end

_any_field(f, field::Ref) = f(field[])
_any_field(f::F, field::AbstractArray) where {F} = any(f, field)
_all_field(f, field::Ref) = f(field[])
_all_field(f::F, field::AbstractArray) where {F} = all(f, field)

@inline _any_fields(f, ::Tuple{}) = false
@inline _any_fields(f::F, fields::Tuple) where {F} = _any_field(f, first(fields)) ||
                                                     _any_fields(f, Base.tail(fields))
@inline _all_fields(f, ::Tuple{}) = true
@inline _all_fields(f::F, fields::Tuple) where {F} = _all_field(f, first(fields)) &&
                                                     _all_fields(f, Base.tail(fields))

"""
    Base.any(f, hv::AbstractHeterogeneousVector; dims = :)
    Base.all(f, hv::AbstractHeterogeneousVector; dims = :)

Test whether `f` returns `true` for any (all) elements of `hv`.

The test is performed field by field and does not allocate. `f` must return a `Bool`.

# Examples
```jldoctest
julia> using HeterogeneousArrays

julia> v = HeterogeneousVector(a = [1.0, 2.0], b = NaN);

julia> any(isnan, v)
true

julia> all(isfinite, v)
false
```
"""
function Base.any(f::F, hv::AbstractHeterogeneousVector; dims::D = :) where {
        F <: Function, D}
    dims === (:) || return invoke(any, Tuple{Function, AbstractArray}, f, hv; dims)
    return _any_fields(f, _fields(hv))
end

function Base.all(f::F, hv::AbstractHeterogeneousVector; dims::D = :) where {
        F <: Function, D}
    dims === (:) || return invoke(all, Tuple{Function, AbstractArray}, f, hv; dims)
    return _all_fields(f, _fields(hv))
end

Base.any(hv::AbstractHeterogeneousVector; dims = :) = any(identity, hv; dims)
Base.all(hv::AbstractHeterogeneousVector; dims = :) = all(identity, hv; dims)
