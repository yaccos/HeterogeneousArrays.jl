# Field-wise reductions over a CollectionVector. The function sees the elements of each
# field through the field view.

"""
    mapreduce(f, op, x::CollectionVector; dims = :, init) -> value

Apply `f` to every element of every field of `x` and reduce the results with `op`.

Elements are read through the field views, so `sum(abs, z)` adds the moduli of the elements
of a complex field rather than of their raw parts. A scalar field contributes `f(value)`, an
array field `mapreduce(f, op, field)`, and the field results are combined left to right with
`op`. If given, `init` is used once, before the first field. `sum`, `prod`, `maximum`,
`minimum`, `extrema`, `reduce`, `count` and `LinearAlgebra.norm` route through this method.

Fields with different units combine as any quantities do: `maximum` over fields in `m` and
`cm` converts, `sum` over fields in `m` and `s` throws a `Unitful.DimensionError`.

Only `dims = :` is supported. Any other `dims` throws an `ArgumentError`.
"""
function Base.mapreduce(f::F, op::OP, x::CollectionVector{T, S};
        dims::D = :, init = NoInit()) where {F, OP, T, S, D}
    # Note: the type parameters force specialization on `f`, `op` and `dims`
    dims === (:) || throw(unsupported_dims(dims))
    return reducefields(f, op, init, x, valnames(Val(keys(S))))
end

function unsupported_dims(dims)
    ArgumentError("dims = $dims is not supported for a CollectionVector; " *
                  "reductions are over all elements of all fields (dims = :)")
end

# Accumulator before the first field when no `init` is given
struct NoInit end
@inline combine(op, ::NoInit, r) = r
@inline combine(op, acc, r) = op(acc, r)

@inline reducefields(f, op, acc, x, ::Tuple{}) = acc
@inline function reducefields(f::F, op::OP, acc, x, names::Tuple) where {F, OP}
    acc = combine(op, acc, reducefield(f, op, x, first(names)))
    return reducefields(f, op, acc, x, Base.tail(names))
end
# Scalar field: `f` of the value. Array field: Base's `mapreduce` over the field view.
# Fields are never empty, so `mapreduce` needs no `init`.
@inline function reducefield(f::F, op::OP, x, ::Val{name}) where {F, OP, name}
    field = getproperty(x, name)
    return field isa AbstractArray ? mapreduce(f, op, field) : f(field)
end

"""
    any(f, x::CollectionVector; dims = :) -> Bool
    all(f, x::CollectionVector; dims = :) -> Bool

Test whether `f` returns `true` for any (all) elements of the fields of `x`.

Elements are read through the field views, so `any(iszero, z)` tests the elements of a
complex field rather than their raw parts. The fields are tested in order and the test stops
at the first field that decides the result. `f` must return a `Bool`.

Only `dims = :` is supported. Any other `dims` throws an `ArgumentError`.
`any(x)` and `all(x)` without a predicate are Base's methods, which test the storage slots.
"""
function Base.any(f::F, x::CollectionVector{T, S}; dims::D = :) where {
        F <: Function, T, S, D}
    dims === (:) || throw(unsupported_dims(dims))
    return anyfields(f, x, valnames(Val(keys(S))))
end

function Base.all(f::F, x::CollectionVector{T, S}; dims::D = :) where {
        F <: Function, T, S, D}
    dims === (:) || throw(unsupported_dims(dims))
    return allfields(f, x, valnames(Val(keys(S))))
end

@inline anyfields(f, x, ::Tuple{}) = false
@inline function anyfields(f::F, x, names::Tuple) where {F}
    anyfield(f, x, first(names)) || anyfields(f, x, Base.tail(names))
end
@inline function anyfield(f::F, x, ::Val{name}) where {F, name}
    field = getproperty(x, name)
    return field isa AbstractArray ? any(f, field) : f(field)
end

@inline allfields(f, x, ::Tuple{}) = true
@inline function allfields(f::F, x, names::Tuple) where {F}
    allfield(f, x, first(names)) && allfields(f, x, Base.tail(names))
end
@inline function allfield(f::F, x, ::Val{name}) where {F, name}
    field = getproperty(x, name)
    return field isa AbstractArray ? all(f, field) : f(field)
end
