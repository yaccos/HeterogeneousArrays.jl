# Types and methods used by test_collectionvector.jl. Right now they live in a
# separate file because I didn't find how to make `struct` and `import` work inside a
# `@testset` block.
import ForwardDiff
import DynamicQuantities

# A Number that is not a quantity we know: must be rejected by the element gate.
struct FakeQuantity{T} <: Number
    value::T
    unit::Symbol
end

# A third-party "quantity" type that opts into the element API from outside the package. Its
# unit is a type parameter, so a value has the memory layout of one raw number, which the
# `reinterpret` view requires.
struct TaggedNumber{T, U} <: Number
    value::T
end
TaggedNumber(x, u::Symbol) = TaggedNumber{typeof(x), u}(x)
# Must define `==` for the tests
Base.:(==)(a::TaggedNumber, b::TaggedNumber) = typeof(a) === typeof(b) && a.value == b.value

HeterogeneousArrays.isstorable(::Type{<:TaggedNumber}) = true
HeterogeneousArrays.rawtype(::Type{TaggedNumber{T, U}}) where {T, U} = T
HeterogeneousArrays.field_type(::TaggedNumber{T, U}) where {T, U} = U
function HeterogeneousArrays.strip_type(u::Symbol, q::TaggedNumber{T, V}) where {T, V}
    V === u || error("unit mismatch: $V vs $u")
    q.value
end
HeterogeneousArrays.attach_type(u::Symbol, x) = TaggedNumber{typeof(x), u}(x)
HeterogeneousArrays.elementtype(::Type{T}, u::Symbol) where {T} = TaggedNumber{T, u}

# A third-party subtype of `Unitful.AbstractQuantity`, to check that the Unitful methods of the
# element API are not restricted to `Unitful.Quantity`. Unitful gives such a type `unit`,
# `dimension` and the arithmetic for free, but `uconvert` (used by `ustrip(u, q)`) is only
# defined for `Quantity`, so the subtype must provide it.
struct FrozenQuantity{T, D, U} <: Unitful.AbstractQuantity{T, D, U}
    val::T
end
FrozenQuantity(x::Number, u::Unitful.Units) = FrozenQuantity{typeof(x), Unitful.dimension(u), typeof(u)}(x)
Unitful.uconvert(u::Unitful.Units, q::FrozenQuantity) = Unitful.uconvert(u, Unitful.Quantity(q.val, Unitful.unit(q)))
Unitful.ustrip(q::FrozenQuantity) = q.val
