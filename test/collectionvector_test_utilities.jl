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

HeterogeneousArrays.is_storable(::Type{<:TaggedNumber}) = true
HeterogeneousArrays.rawtype(::Type{TaggedNumber{T, U}}) where {T, U} = T
HeterogeneousArrays.field_type(::TaggedNumber{T, U}) where {T, U} = U
# Constructor from another TaggedNumber. CollectionVector stores elements with `convert`, and
# for a `Number` subtype Base's `convert(T, x::Number)` calls this `T(x)`.
function TaggedNumber{T, U}(q::TaggedNumber{S, V}) where {T, U, S, V}
    V === U || error("unit mismatch: $V vs $U")
    TaggedNumber{T, U}(convert(T, q.value))
end
HeterogeneousArrays.elementtype(::Type{T}, u::Symbol) where {T} = TaggedNumber{T, u}

# Structs of three Float64 that implement the element API. `Triple` is stored in Float64
# slots. `MislabeledTriple` declares Float32 storage, which does not match its components.
struct Triple
    a::Float64
    b::Float64
    c::Float64
end
struct TripleField end
HeterogeneousArrays.is_storable(::Type{Triple}) = true
HeterogeneousArrays.rawtype(::Type{Triple}) = Float64
HeterogeneousArrays.field_type(::Triple) = TripleField()
HeterogeneousArrays.elementtype(::Type, ::TripleField) = Triple

struct MislabeledTriple
    a::Float64
    b::Float64
    c::Float64
end
struct MislabeledTripleField end
HeterogeneousArrays.is_storable(::Type{MislabeledTriple}) = true
HeterogeneousArrays.rawtype(::Type{MislabeledTriple}) = Float32
HeterogeneousArrays.field_type(::MislabeledTriple) = MislabeledTripleField()
HeterogeneousArrays.elementtype(::Type, ::MislabeledTripleField) = MislabeledTriple

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
