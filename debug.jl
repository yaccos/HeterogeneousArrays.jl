using HeterogeneousArrays
struct FakeQuantity{T} <: Number
    value::T
    unit::Symbol
end

a = [FakeQuantity(2.2, :kms), FakeQuantity(9.2, :kms)]
CollectionVector(a = a)
