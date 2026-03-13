module HeterogeneousArrays
"""
    HeterogeneousArrays

A Julia package for efficiently storing and operating on heterogeneous data with type-stable broadcasting.

The primary type is [`HeterogeneousVector`](@ref), which allows combining different concrete types 
(including quantities with units) into a single broadcastable vector.

# Features
- Type-stable broadcasting operations
- Support for Unitful quantities with mixed units
- Efficient array-like interface
- Named field access for clarity

# Example
```jldoctest
julia> using HeterogeneousArrays, Unitful

julia> v = HeterogeneousVector(position = 3.0u"m", time = 5.0u"s", count = 42)

julia> v.position
3.0 m

julia> 2.0 .* v .+ v  # Type-stable broadcasting
"""

using Unitful: Unitful
using RecursiveArrayTools: RecursiveArrayTools
import Base: NamedTuple

export AbstractHeterogeneousVector, HeterogeneousVector

# Include components in logical order
include("types.jl")
include("indexing.jl")
include("allocation.jl")
include("broadcasting.jl")

end # module
