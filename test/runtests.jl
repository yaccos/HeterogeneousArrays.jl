using HeterogeneousArrays
using Test
using Unitful
using BenchmarkTools
using OffsetArrays

@testset "HeterogeneousArrays.jl" begin
    include("test_interface.jl")
    include("test_iteration.jl")
    include("test_broadcasting.jl")
    include("test_allocation.jl")
    include("test_nesting.jl")
    include("test_reductions.jl")
    include("test_performance.jl")
    include("test_printing.jl")
end
