using HeterogeneousArrays
using Test
using Unitful

@testset "HeterogeneousArrays.jl" begin
    @testset "Array Interface & Metadata" begin
        x = HeterogeneousVector(a = [1, 2, 3], b = 4.5)

        @testset "Indexing" begin
            @test x[1] == 1
            @test x[4] == 4.5
            @test_throws BoundsError x[5]
        end

        @testset "Iteration" begin
            v = HeterogeneousVector(d = [3.1u"m", 4.2u"m"], t = 5.0u"s")
            collected = collect(v)
            @test length(collected) == 3
            @test collected[1] == 3.1u"m"
            @test collected[3] == 5.0u"s"
        end

        @testset "Metadata" begin
            @test summary(x) == "$(typeof(x)) with members:"
            @test propertynames(x) == (:a, :b)
        end
    end

    @testset "State & Mutability" begin
        @testset "Direct Access" begin
            x = HeterogeneousVector(a = [1, 2, 3], b = 4.5)
            x.b = 7.8
            x.a[2] = 42
            @test x.b == 7.8
            @test x.a[2] == 42
            @test_throws ErrorException x.missing_field = 10
        end

        @testset "Copying & Identity" begin
            x = HeterogeneousVector(a = [1, 2, 3], b = 4.5)
            y = copy(x)
            @test y == x
            @test y !== x
            @test y.a !== x.a # Ensure deep copy of segments

            z = zero(x)
            copyto!(z, x)
            @test z == x
        end
    end

    @testset "Unit Logic & Conversions" begin
        x = HeterogeneousVector(pos = [1.0u"m", 2.0u"m"], time = 10.0u"s")

        @testset "Compatible Assignment" begin
            x.time = 1.0u"minute"
            # Numerical vs Physical equality
            @test x.time ≈ 60.0u"s"
            @test x.time == 1.0u"minute"
            @test Unitful.ustrip(x.time) ≈ 60.0

            new_pos = [1500.0u"mm", 3000.0u"mm"]
            x.pos = new_pos
            @test Unitful.ustrip.(x.pos) ≈ [1.5, 3.0]
        end

        @testset "Dimension Mismatch" begin
            @test_throws Exception x.pos = [1.0u"s", 2.0u"s"]
            @test_throws Exception x.time = 42.0
        end
    end

    @testset "Broadcasting Mechanics" begin
        x = HeterogeneousVector(a = 1.0u"m", b = [2.0u"m", 3.0u"m"])
        y = HeterogeneousVector(a = 4.0u"m", b = [5.0u"m", 6.0u"m"])

        @testset "Out-of-Place (Allocation)" begin
            res = x .+ y
            @test res isa HeterogeneousVector
            @test res.a == 5.0u"m"
            @test res.b == [7.0u"m", 9.0u"m"]
            @test res !== x
        end

        @testset "In-Place (Mutation)" begin
            original_b_ptr = pointer(x.b)
            x .= 2.0 .* x .+ y

            @test x.a ≈ 6.0u"m"
            @test x.b ≈ [9.0u"m", 12.0u"m"]
            @test pointer(x.b) == original_b_ptr # Verify zero-allocation update
        end

        @testset "Compound & Mixed Mutation" begin
            v = HeterogeneousVector(val = [10.0, 20.0], scale = 2.0)
            v .+= 5.0
            @test v.val == [15.0, 25.0]
            @test v.scale == 7.0

            v .= 100.0
            @test all(v.val .== 100.0) && v.scale == 100.0
        end

        @testset "Fusion & Complex Trees" begin
            v1 = HeterogeneousVector(a = 1.0, b = [2.0])
            v2 = HeterogeneousVector(a = 10.0, b = [20.0])
            res = @. exp(v1) + log(v2) * 2.0
            @test res.a ≈ exp(1.0) + log(10.0) * 2.0
        end
    end

    @testset "Allocation Logic (similar)" begin
        x = HeterogeneousVector(pos = [1.0u"m"], id = [1])

        @testset "Helpers & Zero-Initialization" begin
            # Test Ref
            r_sim = HeterogeneousArrays._similar_field(Ref(10), Float64)
            @test r_sim isa Ref{Float64}
            @test r_sim[] == 0.0
        end

        @testset "Type Overrides" begin
            y_float = similar(x, Float64)
            @test y_float isa HeterogeneousVector
            @test eltype(y_float.id) == Float64
            @test !(y_float isa Array) # Confirm no collapse to standard Array
        end
    end
end

using BenchmarkTools

function compute_inplace!(d, src1, src2)
    d .= src1 .+ src2
    return nothing
end

@testset "Performance & Static Dispatch Validation" begin
    x = HeterogeneousVector(a = 1.0u"m", b = [2.0u"m", 3.0u"m"])
    y = HeterogeneousVector(a = 4.0u"m", b = [5.0u"m", 6.0u"m"])

    @testset "Type Stability & Inference (Concrete)" begin
        type_a = typeof(x.a)
        type_b = typeof(x.b)
        type_hv = typeof(x)

        get_a(v) = v.a
        get_b(v) = v.b

        # 1. Field Access: Verify exact concrete types
        # If the compiler inferred Vector{Any}, these will fail.
        @test (@inferred get_a(x)) isa type_a
        @test (@inferred get_b(x)) isa type_b

        # 2. Broadcast Inference: Verify the container and its full type parameters
        # This checks that Names, T, and S (the NamedTuple storage) are all preserved.
        add_vecs(v1, v2) = v1 .+ v2
        @test (@inferred add_vecs(x, y)) isa type_hv

        # 3. Complex Fusion: The most rigorous test
        # We check if the result of math operations is still fully inferable
        # to the exact same HeterogeneousVector type.
        f_fused(v1, v2) = @. exp(v1 / 1.0u"m") + v2 / 1.0u"m"

        # We use 'typeof' on a manual execution to get the target for the inference test
        expected_result_type = typeof(f_fused(x, y))
        @test (@inferred f_fused(x, y)) isa expected_result_type
    end

    @testset "Concrete Type Structural Validation" begin
        x = HeterogeneousVector(a = 1.0u"m", b = [2.0u"m", 3.0u"m"])

        ConcreteVectorType = Vector{Unitful.Quantity{Float64, Unitful.𝐋, typeof(u"m")}}

        # Use a local function to avoid the 'Global Scope' inference trap
        get_b(hv) = hv.b

        @test (@inferred get_b(x)) isa ConcreteVectorType

        get_a_raw(hv) = getfield(NamedTuple(hv), :a)
        ConcreteRefType = Base.RefValue{Unitful.Quantity{Float64, Unitful.𝐋, typeof(u"m")}}
        @test (@inferred get_a_raw(x)) isa ConcreteRefType
    end

    @testset "Zero-Allocation In-Place Updates" begin
        # Pre-allocate destination to test memory efficiency.
        dest = zero(x)
        compute_inplace!(dest, x, y) # Warmup to ensure JIT compilation is finished.

        # We interpolate ($) variables so the benchmark doesn't include the cost
        # of looking up global variable names. We want to measure only the math.
        b = @benchmarkable compute_inplace!($dest, $x, $y)
        res = run(b)

        # THE GOLD STANDARD TEST: 
        # res.allocs == 0 proves that no 'boxing' or 'runtime dispatch' occurred.
        # 'Boxing' happens when Julia isn't sure of a type and has to wrap data 
        # in a generic box at runtime, which costs memory and time.
        # 0 allocations means the code is running as fast as C.
        @test res.allocs == 0
    end
end

@testset "Advanced Allocation Logic (Multi-Type similar)" begin
    # pos is Vector{Length}, id is Vector{Int}
    x = HeterogeneousVector(pos = [1.0, 2.0]u"m", id = [10, 20])

    @testset "Variadic Type Overrides" begin
        y = @inferred similar(x, Float32, Int16)
        @test eltype(y.pos) === Float32
        @test eltype(y.id) === Int16
        @test y isa HeterogeneousVector
    end

    @testset "Unitful Type Overrides" begin
        # Get the dimension safely
        L_dim = dimension(u"m")
        # Define the exact Quantity type we expect
        # We use u"m" directly in the type construction for clarity
        TargetUnitType = Quantity{Float64, L_dim, typeof(u"m")}

        y = @inferred similar(x, TargetUnitType, Float64)

        @test y isa HeterogeneousVector
        # Test the eltype directly (this is usually more stable)
        @test eltype(y.pos) === TargetUnitType
        @test eltype(y.id) === Float64

        # To avoid the 'showrep' bug if this fails, we check the unit 
        # by converting it to a string or comparing to a simple unit.
        @test unit(y.pos[1]) === u"m"
    end

    @testset "Uniform Override" begin
        y = @inferred similar(x, Float64)
        @test eltype(y.pos) === Float64
        @test eltype(y.id) === Float64
    end

    @testset "Error Handling" begin
        # Provide 3 types for 2 fields -> Should throw DimensionMismatch
        @test_throws DimensionMismatch similar(x, Float64, Int, Bool)
    end
end

@testset "Zero-Initialization" begin
    # pos is Vector{Length}, id is Vector{Int}
    x = HeterogeneousVector(pos = [1.0, 2.0]u"m", id = [10, 20])

    # Generate the zero representation
    z = @inferred zero(x)

    @test z isa HeterogeneousVector
    @test propertynames(z) == (:pos, :id)

    # Check numerical values
    @test all(z.pos .== 0.0u"m")
    @test all(z.id .== 0)

    # Ensure types and units are strictly preserved
    @test eltype(z.pos) === eltype(x.pos)
    @test eltype(z.id) === eltype(x.id)

    # Ensure it is a new allocation (not a view/alias)
    @test z.pos !== x.pos
    @test z.id !== x.id
end

@testset "Keyword-based similar" begin
    x = HeterogeneousVector(pos = [1.0, 2.0]u"m", id = [10, 20])

    # Override only one field
    y = similar(x, id = Float32)
    @test eltype(y.id) === Float32
    @test eltype(y.pos) === eltype(x.pos) # Preserved

    # Override multiple fields in arbitrary order
    z = similar(x, id = Float64, pos = Float64)
    @test eltype(z.id) === Float64
    @test eltype(z.pos) === Float64
end

@testset "Reference vs Copy Behavior" begin
    original_array = [1.0, 2.0, 3.0]
    original_scalar = 42.0

    v = HeterogeneousVector(vec = original_array, scalar = original_scalar)

    @testset "Array Referencing" begin
        # Modifying original should affect v
        original_array[1] = 99.0
        @test v.vec[1] == 99.0

        # Modifying v should affect original
        v.vec[2] = -7.0
        @test original_array[2] == -7.0

        # Verify they share the same memory location
        @test pointer(v.vec) == pointer(original_array)
    end

    @testset "Scalar Wrapping" begin
        # Scalars are wrapped in Ref, so the original value (being immutable) 
        # cannot be linked, but we should verify the Ref contains the value.
        @test v.scalar == 42.0

        v.scalar = 100.0
        @test original_scalar == 42.0 # Original local variable remains unchanged
    end
end

@testset "Nested HeterogeneousVectors" begin
    inner = HeterogeneousVector(x = [1.0, 2.0], y = 3.0)
    outer = HeterogeneousVector(sub = inner, bulk = [10.0, 20.0])

    @testset "Nested Access" begin
        @test outer.sub.x[1] == 1.0
        @test outer.sub.y == 3.0
    end

    @testset "Nested Broadcasting" begin
        # Test addition across nested layers
        res = outer .+ 1.0

        @test res isa HeterogeneousVector
        @test res.sub isa HeterogeneousVector

        # Verify recursion reached the bottom
        @test res.sub.x == [2.0, 3.0]
        @test res.sub.y == 4.0
        @test res.bulk == [11.0, 21.0]
    end
end

function compute_nested!(d, o1, o2)
    @. d = o1 + o2 * 2.0
    return nothing
end

@testset "Nested HeterogeneousVectors" begin
    # Construct a 2-level nested structure
    # Level 2 (Inner)
    inner1 = HeterogeneousVector(a = [1.0, 2.0], b = 3.0)
    inner2 = HeterogeneousVector(a = [4.0, 5.0], b = 6.0)

    # Level 1 (Outer)
    outer1 = HeterogeneousVector(sub = inner1, val = 10.0)
    outer2 = HeterogeneousVector(sub = inner2, val = 20.0)
    dest = zero(outer1)

    compute_nested!(dest, outer1, outer2)
    @test dest.sub.a == [9.0, 12.0]
    @test dest.sub.b == 15.0
    @test dest.val == 50.0
end

# @testset "Deep Nesting" begin
#     # Create a 10-level deep nested structure
#     v = HeterogeneousVector(x = 1.0)
#     for i in 1:10
#         v = HeterogeneousVector(inner = v, val = Float64(i))
#     end
#     
#     # If this is inferred as a concrete type, recursion is working!
#     @test !(@inferred(v .+ 1.0) isa Any)
# end
