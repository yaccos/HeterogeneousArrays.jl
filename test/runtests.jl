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

            # Test Array (Ensuring initialized memory)
            a_sim = HeterogeneousArrays._similar_field([1, 2], Float64)
            @test all(==(0.0), a_sim)
        end

        @testset "Type Overrides" begin
            y_float = similar(x, Float64)
            @test y_float isa HeterogeneousVector
            @test eltype(y_float.id) == Float64
            @test !(y_float isa Array) # Confirm no collapse to standard Array
        end
    end
end
