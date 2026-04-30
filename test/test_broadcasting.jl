@testset "Unit Logic & Conversions" begin
    x = HeterogeneousVector(pos = [1.0u"m", 2.0u"m"], time = 10.0u"s")

    @testset "Compatible Assignment" begin
        x.time = 1.0u"minute"
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
        @test pointer(x.b) == original_b_ptr
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

@testset "Boundary cases with ordinary arrays" begin
    @testset "Homogeneous array mutated in-place over a heterogeneous broadcast" begin
       v = HeterogeneousVector(pos = [1.0u"m", 2.0u"m"], time = 10.0u"s")
       v_projection = HeterogeneousVector(pos = [1.5u"m", 3.0u"m"], time = 5.0u"s")
       residuals = Vector{Float64}(undef, length(v))
       residuals .= ustrip.(v .- v_projection)
       @test residuals ≈ [-0.5, -1.0, 5.0]
    end
    @testset "Mixed allocating broadcast with both homogeneous and heterogeneous arguments" begin
        res_broadcasted = [1.0, 2.0, 3.0] .* HeterogeneousVector(length=1.0u"m", mass=1.0u"kg", time=1.0u"s")
        res_expected = HeterogeneousVector(length=1.0u"m", mass = 2.0u"kg", time = 3.0u"s")
        @test res_broadcasted ≈ res_expected
    end
end


@testset "Multi-dimensional array broadcast is rejected" begin
        hv = HeterogeneousVector(a=1.0u"m", b=2.0u"m")
        mat = reshape([1.0, 2.0], 1, 2)
        @test_throws ArgumentError mat .* hv
end
