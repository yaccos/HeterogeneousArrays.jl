@testset "Reductions" begin
    x = HeterogeneousVector(a = 1.0, b = [2.0, -3.0], c = 4.0)
    reference = collect(x)

    @testset "Results match the flattened vector" begin
        @test sum(x) == sum(reference)
        @test prod(x) == prod(reference)
        @test maximum(x) == maximum(reference)
        @test minimum(abs, x) == minimum(abs, reference)
        @test extrema(x) == extrema(reference)
        @test count(>(0), x) == count(>(0), reference)
        @test mapreduce(abs2, +, x) == mapreduce(abs2, +, reference)
        @test any(<(0), x) && !any(>(10), x)
        @test all(isfinite, x) && !all(>(0), x)
        @test any(HeterogeneousVector(a = [false], b = true))
        @test !all(HeterogeneousVector(a = [true], b = false))
    end

    @testset "init is applied once" begin
        @test mapreduce(abs2, +, x; init = 100.0) == 100.0 + sum(abs2, reference)
        @test sum(x; init = 10.0) == 10.0 + sum(reference)
    end

    @testset "Non-commutative op keeps element order" begin
        s = HeterogeneousVector(a = "x", b = ["y", "z"], c = "w")
        @test mapreduce(identity, *, s) == "xyzw"
        @test mapreduce(identity, *, s; init = ">") == ">xyzw"
    end

    @testset "Empty array fields" begin
        leading = HeterogeneousVector(a = Float64[], b = 2.0, c = [3.0, 4.0])
        @test sum(leading) == 9.0
        @test maximum(leading) == 4.0
        middle = HeterogeneousVector(a = 1.0, b = Float64[], c = 3.0)
        @test mapreduce(abs2, +, middle) == 10.0
        @test any(==(3.0), middle) && all(>(0), middle)

        all_empty = HeterogeneousVector(a = Float64[], b = Float64[])
        @test sum(all_empty) == 0.0
        @test sum(all_empty; init = 1.0) == 1.0
        # Same error as Base for an empty collection (MethodError before Julia 1.11, ArgumentError after)
        base_error = try
            maximum(Float64[])
        catch e
            typeof(e)
        end
        @test_throws base_error maximum(all_empty)
        @test !any(isnan, all_empty)
        @test all(isnan, all_empty)
    end

    @testset "Nested and unitful vectors" begin
        nested = HeterogeneousVector(sub = HeterogeneousVector(p = [1.0, 2.0], q = 3.0), r = 4.0)
        @test sum(nested) == 10.0
        @test any(==(3.0), nested) && !any(==(5.0), nested)

        u = HeterogeneousVector(d = [1.0u"m", 2.0u"m"], t = 3.0u"s")
        @test mapreduce(v -> abs2(ustrip(v)), +, u) == 14.0
        @test all(v -> isfinite(ustrip(v)), u)
        @test sum(HeterogeneousVector(a = 1.0u"m", b = [2.0u"m"])) == 3.0u"m"
    end

    @testset "dims keyword" begin
        @test sum(x; dims = 1) == [sum(reference)]
        @test any(<(0), x; dims = 1) == [true]
        @test all(<(0), x; dims = 1) == [false]
    end

    # Regression: reductions used to iterate element-wise over the type-unstable `Chain`
    # iterator, allocating several times per element (e.g. in ODE solver norms and NaN checks)
    @testset "Type stability & zero allocations" begin
        hv = HeterogeneousVector(A = 1.0, T = 1.0, AT = 1.0, B = rand(101))
        u = HeterogeneousVector(A = 1.0u"m", T = 2.0u"s", B = rand(101))
        unitless_abs2(v) = abs2(ustrip(v))
        unitless_isnan(v) = isnan(ustrip(v))

        @test (@inferred sum(hv)) isa Float64
        @test (@inferred mapreduce(abs2, +, hv; init = 0.0)) isa Float64
        @test (@inferred mapreduce(unitless_abs2, +, u; init = 0.0)) isa Float64
        @test (@inferred any(isnan, hv)) isa Bool
        @test (@inferred all(unitless_isnan, u)) isa Bool

        @test run(@benchmarkable sum($hv)).allocs == 0
        @test run(@benchmarkable maximum($hv)).allocs == 0
        @test run(@benchmarkable prod($hv)).allocs == 0
        @test run(@benchmarkable mapreduce(abs2, +, $hv; init = 0.0)).allocs == 0
        @test run(@benchmarkable mapreduce($unitless_abs2, +, $u; init = 0.0)).allocs == 0
        @test run(@benchmarkable any(isnan, $hv)).allocs == 0
        @test run(@benchmarkable all(isfinite, $hv)).allocs == 0
        @test run(@benchmarkable any($unitless_isnan, $u)).allocs == 0
    end
end
