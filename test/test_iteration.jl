@testset "Flat Iteration & Indexing" begin
    @testset "Order and empty fields" begin
        v = HeterogeneousVector(
            a = Float64[], b = 1.0, c = [2.0, 3.0], d = Float64[], e = 4.0)
        @test collect(v) == [1.0, 2.0, 3.0, 4.0]
        @test [x for x in v] == [1.0, 2.0, 3.0, 4.0]
        @test [v[i] for i in 1:length(v)] == [1.0, 2.0, 3.0, 4.0]
        @test isempty(collect(HeterogeneousVector(a = Float64[], b = Float64[])))
        @test iterate(HeterogeneousVector(a = Float64[])) === nothing
    end

    @testset "Bounds" begin
        v = HeterogeneousVector(a = [1, 2], b = 3.0)
        @test_throws BoundsError v[0]
        @test_throws BoundsError v[-1]
        @test_throws BoundsError v[4]
        @test_throws BoundsError (v[4] = 1.0)
    end

    @testset "setindex!" begin
        v = HeterogeneousVector(a = [1.0, 2.0], b = 3.0, c = [4.0])
        v[2] = 20.0
        v[3] = 30.0
        v[4] = 40.0
        @test v.a == [1.0, 20.0] && v.b == 30.0 && v.c == [40.0]
    end

    @testset "Arrays with non-standard indices" begin
        o = OffsetArray([10.0, 20.0, 30.0], -1:1)
        v = HeterogeneousVector(s = 1.0, o = o)
        @test collect(v) == [1.0, 10.0, 20.0, 30.0]
        @test v[2] == 10.0 && v[4] == 30.0
        v[3] = 99.0
        @test o[0] == 99.0
    end

    @testset "Matrix fields iterate in linear order" begin
        m = [1.0 3.0; 2.0 4.0]
        v = HeterogeneousVector(m = m, s = 5.0)
        @test collect(v) == [1.0, 2.0, 3.0, 4.0, 5.0]
        @test v[3] == 3.0
    end

    @testset "Nested vectors" begin
        inner = HeterogeneousVector(p = [1.0, 2.0], q = 3.0)
        outer = HeterogeneousVector(sub = inner, r = 4.0)
        @test collect(outer) == [1.0, 2.0, 3.0, 4.0]
        @test outer[3] == 3.0
        outer[3] = 30.0
        @test inner.q == 30.0
    end

    @testset "Unitful elements keep their units" begin
        v = HeterogeneousVector(x = 1.0u"m", t = 2.0u"s", B = [3.0u"kg", 4.0u"kg"], n = 5.0)
        elements = [x for x in v]
        @test elements == [1.0u"m", 2.0u"s", 3.0u"kg", 4.0u"kg", 5.0]
        @test typeof.(elements) == [typeof(1.0u"m"), typeof(2.0u"s"), typeof(3.0u"kg"),
            typeof(4.0u"kg"), Float64]
        @test v[2] === 2.0u"s"
        @test sum(ustrip(x) for x in v) == 15.0
        @test_throws Unitful.DimensionError sum(x for x in v)
    end

    # Regression: iteration used to go through a `Chain` iterator whose state changed type
    # between fields, boxing the state and the element for every element
    @testset "Type stability & zero allocations" begin
        hv = HeterogeneousVector(A = 1.0, T = 1.0, AT = 1.0, B = rand(101))
        u = HeterogeneousVector(A = 1.0u"m", T = 2.0u"s", B = rand(101))
        loop_sum(v) = (s = 0.0; for x in v
                s += ustrip(x)
            end; s)
        generator_sum(v) = sum(ustrip(x) for x in v)
        # Pattern of DiffEqBase's ODE_DEFAULT_NORM for unitful states
        zip_norm(v, t) = sqrt(sum(((x, _),) -> abs2(ustrip(x)), zip((y for y in v), Iterators.repeated(t))) /
                              length(v))
        index_sum(v) = (s = 0.0; for i in 1:length(v)
                s += ustrip(v[i])
            end; s)
        set_all!(v) = (for i in 1:length(v)
                v[i] = 1.0
            end; v)

        # `iterate` returns either `nothing` or a concrete (element, state) tuple
        @test (@inferred Nothing iterate(hv)) isa Tuple{Float64, Tuple{Int, Int}}
        @test (@inferred Nothing iterate(hv, (4, 101))) isa Tuple{Float64, Tuple{Int, Int}}
        @test (@inferred hv[50]) isa Float64

        for v in (hv, u)
            @test loop_sum(v) ≈ sum(ustrip, collect(v))
            @test run(@benchmarkable $loop_sum($v)).allocs == 0
            @test run(@benchmarkable $generator_sum($v)).allocs == 0
            @test run(@benchmarkable $zip_norm($v, 1.0)).allocs == 0
            @test run(@benchmarkable $index_sum($v)).allocs == 0
        end
        @test run(@benchmarkable $set_all!($(copy(hv)))).allocs == 0
    end
end
