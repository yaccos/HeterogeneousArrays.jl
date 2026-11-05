using HeterogeneousArrays: HeterogeneousArrays, rawdata, shapeof

@testset "CollectionVector map is field-wise" begin
    # `abs` sees the complex elements rather than their raw parts
    z = CollectionVector(z = [1.0 + 2im, 3.0 - 1im], r = 2.0u"m")
    a = @inferred map(abs, z)
    @test a isa CollectionVector
    @test shapeof(a) == (z = (1:2, nothing), r = (3, u"m"))
    @test rawdata(a) ≈ [sqrt(5.0), sqrt(10.0), 2.0]
    # Field types follow the result type of the function
    x = CollectionVector(θ = 0.1u"rad", pos = [1.0, 2.0]u"m")
    sq = @inferred map(q -> q * q, x)
    @test shapeof(sq) == (θ = (1, u"rad^2"), pos = (2:3, u"m^2"))
    @test rawdata(sq) ≈ [0.01, 1.0, 4.0]
    # Several CollectionVectors with the same field names
    y = CollectionVector(θ = 0.2u"rad", pos = [3.0, 4.0]u"m")
    s = @inferred map(+, x, y)
    @test shapeof(s) == shapeof(x) && rawdata(s) ≈ [0.3, 4.0, 6.0]
    @test map(+, x, y) == x .+ y
end

@testset "CollectionVector map rejects mismatched fields" begin
    x = CollectionVector(θ = 0.1u"rad", pos = [1.0, 2.0]u"m")
    @test_throws ArgumentError map(+, x, CollectionVector(φ = 0.1u"rad", pos = [1.0, 2.0]u"m"))
    # Fields of different lengths throw rather than being truncated as in Base's `map`
    @test_throws DimensionMismatch map(+, x,
        CollectionVector(θ = 0.1u"rad", pos = [1.0, 2.0, 3.0]u"m"))
end

@testset "CollectionVector map! converts into the destination" begin
    x = CollectionVector(θ = 0.1u"rad", pos = [1.0, 2.0]u"m")
    y = CollectionVector(θ = 0.2u"rad", pos = [3.0, 4.0]u"m")
    # Into a CollectionVector: results convert into the destination's units
    dest = CollectionVector(θ = 0.0u"°", pos = [0.0, 0.0]u"cm")
    @test map!(+, dest, x, y) === dest
    @test rawdata(dest) ≈ [rad2deg(0.3), 400.0, 600.0]
    # Into a plain vector laid out like the result's storage]
    v = zeros(3)
    map!(ustrip, v, x)
    @test v == [0.1, 1.0, 2.0]
    @test_throws DimensionMismatch map!(ustrip, zeros(4), x)
    # As with `.=`, the destination needs the slot count of the arguments, so a function
    # that changes the slots per element does not work with `map!`
    z = CollectionVector(z = [1.0 + 2im, 3.0 - 1im])
    @test_throws DimensionMismatch map!(abs, zeros(2), z)
    @test_throws DimensionMismatch map!(abs, zeros(4), z)
end

@testset "CollectionVector mapreduce is field-wise" begin
    # Unit-free complex field: the elements are added rather than their raw parts
    z = CollectionVector(z = [1 + 2im, 3 - 1im])
    @test sum(z) == 4 + 1im
    @test sum(abs2, z) == 15
    @test sum(abs, z) ≈ sqrt(5) + sqrt(10)
    # Compatible units convert, incompatible units throw
    @test maximum(CollectionVector(a = [1.0, 2.0]u"m", b = 150.0u"cm")) == 2.0u"m"
    @test maximum(CollectionVector(a = [1.0, 2.0]u"m", b = 250.0u"cm")) == 250.0u"cm"
    @test_throws Unitful.DimensionError sum(CollectionVector(a = [1.0, 2.0]u"m", b = 3.0u"s"))
    # `init` is used once, before the first field
    x = CollectionVector(a = [1.0, 2.0]u"m", b = 3.0u"m", c = [4.0]u"m")
    @test sum(x; init = 10.0u"m") == 20.0u"m"
    @test mapreduce(identity, (acc, q) -> acc + 1, CollectionVector(a = 1.0, b = 2.0);
        init = 0) == 2
    @test reduce(+, x) == 10.0u"m"
    @test count(>(1.5u"m"), x) == 3
    @test extrema(x) == (1.0u"m", 4.0u"m")
    # Only reductions over all elements are supported
    @test_throws "dims" sum(x; dims = 1)
    @test_throws ArgumentError mapreduce(identity, +, x; dims = 1)
end

@testset "CollectionVector single-unit reductions infer and do not allocate" begin
    x = CollectionVector(a = [1.0, 2.0]u"m", b = 3.0u"m", c = [4.0, 5.0]u"m")
    @test @inferred(sum(x)) == 15.0u"m"
    @test @inferred(maximum(x)) == 5.0u"m"
    @test @inferred(sum(x; init = 0.0u"m")) == 15.0u"m"
    # Measure inside functions: at the top level of a test set the result of `sum(x)` is
    # boxed, and Julia 1.10 counts that box as an allocation.
    allocated_sum(x) = @allocated sum(x)
    allocated_maximum(x) = @allocated maximum(x)
    allocated_sum_init(x, init) = @allocated sum(x; init)
    allocated_sum(x), allocated_maximum(x), allocated_sum_init(x, 0.0u"m")
    @test allocated_sum(x) == 0
    @test allocated_maximum(x) == 0
    @test allocated_sum_init(x, 0.0u"m") == 0
end

@testset "CollectionVector any and all see elements" begin
    # The real part of the first element is zero, but no element is zero
    z = CollectionVector(z = [0.0 + 1im, 2.0 + 0im], r = 1.0u"m")
    @test !any(iszero, z)
    @test any(iszero, CollectionVector(z = [0.0 + 1im, 0.0 + 0im], r = 1.0u"m"))
    @test all(q -> q != zero(q), z)
    x = CollectionVector(a = [1.0, 2.0]u"m", b = NaN * u"s")
    @test any(isnan, x) && !all(isfinite, x)
    @test all(isfinite, CollectionVector(a = [1.0, 2.0]u"m", b = 3.0u"s"))
    @test_throws ArgumentError any(isnan, x; dims = 1)
    @test_throws ArgumentError all(isnan, x; dims = 1)
end

@testset "CollectionVector iteration stays slot-wise" begin
    z = CollectionVector(z = [1.0 + 2im, 3.0 - 1im], r = 2.0u"m")
    @test length(z) == 5
    @test length(collect(z)) == 5
    @test collect(z)[1:4] == [1.0, 2.0, 3.0, -1.0]
end
