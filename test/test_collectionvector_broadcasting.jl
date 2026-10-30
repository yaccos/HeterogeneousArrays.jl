using HeterogeneousArrays: HeterogeneousArrays, rawdata, shapeof, attach

@testset "CollectionVector broadcasting: out of place" begin
    x = CollectionVector(θ = 0.1u"rad", pos = [1.0, 2.0]u"m")
    y = CollectionVector(θ = 0.2u"rad", pos = [3.0, 4.0]u"m")

    s = @inferred(((x, y) -> x .+ y)(x, y))
    @test s isa CollectionVector && shapeof(s) == shapeof(x)
    @test s.θ ≈ 0.3u"rad" && s.pos ≈ [4.0, 6.0]u"m"
    @test rawdata(@inferred((x -> 2 .* x)(x))) ≈ [0.2, 2.0, 4.0]
    @test rawdata(@inferred(((x, y) -> x .+ 2 .* (y .- x))(x, y))) ≈ [0.3, 5.0, 6.0]
    # Field types follow the function's result type
    sq = @inferred((x -> x .* x)(x))
    @test shapeof(sq) == (θ = (1, u"rad^2"), pos = (2:3, u"m^2"))
    @test rawdata(sq) ≈ [0.01, 1.0, 4.0]
    st = @inferred((x -> ustrip.(x))(x))
    @test shapeof(st) == (θ = (1, nothing), pos = (2:3, nothing)) && eltype(st) == Float64
    rt = @inferred((x -> sqrt.(x))(x))
    @test shapeof(rt) == (θ = (1, u"rad^(1/2)"), pos = (2:3, u"m^(1/2)"))
    @test rt.pos ≈ [1.0, sqrt(2.0)]u"m^(1/2)"
    # Storage type is the promotion of the fields' storage types
    gt = @inferred((x -> x .> zero.(x))(x))
    @test rawdata(gt) isa Vector{Bool} && rawdata(gt) == [true, true, true]
    @test rawdata(@inferred((g -> g .+ 1.5)(gt))) isa Vector{Float64}
    positive(q) = q isa Unitful.Quantity ? q > zero(q) : q
    w = @inferred((w -> positive.(w))(CollectionVector(a = [1.0, -2.0]u"m", b = 3.0)))
    @test rawdata(w) isa Vector{Float64} && rawdata(w) == [1.0, 0.0, 3.0]

    # Complex fields: a result may change the number of slots per element
    z = CollectionVector(z = [1.0 + 2im, 3.0 - 1im], r = 2.0u"m")
    zz = @inferred((z -> z .* z)(z))
    @test zz.z ≈ [-3.0 + 4.0im, 8.0 - 6.0im] && zz.r ≈ 4.0u"m^2"
    a = @inferred((z -> abs.(z))(z))
    @test shapeof(a) == (z = (1:2, nothing), r = (3, u"m"))
    @test rawdata(a) ≈ [sqrt(5.0), sqrt(10.0), 2.0]
    r = CollectionVector(a = [1.0, 2.0], b = 3.0)
    c = @inferred((r -> r .+ 0im)(r))
    @test shapeof(c) == (a = (1:4, HeterogeneousArrays.ComplexField()),
        b = (5, HeterogeneousArrays.ComplexField()))
    @test rawdata(c) == [1.0, 0.0, 2.0, 0.0, 3.0, 0.0]

    # Scalar, array and complex fields together
    m = CollectionVector(a = [1.0, 2.0]u"m", t = 3.0u"s", c = [1.0 + 2im], n = 4)
    m2 = @inferred((m -> m .* 2)(m))
    @test m2.a ≈ [2.0, 4.0]u"m" && m2.t ≈ 6.0u"s" && m2.c ≈ [2.0 + 4.0im] && m2.n ≈ 8.0
end

@testset "CollectionVector broadcasting: in place" begin
    function addmeter!(x)
        x .= x .+ 1u"m"
        return nothing
    end
    x = CollectionVector(a = [1.0, 2.0]u"m", b = 3.0u"m")
    addmeter!(x)
    @test rawdata(x) ≈ [2.0, 3.0, 4.0]
    @test (@allocated addmeter!(x)) == 0
    x .= 0.0u"m"
    @test rawdata(x) == zeros(3)
    # Values convert into the destination's units and storage type
    x .= CollectionVector(a = [100.0, 200.0]u"cm", b = 50.0u"cm")
    @test rawdata(x) ≈ [1.0, 2.0, 0.5]
    x .= CollectionVector(a = Float32[1, 2]u"cm", b = 3.0f0u"cm")
    @test isapprox(rawdata(x), [0.01, 0.02, 0.03]; rtol = 1e-6)
    # A complex field keeps its type (as we are broadcasting in-place)
    z = CollectionVector(z = [3.0 + 4im, 0.0 - 1im], r = 1.0)
    z .= abs.(z)
    @test rawdata(z) == [5.0, 0.0, 1.0, 0.0, 1.0]
    z .= 2.0
    @test z.z == [2.0 + 0.0im, 2.0 + 0.0im] && z.r == 2.0
    # The right-hand side is read in full before any field is written
    x = CollectionVector(a = [1.0, 2.0]u"m", b = 3.0u"m")
    x .= view(rawdata(x), 3:-1:1) .* 1.0u"m"
    @test rawdata(x) == [3.0, 2.0, 1.0]
end

@testset "CollectionVector broadcasting: ordinary arrays" begin
    u = CollectionVector(a = [0.0, 0.0], b = 0.0)
    v = [1.0, 2.0, 3.0]
    @test rawdata(@inferred(((u, v) -> u .+ v)(u, v))) == v
    u .= v
    @test u.a == [1.0, 2.0] && u.b === 3.0
    @test rawdata(u .* [10.0, 20.0, 30.0]) == [10.0, 40.0, 90.0]
    z = CollectionVector(c = [1.0 + 2im], b = 1.0)
    @test_throws "Cannot broadcast field 'c' with an ordinary array: its elements occupy more than one storage slot each" z .*
                                                                                                                          v
    # Unit-free values do not convert into quantities
    x = CollectionVector(a = [0.0, 0.0]u"m", b = 0.0u"m")
    @test_throws "Field 'a': broadcasting into elements of type" x .= v
end

@testset "CollectionVector broadcasting: plain destination" begin
    x = CollectionVector(a = [3.0, 4.0]u"m", b = 5.0u"m")
    y = CollectionVector(a = [1.0, 1.0]u"m", b = 2.0u"m")
    r = zeros(3)
    r .= ustrip.(x .- y)
    @test r == [2.0, 3.0, 3.0]
    @test_throws Unitful.DimensionError r .= x .- y
    @test_throws DimensionMismatch zeros(4) .= ustrip.(x .- y)
    # The destination is laid out like the result, which must have as many slots as the input
    z = CollectionVector(c = [1.0 + 2im], b = 1.0)
    @test_throws "Destination has length 3, the broadcast result has 2 storage slots" (zeros(3) .= abs.(z))
    r = CollectionVector(a = [1.0, 2.0], b = 3.0)
    @test_throws "Destination has length 3, the broadcast result has 6 storage slots" (zeros(3) .= r .+
                                                                                                   0im)
    @test_throws "Cannot broadcast field 'c' with an ordinary array" (zeros(3) .= z .* 1)
end

@testset "CollectionVector broadcasting: rejected inputs" begin
    x = CollectionVector(θ = 0.1u"rad", pos = [1.0, 2.0]u"m")
    q = CollectionVector(q = 1.0)
    @test_throws "Cannot broadcast CollectionVectors with different field names: (:θ, :pos) vs (:q,)" x .+
                                                                                                      q
    bc = Broadcast.instantiate(Broadcast.broadcasted(+, x, 1))
    @test_throws "Cannot broadcast into a CollectionVector with fields (:q,) from a broadcast over fields (:θ, :pos)" copyto!(
        q, bc)
    hv = HeterogeneousVector(θ = 0.1u"rad", pos = [1.0, 2.0]u"m")
    msg = "Cannot broadcast a CollectionVector (fields (:θ, :pos)) together with a HeterogeneousVector (fields (:θ, :pos))"
    @test_throws msg x .+ hv
    @test_throws msg hv .+ x
    @test_throws "Cannot broadcast a CollectionVector with a Tuple" x .+ (1.0, 2.0, 3.0)
    @test_throws "Cannot broadcast a CollectionVector with a 2-dimensional array" x .+
                                                                                  ones(3, 3)
    @test_throws "Field 'θ': the broadcast produces elements of type String, which CollectionVector does not support" string.(x)
    # Complex results are storable only in unit-free fields
    @test_throws "Field 'θ': the broadcast produces elements of type Quantity{ComplexF64" x .*
                                                                                          (1 +
                                                                                           0im)
    @test_throws Unitful.DimensionError x .+ 0im
    # Later elements must convert into the field type set by the first element
    seconds_above(p) = p > 3.5u"m" ? ustrip(p) * u"s" : p
    l = CollectionVector(a = [3.0, 4.0]u"m")
    @test_throws "Field 'a': broadcasting into elements of type Quantity{Float64, 𝐋" seconds_above.(l)
    @test_throws "Cause: DimensionError" x .= x .* 1.0u"s"
    # Failing field get named in the error message. Earlier fields keep their new values
    p = CollectionVector(a = [1.0, 2.0]u"m", b = 3.0u"m")
    q = CollectionVector(a = [5.0, 6.0]u"m", b = 7.0u"s")
    @test_throws "Field 'b': broadcasting into elements of type Quantity{Float64, 𝐋" p .= q
    @test p.a == [5.0, 6.0]u"m" && p.b == 3.0u"m"
    # TODO: add test that also change Tnew, e.g. Complex that turns into a Float64
    # TODO: add test similar to seconds_above but that converts to ft
end
