using HeterogeneousArrays: HeterogeneousArrays, rawdata, shapeof, attach

@testset "CollectionVector: construction" begin
    x = CollectionVector(θ = 0.1u"rad", pos = [1.0, 2.0]u"m")
    @test x isa CollectionVector{Float64}
    @test rawdata(x) == [0.1, 1.0, 2.0]
    @test shapeof(x) == (θ = (1, u"rad"), pos = (2:3, u"m"))
    @test eltype(x) == Unitful.Quantity{Float64}
    @test length(x) == 3
    @test CollectionVector((θ = 0.1u"rad", pos = [1.0, 2.0]u"m")) == x      # NamedTuple form

    @testset "plain-number fields" begin
        y = CollectionVector(n = 3.0, w = [1.0, 2.0], v = [3.0, 4.0]u"m/s")
        @test shapeof(y) == (n = (1, nothing), w = (2:3, nothing), v = (4:5, u"m/s"))
        @test y.w isa SubArray                # no reinterpret needed
        @test y.n === 3.0
        @test eltype(y) == Unitful.Quantity{Float64}
    end

    @testset "mixed units and number types within a field" begin
        @test rawdata(CollectionVector(a = Any[4.5u"m", 9.5u"ft"])) ≈ [4.5, 2.8956]
        @test rawdata(CollectionVector(a = [1.0u"m", 50.0u"cm"])) == [1.0, 0.5]
        @test eltype(rawdata(CollectionVector(a = Any[4.5f0u"m", 9.5u"ft"]))) == Float64
        @test eltype(rawdata(CollectionVector(a = [1.0f0, 2.0f0]u"m", b = 1.0f0))) == Float32
        @test eltype(rawdata(CollectionVector(a = [1, 2]u"m", b = 1.5))) == Float64
        @test eltype(rawdata(CollectionVector(a = Any[402.24f0, 41.2]))) == Float64
    end

    @testset "rejected inputs" begin
        @test_throws ArgumentError CollectionVector()
        @test_throws ArgumentError CollectionVector(a = Float64[])
        e = try CollectionVector(a = [4.5u"m", 9.5u"kg"]); catch e; e; end
        @test e isa ArgumentError
        @test occursin("Field 'a'", e.msg) && occursin("element 2 is 9.5 kg", e.msg) && occursin("DimensionError", e.msg)
        for bad in ([FakeQuantity(2.2, :kms)], [2.2]DynamicQuantities.us"km/s", [2.2]DynamicQuantities.u"km/s",
                    [1.0 + 2im]u"m")
            e = try CollectionVector(a = bad); catch e; e; end
            @test e isa ArgumentError && occursin("does not support", e.msg) && occursin("isstorable", e.msg)
        end
        e = try CollectionVector(a = big(1.0)u"m"); catch e; e; end
        @test e isa ArgumentError && occursin("isbits", e.msg)
    end
end

@testset "CollectionVector: property access" begin
    x = CollectionVector(θ = 0.1u"rad", pos = [1.0, 2.0]u"m")
    @test x.θ === 0.1u"rad"
    @test x.pos isa Base.ReinterpretArray
    @test x.pos[2] === 2.0u"m"
    @test propertynames(x) == (:θ, :pos)
    @test_throws ArgumentError x.nope

    x.pos[1] = 50u"cm";           @test rawdata(x) ≈ [0.1, 0.5, 2.0]
    x.pos .= [3.0, 4.0]u"cm";     @test rawdata(x) ≈ [0.1, 0.03, 0.04]
    x.pos .= 0.0u"m";             @test rawdata(x) ≈ [0.1, 0.0, 0.0]
    x.pos .= 2 .* x.pos .+ 1u"m"; @test rawdata(x) ≈ [0.1, 1.0, 1.0]
    x.pos = [1.0, 2.0]u"m";       @test rawdata(x) ≈ [0.1, 1.0, 2.0]
    x.θ = 90u"°";                 @test rawdata(x)[1] ≈ π / 2

    @test_throws Unitful.DimensionError x.θ = 1.0u"m"
    @test_throws Unitful.DimensionError x.pos[1] = 7.0
    @test_throws Unitful.DimensionError x.pos[1] = 1.0u"s"
    @test_throws ArgumentError x.pos = 1.0u"m"
    @test_throws DimensionMismatch x.pos = [1.0]u"m"
end

@testset "CollectionVector: flat indexing" begin
    x = CollectionVector(θ = 0.1u"rad", pos = [1.0, 2.0]u"m")
    @test x[1] === 0.1u"rad"
    @test x[2] === 1.0u"m"
    @test collect(x) == [0.1u"rad", 1.0u"m", 2.0u"m"]
    x[2] = 50u"cm";  @test rawdata(x)[2] ≈ 0.5
    @test_throws Unitful.DimensionError x[1] = 1.0u"m"
    @test_throws BoundsError x[4]
end

@testset "CollectionVector: attach, rawdata, shapeof" begin
    x = CollectionVector(θ = 0.1u"rad", pos = [1.0, 2.0]u"m")
    S = shapeof(x)
    @test attach(S, rawdata(x)) == x
    @test rawdata(attach(S, [1.0, 2.0, 3.0])) === [1.0, 2.0, 3.0] || rawdata(attach(S, [1.0, 2.0, 3.0])) == [1.0, 2.0, 3.0]
    @test_throws DimensionMismatch attach(S, [1.0, 2.0])
    @test rawdata([1.0, 2.0]) == [1.0, 2.0]                     # identity on plain vectors

    x32 = attach(S, Float32[1, 2, 3])
    @test x32 isa CollectionVector{Float32} && x32.pos[1] === 2.0f0u"m"

    # Test with Duals from ForwardDiff.
    xd = attach(S, ForwardDiff.Dual.([1.0, 2.0, 3.0], 1.0))
    @test xd.pos[1] isa Unitful.Quantity{<:ForwardDiff.Dual}
    @test ForwardDiff.value(ustrip(xd.θ)) == 1.0
end

@testset "CollectionVector: NamedTuple and show" begin
    x = CollectionVector(θ = 0.1u"rad", pos = [1.0, 2.0]u"m")
    nt = NamedTuple(x)
    @test nt.θ === 0.1u"rad" && nt.pos == [1.0, 2.0]u"m" && nt.pos isa Vector
    s = sprint(show, MIME"text/plain"(), x)
    @test occursin("3-element CollectionVector{Float64} with fields (:θ, :pos)", s)
    @test occursin("θ = 0.1 rad", s) && occursin("pos = [1.0, 2.0] m", s)
end

@testset "CollectionVector: complex fields" begin
    c = CollectionVector(z = [1.0 + 2im, 3.0 + 4im], r = 5.0, s = 1.0 + 0im)
    @test rawdata(c) == [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 0.0]        # two real slots per complex element
    @test eltype(rawdata(c)) == Float64
    @test shapeof(c) == (z = (1:4, HeterogeneousArrays.ComplexParts()), r = (5, nothing), s = (6, HeterogeneousArrays.ComplexParts()))
    @test length(c) == 7                                            # flat indexing counts slots
    @test eltype(c) == Float64

    @test c.z isa Base.ReinterpretArray && c.z[2] === 3.0 + 4.0im
    @test c.r === 5.0                                               # not promoted to complex
    @test c.s === 1.0 + 0.0im

    c.z[1] = 7.0 - 1im;               @test rawdata(c)[1:2] == [7.0, -1.0]
    c.z .= im .* c.z;                 @test rawdata(c)[1:4] == [1.0, 7.0, -4.0, 3.0]
    c.z = [0.0 + 1im, 1.0 + 0im];     @test rawdata(c)[1:4] == [0.0, 1.0, 1.0, 0.0]
    c.s = 2.0 + 3im;                  @test rawdata(c)[6:7] == [2.0, 3.0]
    @test c[2] === 1.0 && c[7] === 3.0                              # raw parts through the flat face
    c[2] = 9.0;                       @test c.z[1] === 0.0 + 9.0im
    @test NamedTuple(c).z isa Vector{ComplexF64}
    @test occursin("z = ComplexF64[0.0+9.0im, 1.0+0.0im]", sprint(show, MIME"text/plain"(), c))
    @test_throws DimensionMismatch c.z = [1.0 + 0im]

    @test attach(shapeof(c), rawdata(c)) == c
    # Test that attach works with Duals. Note that Complex of Duals is something but not
    # Dual of Complexes (Dual is defined as a subtype of Real inside ForwardDiff, and has
    # to be for mathematical reasons).
    cd = attach(shapeof(c), ForwardDiff.Dual.(rawdata(c), 1.0))
    @test cd.z[1] isa Complex{<:ForwardDiff.Dual}
end

@testset "CollectionVector: element API from outside" begin
    t = CollectionVector(a = [TaggedNumber(2.2, :kms), TaggedNumber(9.2, :kms)], b = TaggedNumber(1.0, :m))
    @test rawdata(t) == [2.2, 9.2, 1.0]
    @test shapeof(t) == (a = (1:2, :kms), b = (3, :m))
    @test t.b == TaggedNumber(1.0, :m)
    @test t.a[2] == TaggedNumber(9.2, :kms)
    @test t[1] == TaggedNumber(2.2, :kms)
    t.a[1] = TaggedNumber(5.0, :kms);  @test rawdata(t)[1] == 5.0
    e = try CollectionVector(a = [TaggedNumber(2.2, :kms), TaggedNumber(9.2, :m)]); catch e; e; end
    @test e isa ArgumentError && occursin("element 2", e.msg) && occursin("unit mismatch", e.msg)
    # A type whose values are not laid out like one raw number is caught by the layout check.
    HeterogeneousArrays.isstorable(::Type{<:FakeQuantity}) = true
    HeterogeneousArrays.rawtype(::Type{FakeQuantity{T}}) where {T} = T
    HeterogeneousArrays.field_type(q::FakeQuantity) = q.unit
    HeterogeneousArrays.elementtype(::Type{T}, ::Symbol) where {T} = FakeQuantity{T}   # overrides TaggedNumber's for this check
    e = try CollectionVector(a = FakeQuantity(1.0, :m)); catch e; e; end
    @test e isa ArgumentError && occursin("memory layout", e.msg)
    HeterogeneousArrays.elementtype(::Type{T}, u::Symbol) where {T} = TaggedNumber{T, u}  # restore
end

@testset "CollectionVector: Unitful.AbstractQuantity subtypes" begin
    x = CollectionVector(a = [FrozenQuantity(1.0, u"m"), FrozenQuantity(50.0, u"cm")], b = FrozenQuantity(0.5, u"rad"))
    @test rawdata(x) == [1.0, 0.5, 0.5]
    @test shapeof(x) == (a = (1:2, u"m"), b = (3, u"rad"))
    # The field type is the unit alone (for the current version of the prototype), so
    # elements read back as plain `Quantity`s
    @test x.b === 0.5u"rad"
    @test x.a == [1.0, 0.5]u"m"
    @test eltype(x) == Unitful.Quantity{Float64}
    # Assignment converts into the field's unit, from either quantity type
    x.b = FrozenQuantity(180.0, u"°");  @test rawdata(x)[3] ≈ Float64(π)
    x.a[1] = FrozenQuantity(3.0, u"m"); @test rawdata(x)[1] == 3.0
    x[2] = FrozenQuantity(40.0, u"cm"); @test rawdata(x)[2] ≈ 0.4
    @test_throws Unitful.DimensionError x.b = FrozenQuantity(1.0, u"kg")
    # Mixed with regular quantities in one field, and promotion of the raw type
    @test rawdata(CollectionVector(a = Any[FrozenQuantity(1.0f0, u"m"), 50.0u"cm"])) == [1.0, 0.5]
    @test eltype(rawdata(CollectionVector(a = FrozenQuantity(1.0f0, u"m")))) == Float32
end
