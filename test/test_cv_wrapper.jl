using DifferentialEquations
using ADTypes: AutoFiniteDiff
using SciMLBase: SciMLBase, ReturnCode

# Exponential decay N -> energy E, with an exact solution
function decay!(du, u, p, t)
    du.N = -p.k * u.N
    du.E = p.k * u.N * p.e
    return nothing
end
# Out-of-place, returning the derivative in other (compatible) units
function decay(u, p, t)
    CollectionVector(N = uconvert(u"mol/minute", -p.k * u.N), E = p.k * u.N * p.e)
end

@testset "cv_wrapper" begin
    u0 = CollectionVector(N = 1.0u"mol", E = 0.0u"J")
    p = (k = 1.0e-3u"s^-1", e = 5.0e4u"J/mol")
    tol = (N = 1.0e-10u"mol", E = 1.0e-5u"J")
    prob = ODEProblem(decay!, u0, (0.0u"s", 1.0u"hr"), p)
    N_exact(t) = exp(-uconvert(NoUnits, p.k * t)) * u0.N

    @testset "solvers ($(nameof(typeof(alg))))" for alg in (
        Tsit5(), Rodas5P(), FBDF(), KenCarp4(), TRBDF2(),
        Rosenbrock23(autodiff = AutoFiniteDiff()), ImplicitEuler())
        sol = solve(prob, cv_wrapper(alg); abstol = tol, reltol = 1e-8)
        @test sol.retcode == ReturnCode.Success
        @test sol.u[end] isa CollectionVector
        @test sol.t[end] == 3600.0u"s"
        rtol = alg isa ImplicitEuler ? 1e-2 : 1e-5
        @test isapprox(sol.u[end].N, N_exact(1u"hr"); rtol)
        # Conservation: E = e * (N0 - N)
        @test isapprox(sol.u[end].E, p.e * (u0.N - sol.u[end].N); rtol)
    end

    @testset "default algorithm" begin
        sol = solve(prob, cv_wrapper(); abstol = tol)
        @test sol.retcode == ReturnCode.Success
    end

    @testset "solution" begin
        sol = solve(prob, cv_wrapper(Tsit5()); abstol = tol, reltol = 1e-10,
            saveat = 0u"minute":10u"minute":60u"minute")
        @test sol.t ≈ (0:600:3600)u"s"
        @test length(sol.u) == 7
        @test sol.u[1] == u0
        @test sol.stats.naccept > 0  # forwarded to the raw solution
        # interpolation with a unitful time, and its derivative
        sol = solve(prob, cv_wrapper(Tsit5()); abstol = tol, reltol = 1e-10)
        x = sol(30u"minute")
        @test x isa CollectionVector
        @test isapprox(x.N, N_exact(30u"minute"); rtol = 1e-6)
        dx = sol(30u"minute", Val{1})
        @test unit(dx.N) == u"mol/s"
        @test isapprox(dx.N, -p.k * N_exact(30u"minute"); rtol = 1e-6)
        xs = sol([10u"minute", 20u"minute"])
        @test length(xs) == 2 && xs[2] isa CollectionVector
        @test_throws ArgumentError sol(1.0u"m")
    end

    @testset "time units" begin
        # tspan in hours, dt in seconds
        prob_hr = ODEProblem(decay!, u0, (0.0u"hr", 1.0u"hr"), p)
        sol = solve(prob_hr, cv_wrapper(RK4()); dt = 1u"s", adaptive = false)
        @test sol.stats.naccept == 3600
        @test sol.t[end] ≈ 1u"hr"
        @test isapprox(sol.u[end].N, N_exact(1u"hr"); rtol = 1e-8)
        @test_throws ArgumentError solve(prob, cv_wrapper(RK4()); dt = 1u"m")
        @test_throws ArgumentError solve(prob, cv_wrapper(RK4()); dt = 1.0)
        @test_throws ArgumentError solve(prob, cv_wrapper(Tsit5()); abstol = tol,
            saveat = [1.0, 2.0])
        # unitless independent variable
        prob_nounit = ODEProblem((du, u, p, t) -> (du.x = -u.x; nothing),
            CollectionVector(x = 1.0u"m"), (0.0, 1.0))
        sol = solve(prob_nounit, cv_wrapper(Rodas5P()); abstol = (x = 1e-10u"m",))
        @test isapprox(sol.u[end].x, exp(-1.0)u"m"; rtol = 1e-5)
    end

    @testset "out-of-place" begin
        prob_oop = ODEProblem(decay, u0, (0.0u"s", 1.0u"hr"), p)
        sol = solve(prob_oop, cv_wrapper(Rodas5P()); abstol = tol, reltol = 1e-8)
        @test isapprox(sol.u[end].N, N_exact(1u"hr"); rtol = 1e-5)
    end

    @testset "abstol" begin
        ref = solve(prob, cv_wrapper(Tsit5()); abstol = tol).u[end]
        # same tolerance as a CollectionVector in other units
        cvtol = CollectionVector(N = 1.0e-7u"mmol", E = 1.0e-8u"kJ")
        @test solve(prob, cv_wrapper(Tsit5()); abstol = cvtol).u[end] == ref
        # array field
        prob_arr = ODEProblem(
            (du, u, p, t) -> (du.x .= -u.x ./ u"s"; du.y = -u.y / u"s"; nothing),
            CollectionVector(x = [1.0, 2.0]u"m", y = 1.0u"K"), (0.0u"s", 1.0u"s"))
        sol = solve(prob_arr, cv_wrapper(Tsit5());
            abstol = (x = [1e-9, 1e-9]u"m", y = 1e-9u"K"))
        @test isapprox(sol.u[end].x, [1.0, 2.0]u"m" .* exp(-1); rtol = 1e-5)
        # unitless abstol is rejected with a helpful message
        err = try
            solve(prob, cv_wrapper(Tsit5()); abstol = 1e-6)
        catch e
            e
        end
        @test err isa ArgumentError && occursin("has no unit", err.msg)
        # wrong dimension or missing fields
        @test_throws Exception solve(prob, cv_wrapper(Tsit5());
            abstol = (N = 1e-6u"mol", E = 1u"m"))
        @test_throws Exception solve(prob, cv_wrapper(Tsit5()); abstol = (N = 1e-6u"mol",))
    end

    @testset "unit checking in the RHS" begin
        bad!(du, u, p, t) = (du.N = -p.k * u.N; du.E = p.k * u.N; nothing)
        prob_bad = ODEProblem(bad!, u0, (0.0u"s", 1.0u"hr"), p)
        @test_throws Unitful.DimensionError solve(prob_bad, cv_wrapper(Rodas5P()); abstol = tol)
    end

    @testset "unsupported" begin
        @test_throws ArgumentError solve(prob, cv_wrapper(Tsit5()); abstol = tol,
            callback = DiscreteCallback((u, t, i) -> false, i -> nothing))
        hv_prob = ODEProblem((du, u, p, t) -> (du .= -u ./ u"s"; nothing),
            HeterogeneousVector(x = 1.0u"m"), (0.0u"s", 1.0u"s"))
        @test_throws ArgumentError solve(hv_prob, cv_wrapper(Tsit5()))
        @test_throws ArgumentError init(prob, cv_wrapper(Tsit5()))
    end

    @testset "no allocations in the wrapped RHS" begin
        Ext = Base.get_extension(HeterogeneousArrays, :HeterogeneousArraysSciMLBaseExt)
        SU = HeterogeneousArrays.shapeof(u0)
        tu = u"s"
        rhs = Ext.RawRHS{true, SU, Ext.derivative_shape(SU, tu), typeof(decay!), typeof(tu)}(
            decay!, tu)
        u = copy(HeterogeneousArrays.rawdata(u0))
        du = similar(u)
        rhs(du, u, p, 1.0)
        @test (@allocated rhs(du, u, p, 1.0)) == 0
        @test du ≈ [-1e-3, 50.0]
        ud = ForwardDiff.Dual.(u, 1.0)
        dud = similar(ud)
        rhs(dud, ud, p, 1.0)
        @test (@allocated rhs(dud, ud, p, 1.0)) == 0
    end
end
