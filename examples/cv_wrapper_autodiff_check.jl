# Does `cv_wrapper` really use automatic differentiation, or does it silently fall back to
# finite differences?
#
# Stiff solvers (Rodas5P, FBDF, ...) need the Jacobian of the ODE function. With
# `autodiff = AutoForwardDiff()` (the default) it is computed with ForwardDiff: the solver
# calls the ODE function with `ForwardDiff.Dual` numbers, which carry derivatives along with
# the values. With finite differences the ODE function only ever sees plain Float64s.
#
# This script gives three independent pieces of evidence, and stops with an error if any of
# them shows a fallback:
#   1. Which number types reach the ODE function, and how often (Duals vs plain Float64).
#   2. The derivatives inside those Duals equal the exact Jacobian to rounding error
#      (finite differences would be off by ~1e-8 or more).
#   3. An ODE function that cannot handle Duals: with ForwardDiff the solve must fail,
#      because a silent fallback would make it succeed.
#
# Run from the repository root with:  julia --project=examples examples/cv_wrapper_autodiff_check.jl

using HeterogeneousArrays
using HeterogeneousArrays: rawdata
using Unitful
using ForwardDiff: ForwardDiff, Dual
using ADTypes: AutoFiniteDiff, AutoForwardDiff
using LinearAlgebra: norm
import DifferentialEquations as DE

# ------------------------------------------------------------------------------------------
# The problem: Robertson's stiff chemical kinetics, with units
# ------------------------------------------------------------------------------------------

u0 = CollectionVector(A = 1.0u"mol/L", B = 0.0u"mol/L", C = 0.0u"mol/L")
p = (k1 = 0.04u"s^-1", k2 = 1.0e4u"L/(mol*s)", k3 = 3.0e7u"L/(mol*s)")
tspan = (0.0u"s", 1.0e5u"s")
abstol = (A = 1.0e-8u"mol/L", B = 1.0e-14u"mol/L", C = 1.0e-8u"mol/L")
reltol = 1.0e-6

function robertson!(du, u, p, t)
    du.A = -p.k1 * u.A + p.k2 * u.B * u.C
    du.B = p.k1 * u.A - p.k2 * u.B * u.C - p.k3 * u.B^2
    du.C = p.k3 * u.B^2
    return nothing
end

# Exact Jacobian of the raw problem: the solver works on plain numbers in the units of u0
# (mol/L) and tspan (s), so the rate constants enter as plain numbers in those units.
function exact_jacobian(A, B, C)
    k1, k2, k3 = 0.04, 1.0e4, 3.0e7
    return [-k1 k2*C k2*B
            k1 -k2*C-2k3*B -k2*B
            0.0 2k3*B 0.0]
end

# ------------------------------------------------------------------------------------------
# A spy around the ODE function: it records every call and checks every Dual call
# ------------------------------------------------------------------------------------------

mutable struct Spy
    plain::Int            # calls with plain Float64 state and time
    dual_u::Int           # calls with a Dual state: Jacobian evaluations
    dual_t::Int           # calls with a Dual time only: time derivative (Rosenbrock methods)
    widths::Set{Int}      # number of partials per Dual (= Jacobian columns per call)
    max_jac_error::Float64  # largest relative error of the Dual derivatives vs exact_jacobian
end
Spy() = Spy(0, 0, 0, Set{Int}(), 0.0)

# The derivative parts of a vector of numbers, as a matrix (one column per partial).
partials_matrix(x::AbstractVector{<:Dual{T, V, N}}) where {T, V, N} =
    [ForwardDiff.partials(xi, j) for xi in x, j in 1:N]
partials_matrix(x::AbstractVector{<:Real}) = zeros(length(x), 0)
values_of(x) = ForwardDiff.value.(x)

function spied(spy::Spy)
    return function (du, u, p, t)
        robertson!(du, u, p, t)
        U, DU, T = rawdata(u), rawdata(du), ustrip(t)  # what the solver actually passed
        if !(eltype(DU) <: Dual)
            spy.plain += 1
            return nothing
        end
        push!(spy.widths, ForwardDiff.npartials(eltype(DU)))
        if eltype(U) <: Dual
            # Jacobian evaluation: by the chain rule, the partials of du must be J * partials of u
            spy.dual_u += 1
            J = exact_jacobian(values_of(U)...)
            Pu, Pdu = partials_matrix(U), partials_matrix(DU)
            err = norm(Pdu - J * Pu) / (norm(J) * norm(Pu))
            spy.max_jac_error = max(spy.max_jac_error, err)
        else
            # Only t is a Dual: the time derivative, which is zero here (no explicit t)
            T isa Dual && (spy.dual_t += 1)
            @assert iszero(partials_matrix(DU)) "time derivative should be exactly zero"
        end
        return nothing
    end
end

# ------------------------------------------------------------------------------------------
# 1 + 2. Which numbers reach the ODE function, and are the derivatives exact?
# ------------------------------------------------------------------------------------------

cases = [
    # name                                     algorithm                                      expects AD?
    ("Rodas5P()  [default autodiff]",          DE.Rodas5P(),                                  true),
    ("Rodas5P(autodiff = true)",               DE.Rodas5P(autodiff = true),                   true),
    ("Rodas5P(autodiff = AutoForwardDiff())",  DE.Rodas5P(autodiff = AutoForwardDiff()),      true),
    ("Rodas5P(AutoForwardDiff(chunksize=3))",  DE.Rodas5P(autodiff = AutoForwardDiff(chunksize = 3)), true),
    ("FBDF()",                                 DE.FBDF(),                                     true),
    ("KenCarp4()",                             DE.KenCarp4(),                                 true),
    ("DefaultODEAlgorithm()",                  DE.DefaultODEAlgorithm(),                      true),
    ("Rodas5P(autodiff = false)",              DE.Rodas5P(autodiff = false),                  false),
    ("Rodas5P(autodiff = AutoFiniteDiff())",   DE.Rodas5P(autodiff = AutoFiniteDiff()),       false),
    ("FBDF(autodiff = AutoFiniteDiff())",      DE.FBDF(autodiff = AutoFiniteDiff()),          false),
    ("no algorithm: cv_wrapper()",             nothing,                                       false),
]

println("1+2. Calls to the ODE function, and the error of the ForwardDiff Jacobian\n")
println(rpad("algorithm", 40), rpad("njacs", 7), rpad("plain", 7), rpad("Jacobian", 10),
    rpad("d/dt", 6), rpad("width", 7), "max rel. error vs exact J")
println("-"^100)
for (name, alg, expects_ad) in cases
    spy = Spy()
    prob = DE.ODEProblem(spied(spy), u0, tspan, p)
    sol = DE.solve(prob, cv_wrapper(alg); abstol, reltol)
    @assert sol.retcode == DE.ReturnCode.Success
    println(rpad(name, 40), rpad(sol.stats.njacs, 7), rpad(spy.plain, 7),
        rpad(spy.dual_u, 10), rpad(spy.dual_t, 6), rpad(join(sort!(collect(spy.widths)), ","), 7),
        spy.dual_u > 0 ? spy.max_jac_error : "-")

    # The checks: the AD cases must have differentiated with Duals, exactly; the
    # finite-difference cases must never have seen a Dual.
    if expects_ad
        @assert spy.dual_u > 0 "$name: requested ForwardDiff, but no Jacobian was built with Duals"
        @assert spy.max_jac_error < 1.0e-12 "$name: Dual derivatives differ from the exact Jacobian"
    else
        @assert spy.dual_u == spy.dual_t == 0 "$name: requested finite differences, got Duals"
    end
end
println("""

  Jacobian = calls with a Dual state (one per `width` Jacobian columns), d/dt = calls
  with a Dual time (time derivative, Rosenbrock methods only). With ForwardDiff the
  Jacobian is exact to rounding error (~1e-16); finite differences never send a Dual.
  Note `cv_wrapper()` without an algorithm: OrdinaryDiffEq's own default for "no
  algorithm" is finite differences (the same happens without cv_wrapper). Pass
  `cv_wrapper(DE.DefaultODEAlgorithm())` to get the default solver with ForwardDiff.
""")

# ------------------------------------------------------------------------------------------
# 3. An ODE function that cannot handle Duals
# ------------------------------------------------------------------------------------------

# Converting to Float64 throws for a Dual. If ForwardDiff fell back to finite differences
# when it hits a problem, the solve below would succeed; instead it must fail.
function robertson_float64_only!(du, u, p, t)
    A, B, C = (Float64(ustrip(u"mol/L", x)) for x in (u.A, u.B, u.C))
    k1, k2, k3 = 0.04, 1.0e4, 3.0e7
    du.A = (-k1 * A + k2 * B * C)u"mol/(L*s)"
    du.B = (k1 * A - k2 * B * C - k3 * B^2)u"mol/(L*s)"
    du.C = (k3 * B^2)u"mol/(L*s)"
    return nothing
end
prob_no_dual = DE.ODEProblem(robertson_float64_only!, u0, tspan, p)

println("3. ODE function that rejects Dual numbers\n")
for (name, alg, expects_ad) in cases
    outcome = try
        sol = DE.solve(prob_no_dual, cv_wrapper(alg); abstol, reltol)
        "solved ($(sol.retcode))"
    catch e
        "FAILED: $(nameof(typeof(e)))"
    end
    println("  ", rpad(name, 40), outcome)
    if expects_ad
        @assert startswith(outcome, "FAILED") "$name: ForwardDiff should have failed, but the solve succeeded"
    else
        @assert startswith(outcome, "solved") "$name: finite differences should work"
    end
end
println("""

  Every ForwardDiff case fails, and every finite-difference case succeeds: there is no
  silent fallback. (FirstAutodiffJacError is OrdinaryDiffEq re-throwing a failure in the
  first step with advice. DefaultODEAlgorithm starts with a non-stiff method and only
  builds a Jacobian after switching to a stiff one, so the Dual error surfaces as is.)
""")
println("All checks passed.")
