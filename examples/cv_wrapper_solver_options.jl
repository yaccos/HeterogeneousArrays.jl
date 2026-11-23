# Solver options (autodiff, linsolve, limiters) with `cv_wrapper`
#
# `cv_wrapper(alg)` hands `alg` unchanged to the solver, which runs on the unitless storage
# of the CollectionVector. This example checks what that means for the options of a stiff
# solver such as Rodas5P:
#   - `autodiff` is honored: ForwardDiff or finite differences, as requested.
#   - `linsolve` is honored: it solves the linear systems of the raw (unitless) problem.
#   - `step_limiter!` / `stage_limiter!` are called with the raw state: a plain Vector of
#     numbers in the units of u0, and a plain time in the unit of tspan.
#
# Run from the repository root with:  julia --project=examples examples/cv_wrapper_solver_options.jl

using HeterogeneousArrays
using HeterogeneousArrays: rawdata
using Unitful
using ADTypes: AutoFiniteDiff, AutoForwardDiff
using LinearSolve: KrylovJL_GMRES, LUFactorization
import DifferentialEquations as DE

# Exponential decay N -> energy E
u0 = CollectionVector(N = 1.0u"mol", E = 0.0u"J")
p = (k = 1.0e-3u"s^-1", e = 5.0e4u"J/mol")
abstol = (N = 1.0e-10u"mol", E = 1.0e-5u"J")
reltol = 1.0e-8
N_exact = exp(-1.0e-3 * 3600) * u0.N

# ------------------------------------------------------------------------------------------
# 1. autodiff
# ------------------------------------------------------------------------------------------

# The Jacobian mode is visible in what the solver passes to the ODE function. With
# ForwardDiff, the raw storage of `du` holds Dual numbers instead of Float64. The unitful
# view always shows Quantities, so we look at `rawdata`.
dual_calls = Ref(0)
plain_calls = Ref(0)
function decay!(du, u, p, t)
    eltype(rawdata(du)) === Float64 ? (plain_calls[] += 1) : (dual_calls[] += 1)
    du.N = -p.k * u.N
    du.E = p.k * u.N * p.e
    return nothing
end
prob = DE.ODEProblem(decay!, u0, (0.0u"s", 1.0u"hr"), p)

println("autodiff setting                 ODE calls with Duals / plain    N(1 hr)")
for (name, alg) in [
        ("default", DE.Rodas5P()),
        ("autodiff = true", DE.Rodas5P(autodiff = true)),
        ("autodiff = AutoForwardDiff()", DE.Rodas5P(autodiff = AutoForwardDiff())),
        ("autodiff = false", DE.Rodas5P(autodiff = false)),
        ("autodiff = AutoFiniteDiff()", DE.Rodas5P(autodiff = AutoFiniteDiff())),
    ]
    dual_calls[] = plain_calls[] = 0
    sol = DE.solve(prob, cv_wrapper(alg); abstol, reltol)
    println("  ", rpad(name, 30), " ", lpad(dual_calls[], 5), " / ", rpad(plain_calls[], 18),
        isapprox(sol.u[end].N, N_exact; rtol = 1.0e-6) ? "correct" : "WRONG")
end

# ------------------------------------------------------------------------------------------
# 2. linsolve
# ------------------------------------------------------------------------------------------

# The linear solver works on the raw Jacobian (numbers in the units of u0 and tspan), so any
# LinearSolve.jl method can be used: units never reach it.
println("\nlinsolve")
for (name, alg) in [
        ("LUFactorization()", DE.Rodas5P(linsolve = LUFactorization())),
        ("KrylovJL_GMRES()", DE.Rodas5P(linsolve = KrylovJL_GMRES())),
    ]
    sol = DE.solve(prob, cv_wrapper(alg); abstol, reltol)
    println("  ", rpad(name, 20), sol.retcode, ", N(1 hr) = ", sol.u[end].N)
end

# ------------------------------------------------------------------------------------------
# 3. step_limiter! / stage_limiter!
# ------------------------------------------------------------------------------------------

# Limiters are called as `limiter!(u, integrator, p, t)` with the raw state `u` (a plain
# Vector, in the units of u0) and a plain time `t` (in the unit of tspan).
arg_types = Ref{Any}(nothing)
record_args!(u, integrator, p, t) = (arg_types[] = (typeof(u), typeof(t)); nothing)
DE.solve(prob, cv_wrapper(DE.Rodas5P(step_limiter! = record_args!)); abstol, reltol)
println("\nstep_limiter!")
println("  gets u::", arg_types[][1], ", t::", arg_types[][2])

# A limiter that works on the raw numbers runs fine:
clip_raw!(u, integrator, p, t) = (u .= max.(u, 0); nothing)
sol = DE.solve(prob, cv_wrapper(DE.Rodas5P(step_limiter! = clip_raw!)); abstol, reltol)
println("  raw limiter: ", sol.retcode)

# A limiter written for the unitful state fails, since `u` has no fields:
clip_unitful!(u, integrator, p, t) = (u.N = max(u.N, 0.0u"mol"); nothing)
try
    DE.solve(prob, cv_wrapper(DE.Rodas5P(step_limiter! = clip_unitful!)); abstol, reltol)
catch e
    println("  unitful limiter: ", first(sprint(showerror, e), 100))
end
