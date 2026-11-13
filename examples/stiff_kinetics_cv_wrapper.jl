# Solving a stiff, unitful ODE with `cv_wrapper`
#
# Robertson's chemical kinetics problem, a classic stiff test case, written with units:
#
#     A → B          k1 = 0.04 1/s
#     B + B → C + B  k3 = 3e7  L/(mol s)
#     B + C → A + C  k2 = 1e4  L/(mol s)
#
# The reactions run on time scales from ~1e-8 s to ~1e5 s. Explicit solvers need tiny steps
# for stability, and implicit/Rosenbrock solvers need a Jacobian, which they cannot build for
# a state vector with units. `cv_wrapper(alg)` solves on the unitless storage of the
# CollectionVector, while the ODE function, tolerances, times and solution all keep their
# units.
#
# Run from the repository root with:  julia --project=examples examples/stiff_kinetics_cv_wrapper.jl

using HeterogeneousArrays
using Unitful
import DifferentialEquations as DE

# ------------------------------------------------------------------------------------------
# 1. State, parameters and ODE function, all with units
# ------------------------------------------------------------------------------------------

u0 = CollectionVector(A = 1.0u"mol/L", B = 0.0u"mol/L", C = 0.0u"mol/L")
p = (k1 = 0.04u"s^-1", k2 = 1.0e4u"L/(mol*s)", k3 = 3.0e7u"L/(mol*s)")

function robertson!(du, u, p, t)
    du.A = -p.k1 * u.A + p.k2 * u.B * u.C
    du.B = p.k1 * u.A - p.k2 * u.B * u.C - p.k3 * u.B^2
    du.C = p.k3 * u.B^2
    return nothing
end

# `tspan` must have units
tspan = (0.0u"s", 1.0e5u"s")
prob = DE.ODEProblem(robertson!, u0, tspan, p)

# Absolute tolerances per field, with units. A bare number like `abstol = 1e-8` is rejected,
# because its unit would be ambiguous. B stays ~1e-5 mol/L, so it needs a tighter tolerance.
abstol = (A = 1.0e-8u"mol/L", B = 1.0e-14u"mol/L", C = 1.0e-8u"mol/L")
reltol = 1.0e-6  # relative tolerance is dimensionless

# ------------------------------------------------------------------------------------------
# 2. Solve with a stiff solver through the wrapper
# ------------------------------------------------------------------------------------------

# Wrap the solver algorithm. Any OrdinaryDiffEq algorithm works, including implicit and
# Rosenbrock methods with ForwardDiff (the default) or finite-difference Jacobians.
# Times given as keyword arguments (saveat, dt, tstops, ...) take units too.
sol = DE.solve(prob, cv_wrapper(DE.Rodas5P()); abstol, reltol,
    saveat = [0.0, 1.0, 10.0, 100.0, 1000.0] .* u"s")

println("retcode: ", sol.retcode, ", steps: ", sol.stats.naccept,
    ", Jacobians: ", sol.stats.njacs)
println("\nSaved states:")
for (t, u) in zip(sol.t, sol.u)          # sol.t has units, sol.u are CollectionVectors
    println("  t = ", rpad(t, 10), "  A = ", round(u"mol/L", u.A; sigdigits = 5),
        "   B = ", round(u"mol/L", u.B; sigdigits = 5),
        "   C = ", round(u"mol/L", u.C; sigdigits = 5))
end

# Mass is conserved: A + B + C stays 1 mol/L
uend = sol.u[end]
println("\nA + B + C at the end: ", uend.A + uend.B + uend.C)

# ------------------------------------------------------------------------------------------
# 3. Dense solution: interpolate at any time, in any time unit
# ------------------------------------------------------------------------------------------

sol_dense = DE.solve(prob, cv_wrapper(DE.Rodas5P()); abstol, reltol)
x = sol_dense(2u"hr")               # 2 hours, converted to the solver's seconds
dx = sol_dense(2u"hr", Val{1})      # time derivative, in mol/(L s)
println("\nAt t = 2 hr:  C = ", x.C, ",  dC/dt = ", dx.C)

# ------------------------------------------------------------------------------------------
# 4. Why the wrapper is needed
# ------------------------------------------------------------------------------------------

# Without the wrapper, a Jacobian-based solver cannot handle the unitful state:
try
    DE.solve(prob, DE.Rodas5P(); abstol = 1.0e-8 .* oneunit.(u0))
catch e
    println("\nRodas5P without cv_wrapper fails: ", nameof(typeof(e)))
end

# The wrapper works for explicit solvers too. Since the problem is stiff, an explicit solver
# needs far more steps than an implicit one (compared here over the first 100 s only).
short = DE.remake(prob; tspan = (0.0u"s", 100.0u"s"))
sol_explicit = DE.solve(short, cv_wrapper(DE.Tsit5()); abstol, reltol)
sol_implicit = DE.solve(short, cv_wrapper(DE.Rodas5P()); abstol, reltol)
println("Steps for the first 100 s:  Tsit5 = ", sol_explicit.stats.naccept,
    ",  cv_wrapper(Rodas5P()) = ", sol_implicit.stats.naccept)

# ------------------------------------------------------------------------------------------
# 5. Unit errors are caught, with helpful messages
# ------------------------------------------------------------------------------------------

# A bug in the ODE function: du.C gets units of 1/s instead of mol/(L s)
function robertson_bug!(du, u, p, t)
    robertson!(du, u, p, t)
    du.C = p.k1
    return nothing
end
try
    DE.solve(DE.ODEProblem(robertson_bug!, u0, tspan, p), cv_wrapper(DE.Rodas5P()); abstol)
catch e
    println("\nWrong unit in the ODE function:\n  ", sprint(showerror, e))
end

# A tolerance without a unit
try
    DE.solve(prob, cv_wrapper(DE.Rodas5P()); abstol = 1.0e-8)
catch e
    println("\nUnitless abstol:\n  ", sprint(showerror, e))
end

# A time keyword argument with the wrong dimension
try
    DE.solve(prob, cv_wrapper(DE.Rodas5P()); abstol, saveat = 1.0u"m")
catch e
    println("\nsaveat in metres:\n  ", sprint(showerror, e))
end
