# Callbacks with a unitful CollectionVector: refill N whenever it drops to 0.5 mol.
# Run with:  julia --project=examples examples/cv_wrapper_callbacks.jl

using HeterogeneousArrays
using Unitful
import DifferentialEquations as DE

decay!(du, u, p, t) = (du.N = -p.k * u.N; nothing)
u0 = CollectionVector(N = 1.0u"mol")
prob = DE.ODEProblem(decay!, u0, (0.0u"s", 10.0u"s"), (k = 0.5u"s^-1",))
abstol = (N = 1.0e-8u"mol",)

condition(u, t, integrator) = u.N - 0.5u"mol"
refill!(integrator) = (integrator.u.N = 1.0u"mol")
cb = DE.ContinuousCallback(condition, refill!)

function attempt(name, f)
    try
        println("works: ", name, " => ", f())
    catch e
        println("fails: ", name, " => ", first(sprint(showerror, e), 120))
    end
end

# Without the wrapper: the root finder compares the unitful condition with a plain 0.0
attempt("ContinuousCallback with Tsit5",
    () -> DE.solve(prob, DE.Tsit5(); callback = cb).u[end])

# With the wrapper: callbacks are refused, since they would see the raw (unitless) integrator
attempt("ContinuousCallback with cv_wrapper(Tsit5())",
    () -> DE.solve(prob, cv_wrapper(DE.Tsit5()); abstol, callback = cb).u[end])
