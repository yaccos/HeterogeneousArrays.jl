# The integrator interface (`init`, `step!`) with a unitful CollectionVector.
# Run with:  julia --project=examples examples/cv_wrapper_integrator.jl

using HeterogeneousArrays
using Unitful
import DifferentialEquations as DE

decay!(du, u, p, t) = (du.N = -p.k * u.N; nothing)
u0 = CollectionVector(N = 1.0u"mol")
prob = DE.ODEProblem(decay!, u0, (0.0u"s", 10.0u"s"), (k = 0.5u"s^-1",))
abstol = (N = 1.0e-8u"mol",)

function attempt(name, f)
    try
        println("works: ", name, " => ", f())
    catch e
        println("fails: ", name, " => ", first(sprint(showerror, e), 120))
    end
end

# Explicit solver without the wrapper: the integrator works with units
attempt("init + step! with Tsit5", () -> begin
    integrator = DE.init(prob, DE.Tsit5())
    DE.step!(integrator)
    (integrator.t, integrator.u.N)
end)

# Stiff solver without the wrapper: fails while building the Jacobian
attempt("init with Rodas5P", () -> DE.init(prob, DE.Rodas5P()))

# With the wrapper: `init` is not implemented (would need `__init`)
attempt("init with cv_wrapper(Rodas5P())",
    () -> DE.init(prob, cv_wrapper(DE.Rodas5P()); abstol))
