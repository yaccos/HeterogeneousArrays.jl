# User-facing entry point of the CollectionVector ODE solver wrapper.
#
# Only the function name and its documentation live here, in the main package, so that
# `cv_wrapper` can be exported and documented without making SciMLBase a dependency. The
# implementation (the wrapper type, the unit translation of problem, keyword arguments and
# ODE function, and the solution with units) is in the package extension
# ext/HeterogeneousArraysSciMLBaseExt.jl, which Julia loads automatically as soon as SciMLBase
# is loaded (e.g. by `using OrdinaryDiffEq`). Until then `cv_wrapper` has no methods.

"""
    cv_wrapper(alg = nothing)

Wrap an ODE solver algorithm so that a problem whose state is a [`CollectionVector`](@ref) is
solved on the unitless storage ([`rawdata`](@ref)) and the result is given back with units.

```julia
using OrdinaryDiffEq, Unitful, HeterogeneousArrays

function decay!(du, u, p, t)
    du.N = -p.k * u.N          # u.N in mol, du.N in mol/s: unit-checked on assignment
    du.E = p.k * u.N * p.e
end
u0 = CollectionVector(N = 1.0u"mol", E = 0.0u"J")
prob = ODEProblem(decay!, u0, (0.0u"s", 1.0u"hr"), (k = 1.0e-3u"s^-1", e = 5.0e4u"J/mol"))
sol = solve(prob, cv_wrapper(Rodas5P()); abstol = (N = 1.0e-9u"mol", E = 1.0e-3u"J"))
sol.u[end].N, sol(30u"minute").E
```

The ODE function is passed as-is: inside the solver it is called with `du` and `u` as
`CollectionVector`s over the solver's raw arrays (no copies), and `t` with the unit of
`tspan`. `du` has the units of `u` divided by the unit of time, and writing a value with the
wrong dimension into it throws a `DimensionError`.

Since the solver only ever sees a plain `Vector`, solvers that need a Jacobian (implicit,
Rosenbrock, BDF, ...) work, including with ForwardDiff.

# Keyword arguments
- `tspan` (in the problem) sets the unit of the independent variable. `dt`, `dtmin`, `dtmax`,
  `saveat`, `tstops` and `d_discontinuities` must have the same dimension.
- `abstol` is given per field, as a `NamedTuple` (`(N = 1e-9u"mol", E = 1e-3u"J")`, with an
  array for an array field) or a `CollectionVector` with the same fields. A plain number is
  rejected since its unit would be ambiguous. Without `abstol`, the solver default applies to
  the raw values, i.e. in the units of the problem's `u0`.
- `reltol` is a plain (dimensionless) number.
- Callbacks are not supported yet and throw an error. Other keyword arguments are passed on
  to the solver unchanged.
- Only `solve` is supported: the integrator interface (`init`, `step!`, ...) throws an error.

# Solution
The returned solution holds `sol.u` (a `Vector` of `CollectionVector`s viewing the raw
solution) and `sol.t` (with units). It can be interpolated with a unitful time, `sol(t)`, or
`sol(t, Val{1})` for the derivative. Other properties (`retcode`, `stats`, ...) come from the
raw solution, which is available as `sol.raw`.

!!! note
    Requires SciMLBase to be loaded (e.g. through `using OrdinaryDiffEq`).
"""
function cv_wrapper end
