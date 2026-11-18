using HeterogeneousArrays
using Unitful
using LSODA: lsoda
import DifferentialEquations as DE

u0 = CollectionVector(A = 1.0u"mol/L", B = 0.0u"mol/L", C = 0.0u"mol/L")
p = (k1 = 0.04u"s^-1", k2 = 1.0e4u"L/(mol*s)", k3 = 1.08e11u"L/(mol*hr)")

function robertson!(du, u, p, t)
    du.A = -p.k1 * u.A + p.k2 * u.B * u.C
    du.B = p.k1 * u.A - p.k2 * u.B * u.C - p.k3 * u.B^2
    du.C = p.k3 * u.B^2
    return nothing
end

tspan = (0.0u"s", 1.0e5u"s")
prob = DE.ODEProblem(robertson!, u0, tspan, p)

abstol = (A = 1.0e-8u"mol/L", B = 1.0e-14u"mol/L", C = 1.0e-8u"mol/L")
reltol = 1.0e-8

saveat = [0.0u"s", 1.0u"s", 1.0u"minute", 1.0u"hr", 1.0u"d"]
sol = DE.solve(prob, cv_wrapper(lsoda()); abstol, reltol, saveat)

println("retcode: ", sol.retcode)
println("\nSaved states:")
for (t, u) in zip(sol.t, sol.u)          # sol.t has units, sol.u are CollectionVectors
    println("  t = ", rpad(t, 10), "  A = ", round(u"mol/L", u.A; sigdigits = 5),
        "   B = ", round(u"mol/L", u.B; sigdigits = 5),
        "   C = ", round(u"mol/L", u.C; sigdigits = 5))
end

# Mass is conserved: A + B + C stays 1 mol/L
uend = sol.u[end]
println("\nA + B + C at the end: ", uend.A + uend.B + uend.C)
