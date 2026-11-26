# ForwardDiff through a complex quantity field with cv_wrapper.
#
# Run from the repository root with:
#   julia --project=examples examples/cv_wrapper_complex_autodiff_check.jl
#
# The holomorphic ODE z' = λz + μz² is solved with z in m and t in s. The complex field is
# stored as its real and imaginary parts, so a Jacobian is taken with respect to these two
# real numbers. If f'(z) = a + ib, that Jacobian is [a -b; b a].

using HeterogeneousArrays, Unitful
using HeterogeneousArrays: rawdata
using ForwardDiff: ForwardDiff, Dual
using ADTypes: AutoFiniteDiff
import DifferentialEquations as DE

p = (λ = (-1000.0 + 10.0im)u"s^-1", μ = (0.5 + 0.25im)u"m^-1*s^-1")
tspan = (0.0u"s", 0.01u"s")
u0 = CollectionVector(z = (1.0 + 2.0im)u"m")

# (a) the storage is a plain real vector: real part, imaginary part
println("storage: ", rawdata(u0), " :: ", typeof(rawdata(u0)))

# (b), (c): record whether Dual-valued states reach the right-hand side, and compare their
# partials with the analytic Jacobian
calls = (dual = Ref(0), max_jacobian_error = Ref(0.0))
function rhs!(du, u, p, t)
    du.z = p.λ * u.z + p.μ * u.z^2
    U, DU = rawdata(u), rawdata(du)
    if eltype(U) <: Dual
        calls.dual[] += 1
        z = complex(ForwardDiff.value.(U)...) * u"m"
        fz = ustrip(u"s^-1", p.λ + 2 * p.μ * z)  # f'(z) = a + ib
        J = [real(fz) -imag(fz); imag(fz) real(fz)]
        P(x) = [ForwardDiff.partials(xi, j) for xi in x, j in 1:ForwardDiff.npartials(x[1])]
        @assert !iszero(P(U)) "the real and imaginary parts are not seeded"
        err = maximum(abs, P(DU) - J * P(U)) / maximum(abs, J)
        calls.max_jacobian_error[] = max(calls.max_jacobian_error[], err)
    end
    return nothing
end

# (d) exact solution
e = exp(p.λ * tspan[2])
z_exact = u0.z * e / (1 - (p.μ / p.λ) * u0.z * (e - 1))

prob = DE.ODEProblem(rhs!, u0, tspan, p)
for (name, alg) in (("ForwardDiff", DE.Rodas5P()),
    ("finite differences", DE.Rodas5P(autodiff = AutoFiniteDiff())))
    calls.dual[], calls.max_jacobian_error[] = 0, 0.0
    sol = DE.solve(prob, cv_wrapper(alg); reltol = 1e-9, abstol = (z = 1e-10u"m",))
    err = abs(sol.u[end].z - z_exact)
    println(name, ": calls with Dual states = ", calls.dual[],
        ", Jacobian relative error = ", calls.max_jacobian_error[],
        ", error at the final time = ", err)
    @assert sol.retcode == DE.ReturnCode.Success
    @assert (calls.dual[] > 0) == (name == "ForwardDiff")
    @assert calls.max_jacobian_error[] < 1e-12
    @assert err < 1e-9u"m"
end
println("All checks passed.")
