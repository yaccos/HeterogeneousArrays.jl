# Check ForwardDiff through a complex CollectionVector field and cv_wrapper.
#
# Run from the repository root with:
#   julia --project=examples examples/cv_wrapper_complex_autodiff_check.jl
#
# z = x + iy occupies two real storage slots. The Jacobian below is with respect to x and y,
# not a complex derivative: conjugation makes this ODE non-holomorphic.

using HeterogeneousArrays
using HeterogeneousArrays: rawdata
using ForwardDiff: ForwardDiff, Dual
using ADTypes: AutoFiniteDiff, AutoForwardDiff
using LinearAlgebra: norm
import DifferentialEquations as DE

u0 = CollectionVector(z = 1.0 + 2.0im)
@assert rawdata(u0) == [1.0, 2.0]
@assert eltype(rawdata(u0)) == Float64

# The eigenvalues are approximately -2 and -2000, making the real system stiff.
p = (a = 1001.0, b = 1.0, c = 999.0)
J = [-p.a + p.c -p.b; p.b -p.a - p.c]
tspan = (0.0, 1.0)

mutable struct Spy
    plain::Int
    dual_u::Int
    dual_t::Int
    widths::Set{Int}
    seeded::Vector{Bool}
    max_jac_error::Float64
end
Spy() = Spy(0, 0, 0, Set{Int}(), falses(2), 0.0)

partials_matrix(x::AbstractVector{<:Dual{T, V, N}}) where {T, V, N} =
    [ForwardDiff.partials(xi, j) for xi in x, j in 1:N]

function spied(spy)
    return function (du, u, p, t)
        du.z = (-p.a + p.b * im) * u.z + p.c * conj(u.z)
        U, DU = rawdata(u), rawdata(du)
        if eltype(U) <: Dual
            spy.dual_u += 1
            @assert u.z isa Complex{<:Dual}
            @assert du.z isa Complex{<:Dual}
            Pu, Pdu = partials_matrix(U), partials_matrix(DU)
            push!(spy.widths, size(Pu, 2))
            spy.seeded .|= vec(any(!iszero, Pu; dims = 2))
            err = norm(Pdu - J * Pu) / (norm(J) * norm(Pu))
            spy.max_jac_error = max(spy.max_jac_error, err)
        elseif t isa Dual
            spy.dual_t += 1
            @assert iszero(partials_matrix(DU)) "the ODE has no explicit time dependence"
        else
            spy.plain += 1
        end
        return nothing
    end
end

cases = [
    ("ForwardDiff, chunk size 1", DE.Rodas5P(autodiff = AutoForwardDiff(chunksize = 1)), 1),
    ("ForwardDiff, chunk size 2", DE.Rodas5P(autodiff = AutoForwardDiff(chunksize = 2)), 2),
    ("finite differences", DE.Rodas5P(autodiff = AutoFiniteDiff()), 0),
]

# The exact solution of the real system is exp(J*t) * [real(z0), imag(z0)].
exact = exp(J * tspan[end]) * rawdata(u0)
println("Complex ODE: z' = (-a + ib)z + c*conj(z)")
println("Initial real storage: ", rawdata(u0))
println("Real Jacobian:\n", J)
println()

for (name, alg, width) in cases
    spy = Spy()
    # FullSpecialize prevents SciML from wrapping the original unit-free RHS for Float64
    # before cv_wrapper can pass it Dual-valued CollectionVectors.
    f = DE.ODEFunction{true, DE.SciMLBase.FullSpecialize}(spied(spy))
    prob = DE.ODEProblem(f, u0, tspan, p)
    sol = DE.solve(prob, cv_wrapper(alg); reltol = 1e-9)
    @assert sol.retcode == DE.ReturnCode.Success "$name: solve failed"
    solution_error = norm(rawdata(sol.u[end]) - exact)
    @assert solution_error < 1e-6 "$name: solution differs from the exact real-system solution"
    if width > 0
        @assert spy.dual_u > 0 "$name: no Dual state reached the ODE"
        @assert spy.widths == Set([width]) "$name: unexpected ForwardDiff chunk size"
        @assert all(spy.seeded) "$name: not all real and imaginary coordinates were differentiated"
        @assert spy.max_jac_error < 1e-12 "$name: Jacobian partials differ from the analytic result"
    else
        @assert spy.dual_u == spy.dual_t == 0 "$name: finite differences unexpectedly used Duals"
    end
    println(name, ": plain calls=", spy.plain, ", Dual state calls=", spy.dual_u,
        ", Dual time calls=", spy.dual_t, ", widths=", sort!(collect(spy.widths)),
        ", Jacobian relative error=", width > 0 ? spy.max_jac_error : "n/a",
        ", solution error=", solution_error)
end
println("All checks passed.")
