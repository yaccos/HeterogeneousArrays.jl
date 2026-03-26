import Unitful, RecursiveArrayTools, DifferentialEquations as DE
import LinearAlgebra
using HeterogeneousArrays
using Unitful

function f(dy, y, μ, t)
    r = LinearAlgebra.norm(y.x[1])
    dy.x[1] .= y.x[2]
    dy.x[2] .= -μ .* y.x[1] / r^3
end

function f_heterogeneous(dy, y, μ, t)
    r_mag = LinearAlgebra.norm(y.r)

    # In-place broadcasting works because of your custom copyto! 
    # and unpack_broadcast implementation
    dy.r .= y.v
    dy.v .= -μ .* y.r ./ r_mag^3
end

r0 = [1131.340, -2282.343, 6672.423]Unitful.u"km"
v0 = [-5.64305, 4.30333, 2.42879]Unitful.u"km/s"
Δt = 3600.0 * 1Unitful.u"s"
μ = 398600.4418Unitful.u"km^3/s^2"

rv0 = RecursiveArrayTools.ArrayPartition(r0, v0)
prob = DE.ODEProblem(f, rv0, (0.0Unitful.u"s", Δt), μ)
sol_unit = DE.solve(prob, DE.Vern8(), dt = 1e-3)

hv0_u = HeterogeneousVector(r = r0_u, v = v0_u)
prob_u = DE.ODEProblem(f_heterogeneous, hv0_u, (0.0Unitful.u"s", Δt_u), μ_u)
sol_het = DE.solve(prob_u, DE.Vern8(), dt = 1e-3)

# Initial conditions (km and km/s)
r0 = [1131.340, -2282.343, 6672.423]
v0 = [-5.64305, 4.30333, 2.42879]
Δt = 3600.0  # One year in seconds
μ = 398600.4418   # Earth's gravitational parameter

rv0 = RecursiveArrayTools.ArrayPartition(r0, v0)
prob = DE.ODEProblem(f, rv0, (0.0, Δt), μ)
sol = DE.solve(prob, DE.Vern8(), dt = 1e-3)

print(sol_unit[end][6] - sol_het[end].v[3])
print(sol[end][6] - ustrip(sol_het[end].v[3]))
