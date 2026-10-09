using LinearAlgebra, BenchmarkTools
import Unitful
import FlexUnits
import FlexUnits.UnitRegistry as UnitRegistry
import ComponentArrays: ComponentVector
import RecursiveArrayTools: ArrayPartition
import DifferentialEquations as DE
using HeterogeneousArrays

r0_raw, v0_raw = [1131.34, -2282.34, 6672.42], [-5.64, 4.30, 2.42]
μ_raw, Δt_raw = 398600.44, 3600.0
r0_u, v0_u = r0_raw * Unitful.u"km", v0_raw * Unitful.u"km/s"
μ_u, Δt_u = μ_raw * Unitful.u"km^3/s^2", Δt_raw * Unitful.u"s"
r0_f, v0_f = r0_raw * UnitRegistry.u"km", v0_raw * UnitRegistry.u"km/s"
μ_f, Δt_f = μ_raw * UnitRegistry.u"km^3/s^2", Δt_raw * UnitRegistry.u"s"


function f_named!(dy, y, μ, t)
    r_mag = norm(y.r)
    dy.r .= y.v
    dy.v .= -μ .* y.r ./ r_mag^3
end

prob = DE.ODEProblem(f_named!, HeterogeneousVector(r = r0_f, v = v0_f), (0.0UnitRegistry.u"s", Δt_f), μ_f)

DE.solve(prob; alg = DE.Vern8(), dt = 1e-3UnitRegistry.u"s")
