using HeterogeneousArrays
using Unitful
using LinearAlgebra
import DifferentialEquations as DE

println("--- HeterogeneousArrays Example: Pendulum with Mixed Units ---")

# 1. Setup Parameters (Meters, Seconds, Degrees)
L = 1.0u"m"
g = 9.81u"m/s^2"

# Initial state: Angle (Float64) and Angular Velocity (Quantity)
u0 = HeterogeneousVector(θ = 0.785, ω = 0.0u"s^-1")
tspan = (0.0u"s", 5.0u"s")

# 2. Define the Physics
function pendulum_f!(du, u, p, t)
    L, g = p
    du.θ = u.ω
    du.ω = -(g / L) * sin(u.θ)
end

# 3. Solve using a Struct-Based Tolerance
# This ensures θ is solved to 1e-8 absolute error and ω to 1e-8 s^-1
abstol_struct = 1e-8 .* oneunit.(u0)
prob = DE.ODEProblem(pendulum_f!, u0, tspan, (L, g))
sol = DE.solve(prob, DE.Vern8(), abstol = abstol_struct)

# 4. Display Results
println("Final Angle (θ): ", sol.u[end].θ, " rad")
println("Final Velocity (ω): ", sol.u[end].ω)

# Verification of Energy Conservation
function energy(u)
    kin = 0.5 * ustrip(L)^2 * ustrip(u.ω)^2
    pot = ustrip(g) * ustrip(L) * (1 - cos(u.θ))
    return kin + pot
end

E0, Ef = energy(sol.u[1]), energy(sol.u[end])
println("Energy Conservation Error: ", abs(E0 - Ef))
