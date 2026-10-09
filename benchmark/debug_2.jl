using FlexUnits, .UnitRegistry
using OrdinaryDiffEq
# using DifferentialEquations
using StaticArrays
using Plots
using BenchmarkTools
using LinearAlgebra
using ComponentArrays
using HeterogeneousArrays

function acceleration!(du::AbstractVector{<:Quantity},u::AbstractVector{<:Quantity}, p::AbstractVector{<:Quantity}, t)
    fd = -sign(u.v)*0.5*p.ρ*u.v^2*p.Cd*p.A
    du.v = fd/p.m - p.g
    du.h = u.v
    return du
end

u0 = HeterogeneousVector(v=0.0u"m/s", h=100u"m")
p  = HeterogeneousVector(Cd=1.0u"", A=0.1u"m^2", ρ=1.0u"kg/m^3", m=50u"kg", g=9.81u"m/s^2")

tspan = (0.0u"s", 10.0u"s")
prob = ODEProblem{true, OrdinaryDiffEq.SciMLBase.NoSpecialize}(acceleration!, u0, tspan, p,
 abstol=HeterogeneousVector(v=1e-6u"m/s", h=1e-6u"m"), reltol=HeterogeneousVector(v=1e-6u"m/s", h=1e-6u"m"))
sol = solve(prob, Tsit5())
plt = plot(ustrip.(sol.t), [ustrip(u.v) for u in sol.u], label="Tsit5")
