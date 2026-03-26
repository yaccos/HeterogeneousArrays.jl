using HeterogeneousArrays
using Unitful
using BenchmarkTools
using LinearAlgebra
import ComponentArrays: ComponentVector
import RecursiveArrayTools: ArrayPartition
import DifferentialEquations as DE

println("--- HeterogeneousArrays Example: Orbital Mechanics & Units ---")
println("Comparing integration performance of mixed-unit vectors across packages.")

# 1. Setup Physical Constants (Earth Two-Body Problem)
r0_raw = [1131.340, -2282.343, 6672.423] # km
v0_raw = [-5.64305, 4.30333, 2.42879]    # km/s
μ_raw = 398600.4418                   # km^3/s^2
Δt_raw = 3600.0                        # s

# Unitful versions
r0_u, v0_u = r0_raw * u"km", v0_raw * u"km/s"
μ_u, Δt_u = μ_raw * u"km^3/s^2", Δt_raw * u"s"

# 2. Define ODE Kernels
# Kernel for positional/indexed access (ArrayPartition)
f_part(dy, y, μ, t) = begin
    r_mag = norm(y.x[1])
    dy.x[1] .= y.x[2]
    dy.x[2] .= -μ .* y.x[1] ./ r_mag^3
end

# Kernel for named access (HeterogeneousVector & ComponentVector)
f_named(dy, y, μ, t) = begin
    r_mag = norm(y.r)
    dy.r .= y.v
    dy.v .= -μ .* y.r ./ r_mag^3
end

# 3. Define the Problems
common_args = (alg = DE.Vern8(), dt = 1e-3)

probs = [
    ("HeterogeneousVector (No Units)",
        DE.ODEProblem(f_named, HeterogeneousVector(r = r0_raw, v = v0_raw), (0.0, Δt_raw), μ_raw)),
    ("HeterogeneousVector (Units)",
        DE.ODEProblem(f_named, HeterogeneousVector(r = r0_u, v = v0_u), (0.0u"s", Δt_u), μ_u)),
    ("ArrayPartition (No Units)",
        DE.ODEProblem(f_part, ArrayPartition(r0_raw, v0_raw), (0.0, Δt_raw), μ_raw)),
    ("ArrayPartition (Units)",
        DE.ODEProblem(f_part, ArrayPartition(r0_u, v0_u), (0.0u"s", Δt_u), μ_u)),
    ("ComponentVector (No Units)",
        DE.ODEProblem(f_named, ComponentVector(r = r0_raw, v = v0_raw), (0.0, Δt_raw), μ_raw)),
    ("ComponentVector (Units)",
        DE.ODEProblem(f_named, ComponentVector(r = r0_u, v = v0_u), (0.0u"s", Δt_u), μ_u))
]

# 4. Performance Comparison
println("\n" * "─" ^ 65)
println(rpad("Implementation Strategy", 35), lpad("Time (ms)", 15))
println("─" ^ 65)

for (label, prob) in probs
    # Warmup
    DE.solve(prob; common_args...)

    # Benchmarking (Minimum time)
    t = @belapsed DE.solve($prob; $common_args...)

    # Simple formatting for the example output
    ms_str = string(round(t * 1000, digits = 4))
    println(rpad(label, 35), lpad(ms_str, 15))
end
println("─" ^ 65)

# 5. Narrative Conclusion
println("\nOBSERVATIONS:")
println("1. HeterogeneousVector maintains nearly identical performance with or without units.")
println("2. Conventional structures (ArrayPartition/ComponentVector) often face a 'Unitful Tax'.")
println("3. This is because HeterogeneousVector's broadcasting is optimized for type-stability")
println("   even when individual fields have different concrete types.")
