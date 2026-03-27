using LinearAlgebra, Unitful, BenchmarkTools
import ComponentArrays: ComponentVector
import RecursiveArrayTools: ArrayPartition
import DifferentialEquations as DE
using HeterogeneousArrays

# --- Description ---
println("\nOrbital Mechanics ODE Benchmark")
println("This benchmark compares integration performance, statistical error, and memory efficiency")
println("across HeterogeneousVector, ArrayPartition, and ComponentVector.")

# Formatting helper for 4 decimal places without Printf
function format_val(val; digits = 4)
    s = string(round(val, digits = digits))
    if !contains(s, '.')
        return s * "." * ("0"^digits)
    end
    parts = split(s, '.')
    return parts[1] * "." * rpad(parts[2], digits, "0")
end

# 1. Setup
r0_raw, v0_raw = [1131.34, -2282.34, 6672.42], [-5.64, 4.30, 2.42]
μ_raw, Δt_raw = 398600.44, 3600.0
r0_u, v0_u = r0_raw * u"km", v0_raw * u"km/s"
μ_u, Δt_u = μ_raw * u"km^3/s^2", Δt_raw * u"s"

function f_part!(dy, y, μ, t)
    r_mag = norm(y.x[1])
    dy.x[1] .= y.x[2]
    dy.x[2] .= -μ .* y.x[1] ./ r_mag^3
end

function f_named!(dy, y, μ, t)
    r_mag = norm(y.r)
    dy.r .= y.v
    dy.v .= -μ .* y.r ./ r_mag^3
end

common_args = (alg = DE.Vern8(), dt = 1e-3)

probs = [
    ("1. HeterogeneousVector (No Units)",
        DE.ODEProblem(f_named!, HeterogeneousVector(r = r0_raw, v = v0_raw), (0.0, Δt_raw), μ_raw)),
    ("2. HeterogeneousVector (Units)",
        DE.ODEProblem(f_named!, HeterogeneousVector(r = r0_u, v = v0_u), (0.0u"s", Δt_u), μ_u)),
    ("3. ArrayPartition (No Units)",
        DE.ODEProblem(f_part!, ArrayPartition(r0_raw, v0_raw), (0.0, Δt_raw), μ_raw)),
    ("4. ArrayPartition (Units)",
        DE.ODEProblem(f_part!, ArrayPartition(r0_u, v0_u), (0.0u"s", Δt_u), μ_u)),
    ("5. ComponentVector (No Units)",
        DE.ODEProblem(f_named!, ComponentVector(r = r0_raw, v = v0_raw), (0.0, Δt_raw), μ_raw)),
    ("6. ComponentVector (Units)",
        DE.ODEProblem(f_named!, ComponentVector(r = r0_u, v = v0_u), (0.0u"s", Δt_u), μ_u))
]

# 2. Execution
header_strategy = rpad("Implementation Strategy", 34)
header_min = lpad("Min (ms)", 12)
header_std = lpad("StdErr (ms)", 15)
header_allocs = lpad("Allocs", 12)
header_mem = lpad("Memory", 15)

println("\n" * "─" ^ 88)
println(header_strategy, header_min, header_std, header_allocs, header_mem)
println("─" ^ 88)

for (label, prob) in probs
    # Warmup
    DE.solve(prob; common_args...)

    # Run Benchmark
    # We use $ to interpolate variables for accuracy
    trial = @benchmark DE.solve($prob; $common_args...) samples=100

    # Extract stats
    t_min = minimum(trial).time / 1e6   # ns to ms

    # Standard Error of the Mean = StdDev / sqrt(N)
    stderror_ms = (std(trial.times) / sqrt(length(trial.times))) / 1e6

    allocs = trial.allocs
    memory = trial.memory # in bytes

    # Format memory string
    mem_str = memory < 1024 ? "$(memory) B" : "$(round(memory/1024, digits=1)) KiB"

    println(rpad(label, 34),
        lpad(format_val(t_min), 12),
        lpad(format_val(stderror_ms), 15),
        lpad(allocs, 12),
        lpad(mem_str, 15))
end
println("─" ^ 88)
