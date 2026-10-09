using LinearAlgebra, BenchmarkTools
import Unitful
import ComponentArrays: ComponentVector
import RecursiveArrayTools
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
r0_raw = [1131.34, -2282.34, 6672.42]
v0_raw = [-5.64, 4.30, 2.42]
μ_raw = 398600.44
Δt_raw = 3600.0*100 # 100 hours
n_objects = 3

r0_unitful = r0_raw * Unitful.u"km"
v0_unitful = v0_raw * Unitful.u"km/s"
μ_unitful = μ_raw * Unitful.u"km^3/s^2"
Δt_unitful = Δt_raw * Unitful.u"s"

tspan_raw = (0.0, Δt_raw)
tspan_unitful = (0.0 * Unitful.u"s", Δt_unitful)

function named_initial_conditions(unit_handling::Symbol)
    if unit_handling === :none
        return (r0_raw, v0_raw, μ_raw, Δt_raw)
    elseif unit_handling === :unitful
        return (r0_unitful, v0_unitful, μ_unitful, Δt_unitful)
    else
        error("Unknown unit handling: $unit_handling")
    end
end

function f_component_alloc(y, μ, t)
    r_mag = norm(y.r)
    dr = y.v
    dv = -μ .* y.r ./ r_mag^3
    return ComponentVector(r = dr, v = dv)
end

function f_component_inplace!(dy, y, μ, t)
    r_mag = norm(y.r)
    dy.r .= y.v
    dy.v .= -μ .* y.r ./ r_mag^3
    dy
end

function f_arraypartition_alloc(y, μ, t)
    r_mag = norm(y.x[1])
    dr = y.x[2]
    dv = -μ .* y.x[1] ./ r_mag^3
    return ArrayPartition(dr, dv)
end

function f_arraypartition_inplace!(dy, y, μ, t)
    r_mag = norm(y.x[1])
    dy.x[1] .= y.x[2]
    dy.x[2] .= -μ .* y.x[1] ./ r_mag^3
    dy
end

function f_heterogeneous_alloc(y, μ, t)
    r_mag = norm(y.r)
    dr = y.v
    dv = -μ .* y.r ./ r_mag^3
    return HeterogeneousVector(r = dr, v = dv)
end

function f_heterogeneous_inplace!(dy, y, μ, t)
    r_mag = norm(y.r)
    dy.r .= y.v
    dy.v .= -μ .* y.r ./ r_mag^3
    dy
    return dy
end

function build_case(array_structure::Symbol, unit_handling::Symbol, ode_interface::Symbol)
    r, v, μ, dt = named_initial_conditions(unit_handling)
    tspan = unit_handling === :none ? tspan_raw : tspan_unitful

    if array_structure === :componentvector
        u0 = ComponentVector(r = r, v = v)
        f = ode_interface === :allocating ? f_component_alloc : f_component_inplace!
    elseif array_structure === :arraypartition
        u0 = ArrayPartition(r, v)
        f = ode_interface === :allocating ? f_arraypartition_alloc :
            f_arraypartition_inplace!
    elseif array_structure === :heterogeneousvector
        u0 = HeterogeneousVector(r = r, v = v)
        f = ode_interface === :allocating ? f_heterogeneous_alloc : f_heterogeneous_inplace!
    else
        error("Unknown array structure: $array_structure")
    end
    return DE.ODEProblem(f, u0, tspan, μ), dt
end

array_structures = [
    (:componentvector, "ComponentVector"),
    (:arraypartition, "ArrayPartition"),
    (:heterogeneousvector, "HeterogeneousVector")
]

unit_handlings = [
    (:none, "None"),
    (:unitful, "Unitful")
]

ode_interfaces = [
    (:allocating, "Allocating"),
    (:inplace, "Non-allocating")
]

cases = NamedTuple{(:array_label, :unit_label, :interface_label, :prob, :dt)}[]
for (array_symbol, array_label) in array_structures
    for (unit_symbol, unit_label) in unit_handlings
        for (interface_symbol, interface_label) in ode_interfaces
            if array_symbol === :componentvector && unit_symbol === :unitful &&
               interface_symbol === :allocating
                # Incompatible combination which always yields an error because of lack of interface compatibility
                continue
            end
            prob, dt = build_case(array_symbol, unit_symbol, interface_symbol)
            push!(cases,
                (array_label = array_label, unit_label = unit_label,
                    interface_label = interface_label, prob = prob, dt = dt))
        end
    end
end

# 2. Execution
header_array = rpad("Array structure", 22)
header_units = rpad("Unit handling", 14)
header_iface = rpad("ODE interface", 18)
header_min = lpad("Min (ms)", 12)
header_std = lpad("StdErr (ms)", 15)
header_allocs = lpad("Allocs", 12)
header_mem = lpad("Memory", 15)

println("\n" * "─" ^ 110)
println(header_array, header_units, header_iface,
    header_min, header_std, header_allocs, header_mem)
println("─" ^ 110)

for case in cases
    # Warm-up to avoid precompilation from influencing results
    DE.solve(case.prob; alg = DE.Tsit5(), adaptive = true, dt = case.dt)

    trial = @benchmark DE.solve($(case.prob); alg = DE.Tsit5(), adaptive = true, dt = $(case.dt)) samples=100

    t_min = minimum(trial).time / 1e6
    stderror_ms = (std(trial.times) / sqrt(length(trial.times))) / 1e6
    allocs = trial.allocs
    memory = trial.memory
    mem_str = memory < 1024 ? "$(memory) B" : "$(round(memory/1024, digits=1)) KiB"

    println(rpad(case.array_label, 22),
        rpad(case.unit_label, 14),
        rpad(case.interface_label, 18),
        lpad(format_val(t_min), 12),
        lpad(format_val(stderror_ms), 15),
        lpad(allocs, 12),
        lpad(mem_str, 15))
end
println("─" ^ 110)
