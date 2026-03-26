using HeterogeneousArrays
using ComponentArrays
using LinearAlgebra
import DifferentialEquations as DE

println("--- HeterogeneousArrays Example: Lorenz System & Comparison ---")
println("Comparing HeterogeneousVector with ComponentVector for homogeneous data.")

# 1. Setup Parameters and Time
# The Lorenz system is a classic test for ODE solvers
p = (σ = 10.0, ρ = 28.0, β = 8/3)
tspan = (0.0, 10.0) # Increased tspan slightly for a more meaningful solve

# Define the physics/system
# Both packages support getproperty/dot access, allowing for identical kernel code
function lorenz!(du, u, p, t)
    x, y = u.coord
    z = u.z

    du.coord[1] = p.σ * (y - x)
    du.coord[2] = x * (p.ρ - z) - y
    du.z = x * y - p.β * z
    return nothing
end

# 2. Initialize both types with identical data
u0_dict = (coord = [1.0, 0.0], z = 0.0)
u0_het = HeterogeneousVector(; u0_dict...)
u0_comp = ComponentVector(; u0_dict...)

# 3. Create ODE Problems
prob_het = DE.ODEProblem(lorenz!, u0_het, tspan, p)
prob_comp = DE.ODEProblem(lorenz!, u0_comp, tspan, p)

# 4. Solve and Measure
println("\nSolving Lorenz system...")

# Define a helper to ensure identical call signatures
function run_solve(prob)
    DE.solve(prob, DE.Tsit5(), reltol = 1e-6)
end

# --- WARMUP ---
# This forces compilation of the solver, the kernel, and the tolerance logic
sol_het = run_solve(prob_het)
sol_comp = run_solve(prob_comp)

# --- MEASUREMENT ---
# Now we measure. Because run_solve was already compiled for these probs,
# we get consistent numbers even on the "first" include of a session.
allocs_het = @allocated run_solve(prob_het)
allocs_comp = @allocated run_solve(prob_comp)

# 5. Report Findings
println("\n--- Results ---")
println("Numerical Parity (coord): ", sol_het.u[end].coord ≈ sol_comp.u[end].coord)
println("Numerical Parity (z):     ", sol_het.u[end].z ≈ sol_comp.u[end].z)

println("\n--- Memory Overhead ---")
println("HeterogeneousVector allocations: ", allocs_het, " bytes")
println("ComponentVector allocations:     ", allocs_comp, " bytes")
println("Overhead Ratio:                  ", round(allocs_het / allocs_comp, digits = 2), "x")

println("\nNote: For purely homogeneous Float64 data, ComponentArrays is highly optimized.")
println("HeterogeneousArrays shines when fields have different concrete types or units.")
