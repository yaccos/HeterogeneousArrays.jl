using HeterogeneousArrays
using ComponentArrays
using Test
import DifferentialEquations as DE
using Unitful
using RecursiveArrayTools
using LinearAlgebra


@testset "DE Integration: Heterogeneous vs ComponentArrays" begin
    # 1. Setup Parameters and Time
    p = (σ = 10.0, ρ = 28.0, β = 8/3)
    tspan = (0.0, 0.01)
    
    # Define the physics/system once to be reused
    function lorenz!(du, u, p, t)
        # Both packages support getproperty/dot access
        x, y = u.coord
        z = u.z

        du.coord[1] = p.σ * (y - x)
        du.coord[2] = x * (p.ρ - z) - y
        du.z = x * y - p.β * z
        return nothing
    end

    # 2. Initialize both types
    u0_het = HeterogeneousVector(coord = [1.0, 0.0], z = 0.0)
    u0_comp = ComponentVector(coord = [1.0, 0.0], z = 0.0)

    # 3. Warm up (to JIT compile the ODE solver for both types)
    prob_het = DE.ODEProblem(lorenz!, u0_het, tspan, p)
    prob_comp = DE.ODEProblem(lorenz!, u0_comp, tspan, p)
    
    sol_het = DE.solve(prob_het, DE.Tsit5())
    sol_comp = DE.solve(prob_comp, DE.Tsit5())

    # 4. Measure Allocations & Correctness
    # We use @allocated on a fresh solve to ensure zero/minimal overhead
    allocs_het = @allocated DE.solve(prob_het, DE.Tsit5(), reltol=1e-6)
    allocs_comp = @allocated DE.solve(prob_comp, DE.Tsit5(), reltol=1e-6)

    # --- TESTS ---

    # A. Verify Result Parity
    # Check that the final state is numerically identical
    @test sol_het.u[end].coord ≈ sol_comp.u[end].coord
    @test sol_het.u[end].z ≈ sol_comp.u[end].z

    # B. Verify Performance Parity
    # HeterogeneousArrays should not be significantly "heavier" than ComponentArrays
    # for homogeneous Float64 data.
    @testset "Allocation Comparison" begin
        # This is very worying if HeterogeneousArrays allocates more than 2.5x the allocations of ComponentArrays
        # Double check with Jakob 
        @test allocs_het <= allocs_comp * 2.5
        @info "Allocations - Heterogeneous: $allocs_het, Component: $allocs_comp"
    end

    # C. Verify Type Integrity
    @test sol_het.u[end] isa HeterogeneousVector
    @test sol_comp.u[end] isa ComponentVector
end


@testset "Unitful vs Unitless Consistency" begin

    r0 = [1131.340, -2282.343, 6672.423]Unitful.u"km"
    v0 = [-5.64305, 4.30333, 2.42879]Unitful.u"km/s"
    Δt = 86400.0 * 365Unitful.u"s"
    μ = 398600.4418Unitful.u"km^3/s^2"
    rv0 = RecursiveArrayTools.ArrayPartition(r0, v0)


    function f(dy, y, μ, t)
        r = LinearAlgebra.norm(y.x[1])
        dy.x[1] .= y.x[2]
        dy.x[2] .= -μ .* y.x[1] / r^3
    end

    prob = DE.ODEProblem(f, rv0, (0.0Unitful.u"s", Δt), μ)
    sol_unit = DE.solve(prob, DE.Vern8())


    # Initial conditions (km and km/s)
    r0 = [1131.340, -2282.343, 6672.423]
    v0 = [-5.64305, 4.30333, 2.42879]
    Δt = 86400.0 * 365 # One year in seconds
    μ  = 398600.4418   # Earth's gravitational parameter

    rv0 = RecursiveArrayTools.ArrayPartition(r0, v0)


    # Define the problem without Unitful wrappers
    prob = DE.ODEProblem(f, rv0, (0.0, Δt), μ)
    sol = DE.solve(prob, DE.Vern8())



    # 1. Extract values from the Unitful solution and strip units
    # We compare the final states (the last time step)
    final_state_unit = sol_unit.u[end]
    
    # ustrip can be applied to the ArrayPartition or its components
    # Here we convert the entire state to a raw float array
    raw_r_from_unit = ustrip.(final_state_unit.x[1])
    raw_v_from_unit = ustrip.(final_state_unit.x[2])
    
    # 2. Extract values from the Unitless solution
    final_state_raw = sol.u[end]
    raw_r = final_state_raw.x[1]
    raw_v = final_state_raw.x[2]

    # 3. Perform the comparison
    # isapprox is better than == for ODE solvers due to floating point noise
    @test raw_r_from_unit ≈ raw_r atol=1e-8
    @test raw_v_from_unit ≈ raw_v atol=1e-8
    
    # Optionally check if the time steps are identical
    @test ustrip.(sol_unit.t) ≈ sol.t
end


using Test
using HeterogeneousArrays
using Unitful
import RecursiveArrayTools
import DifferentialEquations as DE
import LinearAlgebra: norm

@testset "ODE Integration Consistency" begin
    # --- Setup ---
    r0_raw = [1131.340, -2282.343, 6672.423]
    v0_raw = [-5.64305, 4.30333, 2.42879]
    μ_raw  = 398600.4418
    Δt_raw = 3600.0
    
    r0_u = r0_raw * u"km"
    v0_u = v0_raw * u"km/s"
    μ_u  = μ_raw  * u"km^3/s^2"
    Δt_u = Δt_raw * u"s"

    # ODE functions
    function f_part(dy, y, μ, t)
        r_mag = norm(y.x[1])
        dy.x[1] .= y.x[2]
        dy.x[2] .= -μ .* y.x[1] ./ r_mag^3
    end

    function f_het(dy, y, μ, t)
        r_mag = norm(y.r)
        dy.r .= y.v
        dy.v .= -μ .* y.r ./ r_mag^3
    end

    
    common_kwargs = (
        alg = DE.Vern8(),
        dt = 1e-3,
    )

    # --- 1. Solve Standard ArrayPartition (No Units) ---
    rv0_std = RecursiveArrayTools.ArrayPartition(r0_raw, v0_raw)
    prob_std = DE.ODEProblem(f_part, rv0_std, (0.0, Δt_raw), μ_raw)
    sol_std = DE.solve(prob_std; common_kwargs...)

    # --- 2. Solve HeterogeneousVector (Units) ---
    hv0_het = HeterogeneousVector(r = r0_u, v = v0_u)
    prob_het = DE.ODEProblem(f_het, hv0_het, (0.0u"s", Δt_u), μ_u)
    sol_het = DE.solve(prob_het; common_kwargs...)

    # --- 3. Solve ArrayPartition (Units) ---
    rv0_unit = RecursiveArrayTools.ArrayPartition(r0_u, v0_u)
    prob_unit = DE.ODEProblem(f_part, rv0_unit, (0.0u"s", Δt_u), μ_u)
    sol_unit = DE.solve(prob_unit; common_kwargs...)

    # --- Tests ---
    
    # Extract final states as raw vectors for comparison
    final_std = vcat(sol_std[end].x[1], sol_std[end].x[2])
    final_het = vcat(ustrip.(sol_het[end].r), ustrip.(sol_het[end].v))
    final_unit = vcat(ustrip.(sol_unit[end].x[1]), ustrip.(sol_unit[end].x[2]))

    @testset "Numerical Accuracy" begin
        # Compare HeterogeneousVector with Standard Float64 ArrayPartition
        # We expect very high agreement (close to machine epsilon)
        @test final_het ≈ final_std rtol=1e-13
        
        # Compare HeterogeneousVector with Unitful ArrayPartition
        @test final_het ≈ final_unit rtol=1e-13
    end

    @testset "Structure Preservation" begin
        # Verify that the solver returned the correct custom type
        @test sol_het[end] isa HeterogeneousVector
        @test hasproperty(sol_het[end], :r)
        @test hasproperty(sol_het[end], :v)
        
        # Verify units are preserved in the solution
        @test unit(sol_het[end].r[1]) == u"km"
        @test unit(sol_het[end].v[1]) == u"km/s"
    end
end


using Test, HeterogeneousArrays, Unitful, DifferentialEquations, LinearAlgebra

@testset "Simple Pendulum Test" begin
    # --- 1. Parameters & Initial Conditions ---
    # L: length of pendulum, g: gravity
    L = 1.0u"m"
    g = 9.81u"m/s^2"
    
    # Initial state: θ = 45 degrees, ω = 0 rad/s
    # Note: we use 0.0u"s^-1" to ensure the type is a Quantity
    u0 = HeterogeneousVector(θ = 0.785, ω = 0.0u"s^-1")
    tspan = (0.0u"s", 5.0u"s")

    # --- 2. ODE Function ---
    function pendulum_f!(du, u, p, t)
        L, g = p
        # θ_dot = ω
        du.θ = u.ω
        # ω_dot = - (g/L) * sin(θ)
        du.ω = -(g / L) * sin(u.θ)
    end

    # # --- 3. Solver Setup ---
    # # We use a custom norm to handle the mix of Float64 (angle) and Quantity (ω)
    # const PENDULUM_NORM = (u, t) -> maximum(abs.(ustrip.(u)))

    prob = ODEProblem(pendulum_f!, u0, tspan, (L, g))
    
    # Solve with high precision
    # sol = solve(prob, Vern8(), reltol=1e-12, abstol=1e-12, internalnorm=PENDULUM_NORM)
    sol = solve(prob, Vern8())

    # --- 4. Verification ---
    
    @testset "Physical Consistency" begin
        # 1. Check types
        @test sol.u[end] isa HeterogeneousVector
        @test sol.u[end].ω isa Unitful.Quantity
        
        # 2. Conservation of Energy check
        # E = (1/2) * L^2 * ω^2 + g * L * (1 - cos(θ))
        function energy(u)
            kin = 0.5 * ustrip(L)^2 * ustrip(u.ω)^2
            pot = ustrip(g) * ustrip(L) * (1 - cos(u.θ))
            return kin + pot
        end

        E0 = energy(sol.u[1])
        Ef = energy(sol.u[end])
        
        # Energy should be conserved in a simple pendulum
        @test E0 ≈ Ef rtol=1e-3
    end

    @testset "Named Field Access" begin
        # Verify we can access fields after integration
        @test sol.u[end].θ isa Float64
        @test unit(sol.u[end].ω) == u"s^-1"
    end
end
