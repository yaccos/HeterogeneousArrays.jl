# Welcome to HeterogeneousArrays

HeterogeneousArrays is a Julia package for efficiently storing and operating on heterogeneous data. It introduces a type called `HeterogeneousVector`, a segmented array designed to hold elements of different concrete types while maintaining efficient, type-stable broadcasting.

## Key Features

* **Type-stable heterogeneous state**: Hold fields of different types in a single efficient container
* **Unitful integration**: Enforce physical correctness at compile-time
* **Zero-overhead broadcasting**: Fused broadcast operations maintain type stability
* **ODE-friendly**: Drop-in compatible with some SciML solvers (DifferentialEquations.jl)
* **Physical safety**: Catch dimensional inconsistencies before they propagate

## Installation

The package can be installed with the Julia package manager. From the Julia REPL type `]` and run:

```
pkg> add HeterogeneousArrays
```

Then use it in your code:

```julia
using HeterogeneousArrays
```

## Basic Usage

Create a heterogeneous vector with mixed types and units:

```julia
using HeterogeneousArrays, Unitful

v = HeterogeneousVector(u = 3.1u"m", v = 5.2u"s")

# Access fields
v.u  # 3.1 m
v.v  # 5.2 s

# Type-stable broadcasting
2.0 .* v .+ v
```

## Example: Modeling a Pendulum

`HeterogeneousVector` is ideal for physical systems where state variables have different units. It ensures that your numerical solver respects the underlying physics of your model.

We define a pendulum state with an angle θ (radians) and angular velocity ω (rad/s).

```julia
using HeterogeneousArrays, Unitful, DifferentialEquations

# Define state: angle (dimensionless) and angular velocity (1/s)

u0 = HeterogeneousVector(θ = 0.1u"rad", ω = 0.0u"rad/s")

# Simple Pendulum ODE: θ'' = -(g/L)sin(θ)
function pendulum_eom(du, u, p, t)
    g, L = p
    du.θ = u.ω
    du.ω = -(g/L) * sin(u.θ)
end

params = (9.81u"m/s^2", 1.0u"m")
tspan = (0.0u"s", 10.0u"s")
prob = ODEProblem(pendulum_eom, u0, tspan, params)
sol = solve(prob, Vern8())
```
In SciML solvers (like `DifferentialEquations.jl`), you can provide tolerances. For heterogeneous data, your Absolute Error (`abstol`) must have the same units as your state fields.

```julia
# Absolute tolerance must match the dimensions of u0
abstol_struct = 1e-8 .* oneunit.(u0)
sol = solve(prob, Vern8(), abstol = abstol_struct)
```

## Compatibility with DifferentialEquations.jl

While `HeterogeneousArrays.jl` is designed for SciML integration, it is not compatible with all solvers. It works seamlessly with explicit methods (e.g., `Tsit5()`, `Vern8()`) that rely on broadcasting. 
However, implicit solvers (e.g., `Rosenbrock23()`) are currently unsupported because they require a homogeneous Jacobian matrix for linear algebra operations. 
We are actively working on extending compatibility to stiff solvers in future releases.
