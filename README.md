# HeterogeneousArrays.jl
This repository introduces a type called HeterogeneousVector, a segmented array designed to hold elements of different concrete types while maintaining efficient, type-stable broadcasting.

The goal of this project is to generalize the prototype from [COMBAT.jl](https://github.com/yaccos/COMBAT.jl) into a more versatile and reusable package.

## Key Features

- **Type-stable heterogeneous state**: Hold fields of different types in a single efficient container
- **Unitful integration**: Enforce physical correctness at compile-time
- **Zero-overhead broadcasting**: Fused broadcast operations maintain type stability
- **ODE-friendly**: Drop-in compatible with SciML solvers (DifferentialEquations.jl)
- **Physical safety**: Catch dimensional inconsistencies before they propagate

## Installation

The package can be installed with the Julia package manager. From the Julia REPL type `]` and run:
```
pkg> add HeterogeneousArrays
```

Then use it in your code:
```julia
using HeterogeneousArrays
```

## Example: Pendulum Physics

HeterogeneousVector is ideal for physical systems where state variables have different units. It ensures that your numerical solver respects the underlying physics of your model.
### 1. Modeling a Pendulum

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
sol = DE.solve(prob, DE.Vern8())
```

### 2. Absolute vs. Relative Errors

In SciML solvers (like `DifferentialEquations.jl`), you can provide tolerances. For heterogeneous data, your Absolute Error (`abstol`) must have the same units as your state fields.

```julia
# Absolute tolerance must match the dimensions of u0
abstol_struct = 1e-8 .* oneunit.(u0)
sol = DE.solve(prob, DE.Vern8(), abstol = abstol_struct)

```

## Physical Safety with DimensionError

One of the primary benefits of using HeterogeneousArrays with Unitful is catching physical bugs early. The library enforces strict dimensional consistency during broadcasting and assignment.

Broadcasting Safety: If you try to add a scalar to a field with units (e.g., `u0.ω .+ 1.0`), the system throws a DimensionError.

Assignment Safety: If the ODE solver (or a user) tries to assign a "Meter" value to the "Radian" field, the operation will fail immediately rather than producing a physically impossible result.

## Performance

HeterogeneousVector is built on a specialized in-place mapping interface. This allows it to communicate with SciML solvers with zero memory overhead, achieving performance parity with plain `Vector{Float64}` while maintaining full type and unit safety.

For detailed benchmarks against ComponentArrays and ArrayPartition, see the [Performance Guide](https://yaccos.github.io/HeterogeneousArrays.jl/performance/).

## Documentation

Check out the [Full Documentation](https://yaccos.github.io/HeterogeneousArrays.jl/) for:

* API Reference
* Advanced Broadcasting Logic
* Performance Benchmarks

<!---
## Citation
If you use HeterogeneousArrays.jl in research, please cite:
@article{,
  author = {Jacob Pettersen},
  doi = {},
  url = {https://doi.org/...},
  title = {HeterogeneousArrays.jl ... },
  year = {2026},
  publisher = {},
  volume = {},
  number = {},
  pages = {},
  journal = {Journal of Open Source Software}
}
-->
