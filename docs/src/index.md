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

## CollectionVector

`CollectionVector` stores all fields in one flat vector of plain numbers. Field names, slots and units are part of the type: reading a field attaches its unit, and writing a field converts into it or throws a `DimensionError`.

Use `CollectionVector` when the whole state must also be available as one plain vector: `rawdata(x)` is an ordinary `Vector{Float64}` (or `Vector{Float32}`, a vector of `ForwardDiff.Dual`s, ...) that can be passed to code that knows nothing about fields or units, and `attach` turns such a vector back into a `CollectionVector` without copying. Complex fields are stored as pairs of real numbers, so that code only sees real numbers.

Use `HeterogeneousVector` when the fields must keep their own types: there each field is a separate object of any type (an `Int` field stays an `Int`, and `BigFloat` works), while a `CollectionVector` converts all fields to a single `isbits` number type.

```julia
using HeterogeneousArrays, Unitful
using HeterogeneousArrays: rawdata

x = CollectionVector(θ = 0.1u"rad", pos = [1.0, 2.0]u"m")

x.θ                   # 0.1 rad
x.pos[1] = 50.0u"cm"  # converted to 0.5 m
rawdata(x)            # [0.1, 0.5, 2.0], a Vector{Float64}
x.θ = 1.0u"m"         # DimensionError

# Type-stable broadcasting, field by field
2.0 .* x .+ x
```

Fields can hold real numbers, `Unitful` quantities or complex numbers. The storage can be any `isbits` number type, and `attach(shapeof(x), v)` wraps an existing vector `v` without copying it, for example a vector of `ForwardDiff.Dual`s. The broadcasting rules are described in [`HeterogeneousArrays.CollectionVectorStyle`](@ref).

Other element types can be stored by extending four functions. Since elements are read by reinterpreting their slots, every real-number component of the element type must be of the storage type:

```julia
struct Vec3
    x::Float64
    y::Float64
    z::Float64
end
struct Vec3Field end

HeterogeneousArrays.is_storable(::Type{Vec3}) = true                    # opt in
HeterogeneousArrays.rawtype(::Type{Vec3}) = Float64                     # storage type
HeterogeneousArrays.field_type(::Vec3) = Vec3Field()                    # stored in the shape, like a unit
HeterogeneousArrays.elementtype(::Type{Float64}, ::Vec3Field) = Vec3    # element type of the field

v = CollectionVector(r = [Vec3(1, 2, 3), Vec3(4, 5, 6)], m = 2.0)
rawdata(v)  # [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0]
```
