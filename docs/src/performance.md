# Performance Guide

`HeterogeneousVector` is designed for type-stable operations. Not all usage patterns are equally fast.

## Type Stability

Julia's compiler optimizes when types are known at compile time. When types are unknown, 
the compiler must generate conservative (slow) code for multiple possibilities.

**In HeterogeneousArrays:**
- **Named field access** (`v.position`) → type-stable (compiler knows the exact type)
- **Broadcasting** (`v .+ w`) → type-stable (each field computed separately)
- **Flattened indexing** (`v[i]`) → *not* type-stable (compiler doesn't know which field contains index `i`)

## Three Patterns

### Type-Stable: Named Field Access

The compiler knows exactly what type is stored in each named field.

```julia
v = HeterogeneousVector(x = [1.0, 2.0], y = 5.0)
result = v.x[1] + v.y  # Fast: compiler knows x is Vector{Float64}, y is Float64
```

### Type-Stable: Broadcasting

```julia
v = HeterogeneousVector(a = [1.0, 2.0], b = 3.0)
w = HeterogeneousVector(a = [10.0, 20.0], b = 5.0)
result = v .+ w  # Fast: Computed as (v.a .+ w.a) and (v.b .+ w.b)
```

### Not Type-Stable: Flattened Indexing

Using `v[i]` forces the compiler to check which field contains index i at runtime. This results in "type-shielding" or "boxing," which prevents optimization.

Benchmark Comparison:

```julia
using HeterogeneousArrays, BenchmarkTools

v = HeterogeneousVector(a = rand(1000), b = rand(1000), c = 42.0)

# Each 'v[i]' call is a dynamic lookup.
@btime begin
    total = 0.0
    for i in 1:length($v)
        total += $v[i] 
    end
end

@btime sum($v.a) + sum($v.b) + $v.c
```

## Best Practices

| Goal | Recommended Pattern | Type-Stable? |
|------|-----|:----------:|
| **Loops** | `v.field[i]` | ✓ |
| **Algorithms** | `v.field[i]` | ✓ |
| **Broadcasting** | `v .+ w` or `v .* 2.0` | ✓ |
| **Solvers (ODE, etc.)** | Named fields | ✓ |
| Quick checks | `v[i]` | ✗ (Acceptable)|
| REPL exploration | `v[i]` | ✗ (Acceptable) |

**Rule of thumb:** Use `v.field` for performance-critical inner loops (like ODE steps). Use `v[i]` for REPL exploration, debugging, or non-bottleneck tasks like printing and display.

## Case Study: Orbital Mechanics ODE

To demonstrate the "SciML Bridge" in action, we compare `HeterogeneousVector` against 
the standard Julia tools for structured ODE states: `ArrayPartition` and `ComponentVector`.

### The Benchmark
We solve a standard 2-body Kepler problem (Orbital Mechanics) using the `Vern8()` solver. 
This requires thousands of internal broadcast operations and unit conversions.

All benchmark simulation were single-threaded on a CPU:
```Julia
julia> versioninfo()
Julia Version 1.12.5
Commit 5fe89b8ddc1 (2026-02-09 16:05 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 20 × 13th Gen Intel(R) Core(TM) i7-13700H
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, alderlake)
  GC: Built with stock GC
Threads: 1 default, 1 interactive, 1 GC (on 20 virtual cores)
```


| Implementation Strategy   |            Min (ms)   | StdErr (ms)   |   Allocs   |      Memory |
|:------------------------|:--------------|:------------|:-------|:-------|
|1. HeterogeneousVector (No Units)    |   0.0297    |     0.0015      |   803    |   40.9 KiB |
|2. HeterogeneousVector (Units)        |  0.0295     |    0.0028    |     878   |    43.2 KiB |
|3. ArrayPartition (No Units)           | 0.0347     |    0.0007    |     851   |    57.3 KiB |
|4. ArrayPartition (Units)             |  1.0095     |    0.0328   |    11321   |   676.1 KiB |
|5. ComponentVector (No Units)         |  0.0393    |     0.0003   |     1396   |    68.8 KiB |
|6. ComponentVector (Units)            |  0.8537    |     0.0186   |    17745  |    359.9 KiB |

### Analysis
- **HeterogeneousVector** achieves performance parity with `ArrayPartition` while providing descriptive field names (`.r`, `.v`).
- **Zero-Cost Units:** Thanks to specialized broadcasting kernels, using `Unitful` units results in near-zero performance overhead compared to raw numbers.
- **Memory Efficiency:** The in-place mapping (via a specialized solver interface) ensures that we don't allocate unnecessary temporary arrays during the integration process.
