# Performance Guide

`HeterogeneousVector` is designed for type-stable operations. Not all usage patterns are equally fast.

## Type Stability

Julia's compiler optimizes when types are known at compile time. When types are unknown, 
the compiler must generate conservative (slow) code for multiple possibilities.

**In HeterogeneousArrays:**
- **Named field access** (`v.position`) → type-stable (compiler knows the exact type)
- **Broadcasting** (`v .+ w`) → type-stable (each field computed separately)
- **Flattened indexing** (`v[i]`) → *not* type-stable (compiler doesn't know which field contains index `i`)

The same holds for `CollectionVector`.

## Three Patterns

### Type-Stable: Named Field Access

The compiler knows exactly what type is stored in each named field.

```julia
v = HeterogeneousVector(x = [1.0u"m", 2.0u"m"], y = 5.0u"s")

# Same physical quantity, different units (m + cm): type-stable and unit-safe
sum_len = v.x[1] + 50.0u"cm"

# Different components with different units: output unit is inferred at compile time (m*s)
cross_term = v.x[1] * v.y
```

### Type-Stable: Broadcasting

```julia
v = HeterogeneousVector(a = [1.0u"m", 2.0u"m"], b = 3.0u"s")
w = HeterogeneousVector(a = [10.0u"m", 20.0u"m"], b = 5.0u"s")
result = v .+ w  # Fast: computed field-wise as result.a = v.a .+ w.a and result.b = v.b .+ w.b
result2 = v .* w  # Fast: computed field-wise as result2.a = v.a .* w.a and result2.b = v.b .* w.b
```

### Not Type-Stable: Flattened Indexing

Using `v[i]` forces a runtime check to determine which field contains index `i`.
Because the accessed field is only known at runtime, this can introduce type-instabilities and limit optimization.

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
The code used for this benchmark is available in the [`benchamrk/` directory](https://github.com/yaccos/HeterogeneousArrays.jl/tree/main/benchmark/runbenchmark.jl) of the repository.

All benchmark simulation were single-threaded on a CPU:
```Julia
julia> versioninfo()
Julia Version 1.12.6
Commit 15346901f0 (2026-04-09 19:20 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: Windows (x86_64-w64-mingw32)
  CPU: 14 × Intel(R) Core(TM) Ultra 7 165U
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, alderlake)
  GC: Built with stock GC
Threads: 1 default, 1 interactive, 1 GC (on 14 virtual cores)
```


| Array structure       | Unit handling | ODE interface   | Min (ms) | StdErr (ms) | Allocs | Memory      |
|:----------------------|:--------------|:----------------|---------:|------------:|-------:|------------:|
| ComponentVector       | None          | Allocating      | 16.6861  | 1.0606      | 313448 | 12836.1 KiB |
| ComponentVector       | None          | Non-allocating  | 1.0706   | 0.2697      | 40739  | 1853.8 KiB  |
| ComponentVector       | Unitful       | Non-allocating  | 21.0436  | 0.8741      | 447033 | 8521.4 KiB  |
| ArrayPartition        | None          | Allocating      | 1.5601   | 0.5872      | 200734 | 7906.7 KiB  |
| ArrayPartition        | None          | Non-allocating  | 0.7286   | 0.0211      | 27220  | 1109.7 KiB  |
| ArrayPartition        | Unitful       | Allocating      | 1.9348   | 0.6848      | 232888 | 9261.4 KiB  |
| ArrayPartition        | Unitful       | Non-allocating  | 22.5231  | 1.5366      | 268399 | 13754.3 KiB |
| HeterogeneousVector   | None          | Allocating      | 1.5560   | 0.8799      | 197354 | 7838.0 KiB  |
| HeterogeneousVector   | None          | Non-allocating  | 1.1747   | 0.0317      | 28425  | 1198.6 KiB  |
| HeterogeneousVector   | Unitful       | Allocating      | 2.0311   | 1.2939      | 239882 | 9461.4 KiB  |
| HeterogeneousVector   | Unitful       | Non-allocating  | 0.7000   | 0.0282      | 28453  | 1199.6 KiB  |



### Analysis
- **`HeterogeneousVector`** consistently outperforms `ArrayPartition` in the 'Units'-enabled benchmark for the non-allocating interface and is on par for the allocating interface.
- **`HeterogeneousVector`** achieves performance parity with `ArrayPartition` for the 'No Units' case while providing descriptive field names (`.r`, `.v`).
- **Zero-Cost Units:** Thanks to specialized broadcasting kernels, using `Unitful` units results in near-zero performance overhead compared to raw numbers.
- **Memory Efficiency:** The in-place mapping (via a specialized solver interface) ensures that we don't allocate unnecessary temporary arrays during the integration process. However, the time performance benefit to this approach is modest for `HeterogeneousVector` and `ArrayPartition`.
- **NB!** `HeterogeneousVector` provides substantial runtime performance benefits, however compilation times are longer. Hence, `ComponentVector` may in practice be faster for one-off simulations which only take a few seconds to run.
