# API Reference

This page documents the public API of HeterogeneousArrays.

## Types

```@docs
AbstractHeterogeneousVector
HeterogeneousVector
```

## Indexing

```@docs
Base.getindex(::AbstractHeterogeneousVector, ::Int)
Base.setindex!(::AbstractHeterogeneousVector, ::Any, ::Int)
Base.length(::AbstractHeterogeneousVector)
Base.size(::AbstractHeterogeneousVector)
Base.iterate(::AbstractHeterogeneousVector, ::Any)
```

## Property Access

```@docs
Base.getproperty(::AbstractHeterogeneousVector, ::Symbol)
Base.setproperty!(::AbstractHeterogeneousVector, ::Symbol, ::Any)
```

## Allocation & Copying

```@docs
Base.copy(::AbstractHeterogeneousVector)
Base.copyto!
Base.similar(::AbstractHeterogeneousVector)
Base.zero(::HeterogeneousVector)
```

## Broadcasting

```@docs
Base.BroadcastStyle(::Type{HeterogeneousVector})
```