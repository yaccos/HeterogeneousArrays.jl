#=
Design contract:
- Storage is a flat vector of plain numbers, reached through `rawdata(x)`.
- Everything user-facing is unitful. Flat indexing (`x[i]`) attaches the unit of the field
    owning slot `i`, and `x[i] = v` converts `v` into that unit or throws. Property
    access (`x.θ`, `x.pos`) does the same.

The shape parameter `S` is an isbits NamedTuple value of the form
    (θ = (1, u"rad"), pos = (2:4, u"m"), ...)
mapping each field to its slot (Int for scalars, UnitRange for arrays) and its storage unit.
=#

using Unitful: ustrip, unit, dimension, upreferred, NoUnits

"""
    CollectionVector{T, S, D, E} <: AbstractVector{E}

A structured state vector with flat, contiguous, unitless storage of eltype `T`
and compile-time shape `S` (field names, slot ranges, and units). `E` is the
`Union` of the per-field element types `typeof(one(T) * u)`, and is determined
by `T` and `S`.

Elements are unitful: both flat indexing and property access attach the unit of
the field that owns the slot, and both assignment forms convert into it or
throw. The plain numbers live behind [`rawdata`](@ref).

```julia
julia> using HeterogeneousArrays, Unitful

julia> x = CollectionVector(θ = 0.1u"rad", pos = [1.0, 2.0]u"m");

julia> x.θ            # unit re-attached at compile time, zero cost
0.1 rad

julia> x[2]           # flat index, unit of the owning field
1.0 m

julia> rawdata(x)     # the unitless storage
3-element Vector{Float64}:
 0.1
 1.0
 2.0

julia> x.pos[1] = 50.0u"cm"; x.pos[1]   # unit-checked, converting assignment
0.5 m

julia> x.θ = 1.0u"m"  # wrong dimension
ERROR: DimensionError: ...
```

Construct from a `NamedTuple` of unitful scalars/arrays. Array fields with mixed units of
the same dimension are converted to the unit of their first element.
"""
struct CollectionVector{T, S, D <: AbstractVector{T}, E} <: AbstractVector{E}
    data::D

    CollectionVector{T, S, D, E}(data) where {T, S, D, E} = new{T, S, D, E}(data)
end

# Constructor from a NamedTuple
function CollectionVector(nt::NamedTuple)
    # Check for empty fields
    isempty(nt) && throw(ArgumentError("CollectionVector requires at least one field"))
    for (name, v) in pairs(nt)
        v isa AbstractArray && isempty(v) &&
            throw(ArgumentError("Field '$name' is an empty array. Empty fields are not supported"))
    end
    # Initialize the storage vector
    T = promote_type(map(rawnumtype, values(nt))...)
    data = Vector{T}(undef, sum(map(fieldlen, values(nt))))
    # Walk through the fields, record values in the storage vector (`data`) and their
    # slot/unit in a `specs` vector
    offset = 0
    specs = Any[]
    for v in values(nt)
        u = fieldunit(v)
        if v isa AbstractArray
            r = (offset + 1):(offset + length(v))
            data[r] .= ustrip.(Ref(u), v)
            push!(specs, (r, u))
            offset += length(v)
        else
            offset += 1
            data[offset] = ustrip(u, v)
            push!(specs, (offset, u))
        end
    end
    # Build the shape NamedTuple from the `specs` vector
    shape = NamedTuple{keys(nt)}(Tuple(specs))
    # Build the CollectionVector with the computed shape and storage vector
    return CollectionVector{T, shape, Vector{T}}(data)
end

# Convenience constructor from keyword arguments
CollectionVector(; kwargs...) = CollectionVector(NamedTuple(kwargs))

# Constructor that computes E from T and S
@inline function CollectionVector{T, S, D}(data) where {T, S, D}
    CollectionVector{T, S, D, eltypeof(T, S)}(data)
end
@inline eltypeof(::Type{T}, S::NamedTuple) where {T} = eltypeof(T, values(S))
@inline function eltypeof(::Type{T}, specs::Tuple) where {T}
    Union{slotelt(T, first(specs)[2]), eltypeof(T, Base.tail(specs))}
end
@inline eltypeof(::Type{T}, ::Tuple{}) where {T} = Union{}
@inline slotelt(::Type{T}, u) where {T} = typeof(one(T) * u)

# Helpers for the construction
rawnumtype(v::Number) = typeof(ustrip(v))
rawnumtype(v::AbstractArray) = typeof(ustrip(first(v)))
fieldunit(v::Number) = unit(v)
fieldunit(v::AbstractArray) = unit(first(v))
fieldlen(v::Number) = 1
fieldlen(v::AbstractArray) = length(v)













## Tooling to attach/detach units of a whole CollectionVector at once
"""
    attach(S::NamedTuple, data::AbstractVector) -> CollectionVector

Wrap raw storage `data` with shape `S`, re-attaching structure and units with zero copy.
"""
@inline function attach(S::NamedTuple, data::AbstractVector)
    length(data) == shape_length(S) || throw(DimensionMismatch(
        "Data length $(length(data)) does not match shape length $(shape_length(S))"))
    CollectionVector{eltype(data), S, typeof(data)}(data)
end
shape_length(S::NamedTuple) = sum(spec -> spec_len(spec[1]), values(S))
spec_len(r::UnitRange{Int}) = length(r)
spec_len(::Int) = 1


"""
    rawdata(x) -> AbstractVector

Access the flat unitless storage of a `CollectionVector` (identity on other vectors).
"""
rawdata(x::CollectionVector) = getfield(x, :data)
rawdata(x::AbstractVector) = x

Unitful.ustrip(x::CollectionVector) = rawdata(x)

"""
    shapeof(x::CollectionVector) -> NamedTuple

The compile-time shape: a NamedTuple mapping field names to `(slot, unit)`.
"""
shapeof(::CollectionVector{T, S}) where {T, S} = S
















## Tooling to access and modidy whole property/fields of CollectionVector
# e.g. x.pos = [3.0, 4.0]u"cm"
@inline Base.@constprop :aggressive function Base.getproperty(
        x::CollectionVector{T, S}, name::Symbol) where {T, S}
    if haskey(S, name)
        spec = getfield(S, name)
        return fieldview(getfield(x, :data), spec[1], spec[2])
    else
        throw(ArgumentError("CollectionVector has no field '$name'. Available fields: $(keys(S))"))
    end
end
@inline fieldview(data, i::Int, u) = data[i] * u
@inline fieldview(data, r::UnitRange{Int}, u) = UnitfulSlice(view(data, r), u)


@inline Base.@constprop :aggressive function Base.setproperty!(
        x::CollectionVector{T, S}, name::Symbol, val) where {T, S}
    if haskey(S, name)
        spec = getfield(S, name)
        return setfield!(getfield(x, :data), spec[1], spec[2], val)
    else
        throw(ArgumentError("CollectionVector has no field '$name'. Available fields: $(keys(S))"))
    end
end
@inline function setfield!(data, i::Int, u, val)
    data[i] = ustrip(u, val)
    return val
end
@inline function setfield!(data, r::UnitRange{Int}, u, val::AbstractArray)
    data[r] .= ustrip.(Ref(u), val)
    return val
end
@inline function setfield!(data, r::UnitRange{Int}, u, val)
    throw(ArgumentError("Cannot assign a scalar to array field spanning $(r); assign elementwise or with an array"))
end

# For REPL tab completion of fields
Base.propertynames(::CollectionVector{T, S}) where {T, S} = keys(S)

"""
    NamedTuple(x::CollectionVector)

Materialize the unitful contents as a NamedTuple (scalar fields become
`Quantity`s, array fields become `Vector{Quantity}`s).

This makes a copy of the values, it is not a view into the CollectionVector storage.
"""
function Base.NamedTuple(x::CollectionVector{T, S}) where {T, S}
    map(spec -> materialize_field(getfield(x, :data), spec[1], spec[2]), S)
end
materialize_field(data, i::Int, u) = data[i] * u
materialize_field(data, r::UnitRange{Int}, u) = data[r] .* u











## Tooling to access and modify a segment of raw storage in place
# e.g. x.pos[1] = 50u"cm"
# The goal is to avoid creating/allocating a subarray, and instead use a view directly on
# the raw storage vector, but with units.
"""
    UnitfulSlice(data::AbstractVector, u::Unitful.FreeUnits)

A lazy, unit-attaching view over a segment of raw storage.

Reading `slice[i]` returns `data[i] * u` (a `Unitful.Quantity`). Writing `slice[i] = val`
converts `val` to unit `u` and stores the stripped value, throwing a
`Unitful.DimensionError` if the dimensions do not match.
"""
struct UnitfulSlice{Q, U, D <: AbstractVector} <: AbstractVector{Q}
    data::D

    function UnitfulSlice(data::AbstractVector{T}, u::Unitful.FreeUnits) where {T <: Number}
        Q = typeof(one(T) * u)
        new{Q, u, typeof(data)}(data)
    end
end

Base.size(s::UnitfulSlice) = size(getfield(s, :data))

# TODO: play with @boundscheck/@inbounds?
@inline function Base.getindex(s::UnitfulSlice{Q, U}, i::Int) where {Q, U}
    return getfield(s, :data)[i] * U
end

# TODO: play with @boundscheck/@inbounds?
@inline function Base.setindex!(s::UnitfulSlice{Q, U}, val, i::Int) where {Q, U}
    getfield(s, :data)[i] = ustrip(U, val)
    return val
end








## Tooling to index directly on CollectionVector
# e.g. x[2] = 50u"cm"
# Note that if the index `i` is a runtime value, the compiler cannot know which field owns
# the slot (and as such the unit) and so will the return type will be a Union of all the
# possible units of the CollectionVector. If there are not too many possible units (maximum 3
# on Julia 1.13), the compiler will do some union splitting and produce specialized code for
# each possibilities. But if there are more units than that, it will widen the return type to
# a general `Quantity{T}`, which can then cause runtime dispatch if used in other functions.
# This is not a problem when indexing on fields.
# TODO: sprinkle some @inbounds/@boundscheck?
Base.size(x::CollectionVector) = size(getfield(x, :data))

@inline function Base.getindex(x::CollectionVector{T, S}, i::Int) where {T, S}
    data = getfield(x, :data)
    slotget(data, i, values(S))
end

@inline function Base.setindex!(x::CollectionVector{T, S}, v, i::Int) where {T, S}
    data = getfield(x, :data)
    slotset!(data, v, i, values(S))
end

@inline function slotget(data, i, specs::Tuple)
    spec = first(specs)
    inslot(i, spec[1]) && return data[i] * spec[2]
    return slotget(data, i, Base.tail(specs))
end
@inline slotget(data, i, ::Tuple{}) = throw(BoundsError(data, i))

@inline function slotset!(data, v, i, specs::Tuple)
    spec = first(specs)
    if inslot(i, spec[1])
        data[i] = ustrip(spec[2], v)
        return v
    end
    return slotset!(data, v, i, Base.tail(specs))
end
@inline slotset!(data, v, i, ::Tuple{}) = throw(BoundsError(data, i))

@inline inslot(i::Int, s::Int) = i == s
@inline inslot(i::Int, r::UnitRange{Int}) = i in r










## Display

function Base.summary(io::IO, x::CollectionVector{T, S}) where {T, S}
    print(io, length(getfield(x, :data)), "-element CollectionVector{", T,
        "} with fields ", keys(S))
end

function Base.show(io::IO, ::MIME"text/plain", x::CollectionVector{T, S}) where {T, S}
    summary(io, x)
    print(io, ":")
    ctx = IOContext(io, :limit => true, :compact => true)
    for name in keys(S)
        spec = getfield(S, name)
        println(io)
        print(io, "  ", name, " = ")
        _show_field(ctx, getfield(x, :data), spec[1], spec[2])
    end
end

_show_field(io, data, i::Int, u) = show(io, data[i] * u)
function _show_field(io, data, r::UnitRange{Int}, u)
    # Print raw values with the unit once, avoiding the parametric eltype noise
    # of Vector{Quantity{...}} display.
    show(io, data[r])
    u === NoUnits || print(io, " ", u)
end
