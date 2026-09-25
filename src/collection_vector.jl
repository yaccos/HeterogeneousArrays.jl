using Unitful: ustrip, unit

"""
    CollectionVector{T, S, D, E} <: AbstractVector{E}

A structured state vector with flat, contiguous, unitless storage of eltype `T`
and compile-time shape `S` (field names, slot ranges, and field types). `E` is the
promoted type (`promote_type`) of the per-field element types `elementtype(T, field_type)`,
and is determined by `T` and `S`.

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

    function CollectionVector{T, S, D, E}(data) where {T, S, D, E}
        # Fields are accessed through a `reinterpret` view over the storage vector, which
        # supports isbits number type (e.g. Float64, Float32, Int, ForwardDiff.Dual, etc
        # but not something like BigFloat).
        isbitstype(T) || throw(ArgumentError(
            "CollectionVector storage type must be an isbits number type, got $T"))
        return new{T, S, D, E}(data)
    end
end







## Element type compatibility API
# What the container needs to know about an element type `Q`
#   isstorable(Q)         -> Bool          opt-in gate; false by default, true for supported types
#   rawtype(Q)            -> Type          plain isbits number type stored for a `Q` (storage vector eltype)
#   field_type(q)         -> isbits value  what identifies the field's elements once the raw
#                                          number is removed (a unit, `nothing`, ...); goes into `S`
#   strip_type(ft, q)     -> raw number    convert `q` to the field type `ft` and strip, or throw
#                                          if conversion fails
#   attach_type(ft, x)    -> element       inverse of `strip_type`: attach the field type `ft`
#                                          to a raw number `x` to get an element
#   elementtype(T, ft)    -> Type          element type seen through the container; must have the
#                                          memory layout of `slotcount(ft)` consecutive `T`s
#   slotcount(ft)         -> Int           raw numbers per element (default 1)
# Elements that occupy several slots implement `strip_slots!`/`attach_slots` instead of the
# scalar `strip_type`/`attach_type` (see `Complex` below).

isstorable(::Type) = false
rawtype(::Type{Q}) where {Q <: Number} = typeof(one(Q))   # `one` is unitless in every unit package
function field_type end
function strip_type end
function attach_type end
function elementtype end
slotcount(ft) = 1

# Slot-level versions used by the container: default to the scalar API (one slot per element).
strip_slots!(slots, ft, q) = (slots[1] = strip_type(ft, q); slots)
attach_slots(ft, slots) = attach_type(ft, slots[1])

# Plain real numbers: no unit. Field type set to `nothing`.
isstorable(::Type{<:Real}) = true
field_type(::Real) = nothing
strip_type(::Nothing, x::Real) = x
attach_type(::Nothing, x) = x
elementtype(::Type{T}, ::Nothing) where {T} = T

# Unitful quantities: the field type is the unit.
isstorable(::Type{<:Unitful.AbstractQuantity{<:Real}}) = true
field_type(q::Unitful.AbstractQuantity) = unit(q)
strip_type(u::Unitful.Units, q::Unitful.AbstractQuantity) = ustrip(u, q)
strip_type(u::Unitful.Units, x::Real) = ustrip(u, x) # use DimensionError message of Unitful
attach_type(u::Unitful.Units, x) = x * u
elementtype(::Type{T}, u::Unitful.Units) where {T} = typeof(one(T) * u)

# Complex numbers: two real slots per element, so the storage stays real (solvers and
# ForwardDiff only ever see reals; the RHS sees `Complex{T}` through the reinterpret view).
# TODO: support complex *quantities* (`[1+2im]u"m"`) (we require Unitful.AbstractQuantity{<:Real} above)
isstorable(::Type{<:Complex{<:Real}}) = true
rawtype(::Type{Complex{T}}) where {T} = T # storage eltype is the real part type
struct ComplexParts end
field_type(::Complex) = ComplexParts()    # custom field type
slotcount(::ComplexParts) = 2
elementtype(::Type{T}, ::ComplexParts) where {T} = Complex{T}
strip_slots!(slots, ::ComplexParts, z::Number) = (slots[1] = real(z); slots[2] = imag(z); slots)
attach_slots(::ComplexParts, slots) = Complex(slots[1], slots[2])

# Constructor helpers built on the API
firstelem(v::AbstractArray) = first(v)
firstelem(v) = v
fieldlen(v::AbstractArray) = length(v)
fieldlen(v) = 1

# The gate: is the element type of this field supported?
# TODO: right now we check for a manual isstorable implementation. Could check for
# `hasmethod` on the needed functions instead.
function checkelement(name, v)
    Q = typeof(firstelem(v))
    isstorable(Q) || throw(ArgumentError(
        "Field '$name' has element type $Q, which CollectionVector does not support. " *
        "Implement isstorable, rawtype, field_type, strip_type, attach_type and elementtype for it."))
    return nothing
end

# Storage type contributed by a field: promoted over all its elements
fieldrawtype(v::AbstractArray) = promote_type(map(x -> rawtype(typeof(x)), v)...)
fieldrawtype(v) = rawtype(typeof(v))

# The reinterpret view over an array field is only legitimate if an element is laid out
# exactly like `slotcount(ft)` consecutive raw numbers.
function checklayout(name, ::Type{T}, ft) where {T}
    Q = elementtype(T, ft)
    sizeof(Q) == slotcount(ft) * sizeof(T) || throw(ArgumentError(
        "Field '$name': elements of type $Q do not have the memory layout of $(slotcount(ft)) $T slot(s)"))
    return nothing
end

# Convert one element into the field's slots, wrapping any error with field context.
function stripelement!(slots, name, i, ft, q)
    try
        strip_slots!(slots, ft, q)
    catch e
        where = i === nothing ? "value" : "element $i"
        throw(ArgumentError("Field '$name': $where is $q, which cannot be expressed in the " *
            "field's type $ft (taken from the first element). Cause: $(sprint(showerror, e))"))
    end
    return nothing
end









## Constructor from a NamedTuple
function CollectionVector(nt::NamedTuple)
    # Check for empty fields and for unsupported element types, before anything else
    isempty(nt) && throw(ArgumentError("CollectionVector requires at least one field"))
    for (name, v) in pairs(nt)
        v isa AbstractArray && isempty(v) &&
            throw(ArgumentError("Field '$name' is an empty array. Empty fields are not supported"))
        checkelement(name, v)
    end
    # Storage type: promoted over all fields, must be isbits (checked again in the inner
    # constructor, but here the error comes before any conversion is attempted)
    T = promote_type(map(fieldrawtype, values(nt))...)
    isbitstype(T) || throw(ArgumentError(
        "CollectionVector storage type must be an isbits number type, got $T"))
    # The field type of a field is that of its first element; every other element is converted
    # into it or rejected with a clear message. An element occupies `slotcount(ft)` slots.
    fts = map(v -> field_type(firstelem(v)), values(nt))
    foreach((name, ft) -> checklayout(name, T, ft), keys(nt), fts)
    data = Vector{T}(undef, sum(map((v, ft) -> fieldlen(v) * slotcount(ft), values(nt), fts)))
    # Walk through the fields, record values in the storage vector (`data`) and their
    # slot/field type in a `specs` vector
    offset = 0
    specs = Any[]
    for (name, v, ft) in zip(keys(nt), values(nt), fts)
        k = slotcount(ft)
        if v isa AbstractArray
            r = (offset + 1):(offset + k * length(v))
            for (j, q) in enumerate(v)
                stripelement!(view(data, (offset + (j - 1) * k + 1):(offset + j * k)), name, j, ft, q)
            end
            push!(specs, (r, ft))
            offset += k * length(v)
        else
            stripelement!(view(data, (offset + 1):(offset + k)), name, nothing, ft, v)
            push!(specs, (offset + 1, ft))
            offset += k
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
    promote_type(elementtype(T, flat_ft(first(specs)[2])), eltypeof(T, Base.tail(specs)))
end
@inline eltypeof(::Type{T}, ::Tuple{}) where {T} = Union{}
# Flat indexing works slot by slot: inside a multi-slot element a single slot has no field
# type to attach, so it is seen as a raw number.
@inline flat_ft(ft) = slotcount(ft) == 1 ? ft : nothing













## Tooling to attach/detach units of a whole CollectionVector at once
"""
    rawdata(x) -> AbstractVector

Access the raw (unitless) storage of a `CollectionVector` (identity on other vectors).
"""
rawdata(x::CollectionVector) = getfield(x, :data)
rawdata(x::AbstractVector) = x

Unitful.ustrip(x::CollectionVector) = rawdata(x)

"""
    attach(S::NamedTuple, data::AbstractVector) -> CollectionVector

Wrap raw storage `data` with a shape parameter `S` to create a `CollectionVector`.
The shape parameter of an existing CollectionVector can be accessed with the
[`shapeof`](@ref) function.
"""
@inline function attach(S::NamedTuple, data::AbstractVector)
    length(data) == shape_length(S) || throw(DimensionMismatch(
        "Data length $(length(data)) does not match shape length $(shape_length(S))"))
    CollectionVector{eltype(data), S, typeof(data)}(data)
end
shape_length(S::NamedTuple) = sum(spec -> spec_len(spec[1], spec[2]), values(S))
spec_len(r::UnitRange{Int}, ft) = length(r)
spec_len(::Int, ft) = slotcount(ft)

"""
    shapeof(x::CollectionVector) -> NamedTuple

The compile-time shape of a `CollectionVector`: a NamedTuple mapping field names to
`(slot, field type)`.
"""
shapeof(::CollectionVector{T, S}) where {T, S} = S
















## Tooling to access and modidy property/fields of a CollectionVector
# e.g. x.pos = [3.0, 4.0]u"cm"
# e.g. x.pos[1] = 1.0u"cm"
@inline Base.@constprop :aggressive function Base.getproperty(
        x::CollectionVector{T, S}, name::Symbol) where {T, S}
    if haskey(S, name)
        spec = getfield(S, name)
        return fieldview(getfield(x, :data), spec[1], spec[2])
    else
        throw(ArgumentError("CollectionVector has no field '$name'. Available fields: $(keys(S))"))
    end
end
# Scalar field: a value.
# Note that `slotcount(ft)` is a compile-time constant for a given shape, so the branch
# folds away.
@inline function fieldview(data, i::Int, ft)
    k = slotcount(ft)
    k == 1 ? attach_type(ft, data[i]) : attach_slots(ft, view(data, i:(i + k - 1)))
end
# Array field: a zero-copy view. A plain-number field is just a view. Anything else is the
# same bytes reinterpreted as `Q` (which also handles multi-slot elements such as Complex).
@inline function fieldview(data, r::UnitRange{Int}, ft)
    Q = elementtype(eltype(data), ft)
    v = view(data, r)
    Q === eltype(data) ? v : reinterpret(Q, v)
end


@inline Base.@constprop :aggressive function Base.setproperty!(
        x::CollectionVector{T, S}, name::Symbol, val) where {T, S}
    if haskey(S, name)
        spec = getfield(S, name)
        return setfield!(getfield(x, :data), spec[1], spec[2], val)
    else
        throw(ArgumentError("CollectionVector has no field '$name'. Available fields: $(keys(S))"))
    end
end
@inline function setfield!(data, i::Int, ft, val)
    k = slotcount(ft)
    k == 1 ? (data[i] = strip_type(ft, val)) : strip_slots!(view(data, i:(i + k - 1)), ft, val)
    return val
end
@inline function setfield!(data, r::UnitRange{Int}, ft, val::AbstractArray)
    k = slotcount(ft)
    length(val) * k == length(r) || throw(DimensionMismatch(
        "Cannot assign $(length(val)) elements to an array field of $(length(r) ÷ k) elements"))
    if k == 1
        data[r] .= strip_type.(Ref(ft), val)
    else
        for (j, q) in enumerate(val)
            strip_slots!(view(data, (first(r) + (j - 1) * k):(first(r) + j * k - 1)), ft, q)
        end
    end
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
materialize_field(data, i::Int, ft) = fieldview(data, i, ft)
materialize_field(data, r::UnitRange{Int}, ft) = collect(fieldview(data, r, ft))











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
    inslot(i, spec[1], spec[2]) && return attach_type(flat_ft(spec[2]), data[i])
    return slotget(data, i, Base.tail(specs))
end
@inline slotget(data, i, ::Tuple{}) = throw(BoundsError(data, i))

@inline function slotset!(data, v, i, specs::Tuple)
    spec = first(specs)
    if inslot(i, spec[1], spec[2])
        data[i] = strip_type(flat_ft(spec[2]), v)
        return v
    end
    return slotset!(data, v, i, Base.tail(specs))
end
@inline slotset!(data, v, i, ::Tuple{}) = throw(BoundsError(data, i))

@inline inslot(i::Int, s::Int, ft) = s <= i < s + slotcount(ft)
@inline inslot(i::Int, r::UnitRange{Int}, ft) = i in r










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
        show_field(ctx, getfield(x, :data), spec[1], spec[2])
    end
end

show_field(io, data, i::Int, ft) = show(io, materialize_field(data, i, ft))
function show_field(io, data, r::UnitRange{Int}, ft::Unitful.Units)
    # Print raw values with the unit once, avoiding the parametric eltype noise
    # of Vector{Quantity{...}} display.
    show(io, data[r])
    print(io, " ", ft)
end
show_field(io, data, r::UnitRange{Int}, ft) = show(io, materialize_field(data, r, ft))
