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
        # Check that every field elementtype is made only of `T` values, and that `data` has
        # as many slots as the shape `S` needs.
        checklayouts(T, S)
        length(data) == shape_length(T, S) || throw(DimensionMismatch(
            "Data length $(length(data)) does not match shape length $(shape_length(T, S))"))
        return new{T, S, D, E}(data)
    end
end







## Element type compatibility API
# What the container needs to know about an element type `Q`
#   is_storable(Q)         -> Bool          opt-in gate; false by default, true for supported types
#   rawtype(Q)            -> Type          plain isbits number type stored for a `Q` (storage vector eltype)
#   field_type(q)         -> isbits value  what identifies the field's elements once the raw
#                                          number is removed (a unit, `nothing`, ...); goes into `S`
#   elementtype(T, ft)    -> Type          element type seen through the container. Elements are
#                                          read and written by reinterpreting their slots in the
#                                          storage vector, so every real-number component of this
#                                          type must be a `T` (see `haslayout`)
# A value `q` is stored as `convert(elementtype(T, ft), q)`: this must succeed for every value
# that belongs in the field and throw for any other.

is_storable(::Type) = false
function field_type end
function elementtype end

# Plain real numbers: no unit. Field type set to `nothing`.
is_storable(::Type{<:Real}) = true
rawtype(::Type{Q}) where {Q <: Real} = Q
field_type(::Real) = nothing
elementtype(::Type{T}, ::Nothing) where {T} = T

# Unitful quantities: the field type is the unit.
is_storable(::Type{<:Unitful.AbstractQuantity{<:Real}}) = true
rawtype(::Type{Q}) where {Q <: Unitful.AbstractQuantity} = Unitful.numtype(Q)
field_type(q::Unitful.AbstractQuantity) = unit(q)
elementtype(::Type{T}, u::U) where {T, U <: Unitful.Units} = Unitful.Quantity{T, Unitful.dimension(u), U}

# Complex numbers: two real slots per element, so the storage stays real (solvers and
# ForwardDiff only ever see reals; the RHS sees `Complex{T}` through the reinterpret view).
# TODO: support complex *quantities* (`[1+2im]u"m"`) (we require Unitful.AbstractQuantity{<:Real} above)
is_storable(::Type{<:Complex{<:Real}}) = true
rawtype(::Type{Complex{T}}) where {T} = T # storage eltype is the real part type
struct ComplexParts end
field_type(::Complex) = ComplexParts()    # custom field type
elementtype(::Type{T}, ::ComplexParts) where {T} = Complex{T}

# Constructor helpers built on the API
firstelem(v::AbstractArray) = first(v)
firstelem(v) = v
fieldlen(v::AbstractArray) = length(v)
fieldlen(v) = 1

# The gate: is the element type of this field supported?
# TODO: right now we check for a manual is_storable implementation. Could check for
# `hasmethod` on the needed functions instead.
function checkelement(name, v)
    unsupported(Q) = ArgumentError(
        "Field '$name' has element type $Q, which CollectionVector does not support. " *
        "Extend `HeterogeneousArrays.is_storable`, `rawtype`, `field_type` and `elementtype` " *
        "for it, and make sure `convert` into its element type works.")
    if v isa AbstractArray && !isconcretetype(eltype(v))
        # Abstract eltype (e.g. `Vector{Any}`): check all elements
        for x in v
            is_storable(typeof(x)) || throw(unsupported(typeof(x)))
        end
    else
        # Concrete eltype: cchecl only the first
        Q = typeof(firstelem(v))
        is_storable(Q) || throw(unsupported(Q))
    end
    return nothing
end

# Storage type contributed by a field.
# Promoted over all elements if not concrete.
function fieldrawtype(v::AbstractArray)
    Q = eltype(v)
    isconcretetype(Q) && return rawtype(Q)
    return mapreduce(x -> rawtype(typeof(x)), promote_type, v)
end
fieldrawtype(v) = rawtype(typeof(v))

# A reinterpret view over an element's slots is valid when every real-number component of the
# element type is `T`. The element then occupies `slotcount(T, ft)` slots with no padding.
checklayouts(::Type{T}, S::NamedTuple) where {T} = checklayouts(T, keys(S), values(S))
function checklayouts(::Type{T}, names::Tuple, specs::Tuple) where {T}
    checklayout(first(names), T, first(specs)[2])
    return checklayouts(T, Base.tail(names), Base.tail(specs))
end
# If it is empty
checklayouts(::Type{T}, ::Tuple{}, ::Tuple{}) where {T} = nothing

function checklayout(name, ::Type{T}, ft) where {T}
    Q = elementtype(T, ft)
    haslayout(T, Q) || throw(ArgumentError(
        "Field '$name': elements of type $Q do not have the memory layout of consecutive $T " *
        "slots. Their components are $(componenttypes(Q)), and all must be $T"))
    return nothing
end

haslayout(::Type{T}, ::Type{Q}) where {T, Q} = all(C -> C === T, componenttypes(Q))

# Components are found by descending into struct fields, stopping at `Real` types
# and at types without fields.
componenttypes(::Type{Q}) where {Q <: Real} = (Q,)
function componenttypes(::Type{Q}) where {Q}
    isconcretetype(Q) && fieldcount(Q) > 0 || return (Q,)
    return concat_componenttypes(fieldtypes(Q))
end
concat_componenttypes(ts::Tuple) = (componenttypes(first(ts))..., concat_componenttypes(Base.tail(ts))...)
concat_componenttypes(::Tuple{}) = ()

# Store `q` as element `j` of the field view `fv` (`nothing` for a scalar field), rethrowing
# any conversion error with the field name and position.
function storeelement!(fv, name, j, ft, q)
    try
        fv[something(j, 1)] = q
    catch e
        where = j === nothing ? "value" : "element $j"
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
    # into it or rejected with an error. An element occupies `slotcount(T, ft)` slots.
    fts = map(v -> field_type(firstelem(v)), values(nt))
    foreach((name, ft) -> checklayout(name, T, ft), keys(nt), fts)
    data = Vector{T}(undef, sum(map((v, ft) -> fieldlen(v) * slotcount(T, ft), values(nt), fts)))
    # Walk through the fields, record values in the storage vector (`data`) and their
    # slot/field type in a `specs` vector
    offset = 0
    specs = Any[]
    for (name, v, ft) in zip(keys(nt), values(nt), fts)
        k = slotcount(T, ft)
        if v isa AbstractArray
            r = (offset + 1):(offset + k * length(v))
            fv = fieldview(data, r, ft)
            for (j, q) in enumerate(v)
                storeelement!(fv, name, j, ft, q)
            end
            push!(specs, (r, ft))
            offset += k * length(v)
        else
            storeelement!(fieldview(data, (offset + 1):(offset + k), ft), name, nothing, ft, v)
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
    promote_type(elementtype(T, flat_ft(T, first(specs)[2])), eltypeof(T, Base.tail(specs)))
end
@inline eltypeof(::Type{T}, ::Tuple{}) where {T} = Union{}
# Flat indexing works slot by slot: inside a multi-slot element a single slot has no field
# type of its own, so it is read as a raw number.
@inline flat_ft(::Type{T}, ft) where {T} = slotcount(T, ft) == 1 ? ft : nothing
# Number of storage slots one element of field type `ft` occupies
@inline slotcount(::Type{T}, ft) where {T} = sizeof(elementtype(T, ft)) ÷ sizeof(T)













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
    CollectionVector{eltype(data), S, typeof(data)}(data)
end
shape_length(::Type{T}, S::NamedTuple) where {T} = sum(spec -> spec_len(T, spec[1], spec[2]), values(S))
spec_len(::Type, r::UnitRange{Int}, ft) = length(r)
spec_len(::Type{T}, ::Int, ft) where {T} = slotcount(T, ft)

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
# Scalar field: a value, the single element of the view over the field's slots.
@inline fieldview(data, i::Int, ft) = fieldview(data, elementslots(data, i, ft), ft)[1]
# Slots of the element that starts at slot `i`
@inline elementslots(data, i::Int, ft) = i:(i + slotcount(eltype(data), ft) - 1)
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
# Assignment writes through the view that `fieldview` returns, whose `setindex!` converts the
# value into the element type or throws.
@inline function setfield!(data, i::Int, ft, val)
    fieldview(data, elementslots(data, i, ft), ft)[1] = val
    return val
end
@inline function setfield!(data, r::UnitRange{Int}, ft, val::AbstractArray)
    fv = fieldview(data, r, ft)
    length(val) == length(fv) || throw(DimensionMismatch(
        "Cannot assign $(length(val)) elements to an array field of $(length(fv)) elements"))
    fv .= val
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
    inslot(eltype(data), i, spec[1], spec[2]) && return fieldview(data, i, flat_ft(eltype(data), spec[2]))
    return slotget(data, i, Base.tail(specs))
end
@inline slotget(data, i, ::Tuple{}) = throw(BoundsError(data, i))

@inline function slotset!(data, v, i, specs::Tuple)
    spec = first(specs)
    if inslot(eltype(data), i, spec[1], spec[2])
        return setfield!(data, i, flat_ft(eltype(data), spec[2]), v)
    end
    return slotset!(data, v, i, Base.tail(specs))
end
@inline slotset!(data, v, i, ::Tuple{}) = throw(BoundsError(data, i))

@inline inslot(::Type{T}, i::Int, s::Int, ft) where {T} = s <= i < s + slotcount(T, ft)
@inline inslot(::Type, i::Int, r::UnitRange{Int}, ft) = i in r










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
