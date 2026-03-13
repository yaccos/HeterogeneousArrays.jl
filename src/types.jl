abstract type AbstractHeterogeneousVector{T, S} <: AbstractVector{T} end

# Copy-catted from DiffEqBase DiffEqBaseUnitfulExt.jl
Value(x::Number) = x
Value(x::Type{T}) where {T <: Number} = T
Value(x::Type{Unitful.AbstractQuantity{T, D, U}}) where {T, D, U} = T
Value(x::Unitful.AbstractQuantity) = x.val

# If we do not specify this as a subtype of AbstractVector, the Broadcast machinery will try to convert it into
# a broadcastable representation, but we make this type to be broadcastable as-is due to our customizations of
# copy and copyto!

# Helper function to wrap scalars in Ref for mutability
_make_mutable(x::AbstractArray) = x
_make_mutable(x::Ref) = x
_make_mutable(x) = Ref(x)
@generated _unwrap(x::Ref) = :(x[])
@generated _unwrap(x) = :x
_set_value!(x::Ref, val) = (x[] = val)
_set_value!(x::AbstractArray, val::AbstractArray) = copy!(x, val)
_set_value!(x::AbstractArray, val, idx) = (x[idx] = val)

"""
    HeterogeneousVector{T, S} <: AbstractVector{T}

A segmented vector that stores mixed types while maintaining type-stable broadcasting.

Fields can contain scalars (wrapped for mutability), arrays, or quantities with units.
The flattened view presents all elements as a single `AbstractVector` for broadcasting.

# Constructors

- `HeterogeneousVector(; kwargs...)` - Create with named fields
- `HeterogeneousVector(args...)` - Create with positional args (auto-named field_1, field_2, ...)
- `HeterogeneousVector(x::NamedTuple)` - Create from a NamedTuple

# Arguments
- `x::NamedTuple`: Internal storage (users shouldn't access directly)

# Examples

Named fields:
```jldoctest
julia> using HeterogeneousArrays

julia> v = HeterogeneousVector(x = 1.0, y = 2.5, z = [1, 2, 3])
"""
struct HeterogeneousVector{T, S <: NamedTuple} <: AbstractHeterogeneousVector{T, S}
    x::S

    """
        HeterogeneousVector(x::NamedTuple)

    Construct a HeterogeneousVector from an existing NamedTuple.

    Scalar fields are automatically wrapped in `Ref` for mutability.

    !!! note "Memory Behavior"
    - **Scalars** are wrapped in a new `Ref`, effectively copying them into a mutable container.
    - **Arrays** are stored by **reference**. Modifying the original array used to 
      construct the vector will affect the `HeterogeneousVector` and vice versa. 
      Use `copy(your_array)` during construction if independent storage is required.

    # Arguments
    - `x::NamedTuple`: The data to store

    # Examples
    ```jldoctest
    julia> using HeterogeneousArrays

    julia> nt = (x = 1.0, y = 2.0, z = [1, 2, 3])

    julia> v = HeterogeneousVector(nt)
    ```

    # See Also
    - `HeterogeneousVector(; kwargs...)`
    - `HeterogeneousVector(args...)`
    """
    function HeterogeneousVector(x::NamedTuple)
        mutable_x = map(_make_mutable, x)
        arg_types = map(
            field -> RecursiveArrayTools.recursive_bottom_eltype(_unwrap(field)),
            values(mutable_x)
        )
        T = promote_type(arg_types...)
        new{T, typeof(mutable_x)}(mutable_x)
    end

    """
        HeterogeneousVector(; kwargs...)

    Construct a HeterogeneousVector with named fields.

    # Arguments
    - Arbitrary keyword arguments become named fields

    # Examples
    ```jldoctest
    julia> using HeterogeneousArrays

    julia> v = HeterogeneousVector(x = 1.0, y = 2.0, data = [1, 2, 3])
    ```

    # See Also
    - `HeterogeneousVector(x::NamedTuple)`
    - `HeterogeneousVector(args...)`
    """
    function HeterogeneousVector(; kwargs...)
        x = NamedTuple(kwargs)
        HeterogeneousVector(x)
    end

    """
        HeterogeneousVector(args...)

    Construct a HeterogeneousVector with positional arguments.

    Fields are automatically named `field_1`, `field_2`, etc.

    # Arguments
    - `args...`: Values to store (scalars or arrays)

    # Examples
    ```jldoctest
    julia> using HeterogeneousArrays

    julia> v = HeterogeneousVector(1.0, 2.0, [1, 2, 3])

    julia> v.field_1
    1.0
    ```

    # See Also
    - `HeterogeneousVector(x::NamedTuple)`
    - `HeterogeneousVector(; kwargs...)`
    """
    function HeterogeneousVector(args...)
        names = ntuple(i -> Symbol("field_$i"), length(args))
        x = NamedTuple{names}(args)
        HeterogeneousVector(x)
    end
end

@generated NamedTuple(hv::AbstractHeterogeneousVector{
    T, S}) where {T, S} = :(getfield(hv, :x))

# Custom property access for clean external interface
# Note: For accessing the named tuple field x, we must use getfield or invoking NamedTuple

"""
    Base.getproperty(hv::HeterogeneousVector, name::Symbol)

Access a named field in the HeterogeneousVector.

# Arguments
- `hv::HeterogeneousVector`: The vector
- `name::Symbol`: Field name

# Returns
The value of the field (unwrapped if it's a scalar)

# Examples
```jldoctest
julia> using HeterogeneousArrays

julia> v = HeterogeneousVector(x = 1.0, y = 2.0);

julia> v.x
1.0
```

# See Also
- `Base.setproperty!(::HeterogeneousVector, ::Symbol, ::Any)`
- `propertynames`
"""
@inline Base.@constprop :aggressive function Base.getproperty(hv::AbstractHeterogeneousVector, name::Symbol)
    if name in propertynames(hv)
        field = getfield(NamedTuple(hv), name)
        return _unwrap(field)
    else
        msg = string(string(nameof(typeof(hv))), " has no field ", name,
            ". Available fields: ", join(collect(string.(propertynames(hv))), ", "))
        error(msg)
    end
end

"""
    Base.setproperty!(hv::HeterogeneousVector, name::Symbol, value)

Set a named field in the HeterogeneousVector.

# Arguments
- `hv::HeterogeneousVector`: The vector
- `name::Symbol`: Field name  
- `value`: New value for the field

# Examples
```jldoctest
julia> using HeterogeneousArrays

julia> v = HeterogeneousVector(x = 1.0, y = 2.0);

julia> v.x = 5.0;

julia> v.x
5.0
```

# See Also
- `Base.getproperty(::HeterogeneousVector, ::Symbol)`
- `propertynames`
"""
@inline Base.@constprop :aggressive function Base.setproperty!(
        hv::AbstractHeterogeneousVector, name::Symbol, value)
    if name in propertynames(hv)
        field = getfield(NamedTuple(hv), name)
        _set_value!(field, value)
    else
        msg = string(string(nameof(typeof(hv))), " has no field ", name,
            ". Available fields: ", join(collect(string.(propertynames(hv))), ", "))
        error(msg)
    end
end

@generated Base.propertynames(::AbstractHeterogeneousVector{
    T, S}) where {T, S} = :(fieldnames(S))
