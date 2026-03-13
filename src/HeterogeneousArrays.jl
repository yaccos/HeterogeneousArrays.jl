module HeterogeneousArrays
"""
    HeterogeneousArrays

A Julia package for efficiently storing and operating on heterogeneous data with type-stable broadcasting.

The primary type is [`HeterogeneousVector`](@ref), which allows combining different concrete types 
(including quantities with units) into a single broadcastable vector.

# Features
- Type-stable broadcasting operations
- Support for Unitful quantities with mixed units
- Efficient array-like interface
- Named field access for clarity

# Example
```jldoctest
julia> using HeterogeneousArrays, Unitful

julia> v = HeterogeneousVector(position = 3.0u"m", time = 5.0u"s", count = 42)

julia> v.position
3.0 m

julia> 2.0 .* v .+ v  # Type-stable broadcasting
"""

export AbstractHeterogeneousVector, HeterogeneousVector

using Unitful: Unitful
using RecursiveArrayTools: RecursiveArrayTools
import Base: NamedTuple

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
        # Wrap scalar fields in Ref for mutability
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
        #  Access field and unwrap if it's a Ref
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
        # Set field value, wrapping in Ref if it's a scalar
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

Base.pairs(hv::AbstractHeterogeneousVector{T, S}) where {S, T} = pairs(NamedTuple(hv))

function Base.getindex(hv::AbstractHeterogeneousVector{T, S}, idx::Int) where {T, S}
    current_idx = 1
    for (name, field) in pairs(hv)
        unwrapped_field = _unwrap(field)
        if unwrapped_field isa AbstractArray
            field_length = length(unwrapped_field)
            if current_idx <= idx < current_idx + field_length
                return unwrapped_field[idx - current_idx + 1]
            end
            current_idx += field_length
        else
            if idx == current_idx
                return unwrapped_field
            end
            current_idx += 1
        end
    end
    throw(BoundsError(hv, idx))
end

function Base.setindex!(hv::AbstractHeterogeneousVector{T, S}, val, idx::Int) where {T, S}
    current_idx = 1
    for (name, field) in pairs(hv)
        if field isa AbstractArray
            field_length = length(field)
            if current_idx <= idx < current_idx + field_length
                field[idx - current_idx + 1] = val
                return val
            end
            current_idx += field_length
        else
            if idx == current_idx
                _set_value!(field, val)
                return val
            end
            current_idx += 1
        end
    end
    throw(BoundsError(hv, idx))
end

_field_length(field::Ref) = 1
_field_length(field::AbstractArray) = length(field)
# Update length calculation
Base.length(hv::AbstractHeterogeneousVector) = sum(_field_length, NamedTuple(hv))

Base.size(hv::AbstractHeterogeneousVector) = (length(hv),)
Base.firstindex(hv::AbstractHeterogeneousVector) = 1
Base.lastindex(hv::AbstractHeterogeneousVector) = length(hv)

_copy_field(field::Ref) = Ref(_unwrap(field))
_copy_field(field::AbstractArray) = copy(field)
function Base.copy(hv::HeterogeneousVector)
    copied_x = map(_copy_field, NamedTuple(hv))
    HeterogeneousVector(copied_x)
end

_copy_field!(dst::Ref, src::Ref) = _set_value!(dst, _unwrap(src))
_copy_field!(dst::AbstractArray, src::AbstractArray) = copy!(dst, src)

function Base.copyto!(dst::AbstractHeterogeneousVector, src::AbstractHeterogeneousVector)
    if propertynames(dst) != propertynames(src)
        # throw(ArgumentError("HeterogeneousVectors must have the same field names"))
        error("Cannot copy to $(nameof(typeof(dst))) with different field names: $(propertynames(dst)) vs $(propertynames(src))")
    end
    for name in propertynames(dst)
        src_field = getfield(NamedTuple(src), name)
        dst_field = getfield(NamedTuple(dst), name)
        _copy_field!(dst_field, src_field)
    end
    return dst
end
function Base.copy!(dst::AbstractHeterogeneousVector, src::AbstractHeterogeneousVector)
    Base.copyto!(dst, src)
end

_zero_field(field::Ref) = Ref(zero(_unwrap(field)))
_zero_field(field::AbstractArray) = zero(field)

_similar_field(field::Ref) = Ref(zero(_unwrap(field)))

_similar_field(field::Ref, ::Type{ElType}) where {ElType} = Ref(zero(ElType))

_similar_field(field::AbstractArray) = similar(field)

_similar_field(field::AbstractArray, ::Type{ElType}) where {ElType} = similar(field, ElType)

function Base.similar(hv::HeterogeneousVector{T}) where {T}
    similar_x = map(_zero_field, NamedTuple(hv))
    HeterogeneousVector(similar_x)
end

function Base.zero(hv::HeterogeneousVector)
    zero_x = map(_zero_field, NamedTuple(hv))
    HeterogeneousVector(zero_x)
end

# Broadcasting support for AbstractHeterogeneousVector
function Base.BroadcastStyle(::Type{<:AbstractHeterogeneousVector{T, S}}) where {T, S}
    Broadcast.Style{AbstractHeterogeneousVector{fieldnames(S)}}()
end
function Base.BroadcastStyle(::Broadcast.Style{AbstractHeterogeneousVector{Names1}},
        ::Broadcast.Style{AbstractHeterogeneousVector{Names2}}) where {Names1, Names2}
    error("Cannot broadcast heterogeneous vectors with different field names: $(Names1) vs $(Names2)")
end
function Base.BroadcastStyle(::Broadcast.Style{AbstractHeterogeneousVector{Names}},
        ::Broadcast.Style{AbstractHeterogeneousVector{Names}}) where {Names}
    Broadcast.Style{AbstractHeterogeneousVector{Names}}()
end
function Base.BroadcastStyle(::Broadcast.Style{AbstractHeterogeneousVector{Names}},
        ::Base.Broadcast.BroadcastStyle) where {Names}
    Broadcast.Style{AbstractHeterogeneousVector{Names}}()
end

# Helper function to find HeterogeneousVector in broadcast arguments
function find_heterogeneous_vector(bc::Base.Broadcast.Broadcasted)
    find_heterogeneous_vector(bc.args)
end
function find_heterogeneous_vector(args::Tuple)
    find_heterogeneous_vector(find_heterogeneous_vector(args[1]), Base.tail(args))
end
find_heterogeneous_vector(x::Base.Broadcast.Extruded) = x.x
find_heterogeneous_vector(x) = x
find_heterogeneous_vector(::Tuple{}) = nothing
find_heterogeneous_vector(x::AbstractHeterogeneousVector, rest) = x
find_heterogeneous_vector(::Any, rest) = find_heterogeneous_vector(rest)

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.Style{AbstractHeterogeneousVector{Names}}}) where {Names}
    hv = find_heterogeneous_vector(bc)
    similar(hv)
end
function Base.similar(
        bc::Broadcast.Broadcasted{Broadcast.Style{AbstractHeterogeneousVector{Names}}},
        ::Type{ElType}) where {Names, ElType}
    hv = find_heterogeneous_vector(bc)
    similar_x = map(NamedTuple(hv)) do field
        _similar_field(field, ElType)
    end
    HeterogeneousVector(similar_x)
end

"""
    Base.similar(hv::HeterogeneousVector, ::Type{T}, ::Type{S}, R::DataType...)

Construct a new `HeterogeneousVector` with the same structure and field names as `hv`, 
but with potentially different element types for each individual field.

This variadic method allows for "per-field" type specialization, similar to `ArrayPartition` 
in `RecursiveArrayTools`. It is particularly useful when you need to maintain 
heterogeneity (e.g., keeping one field as an `Int` while converting another to a `Float64`).

# Arguments
- `hv`: The template `HeterogeneousVector`.
- `T`, `S`, `R...`: A sequence of types. The total number of types provided must 
  exactly match the number of fields in `hv`.

# Returns
- A `HeterogeneousVector` where the i-th field has the i-th provided type. 
  Note that memory is uninitialized (via `similar`).

# Errors
- Throws a `DimensionMismatch` if the number of types provided does not match the 
  number of fields in the vector.

# Implementation Note
This function uses `ntuple` with a compile-time length to ensure the resulting 
`NamedTuple` is type-inferred correctly by the Julia compiler.
"""
function Base.similar(hv::HeterogeneousVector, ::Type{T}, ::Type{S}, R::DataType...) where {
        T, S}
    new_types = (T, S, R...)
    names = propertynames(hv)

    if length(new_types) != length(names)
        throw(DimensionMismatch("Number of types ($(length(new_types))) must match number of fields ($(length(names)))"))
    end

    # Use ntuple for type-stable indexing
    new_fields_tuple = ntuple(length(names)) do i
        field = getfield(NamedTuple(hv), names[i])
        _similar_field(field, new_types[i])
    end

    return HeterogeneousVector(NamedTuple{names}(new_fields_tuple))
end

"""
    Base.similar(hv::HeterogeneousVector, ::Type{ElType})

Construct a new `HeterogeneousVector` with the same structure and field names as `hv`, 
but with all fields converted to the same uniform element type `ElType`.

This method satisfies the standard `AbstractArray` interface. It is essential for 
ensuring that broadcasting operations (e.g., `hv .* 1.0`) return a `HeterogeneousVector` 
rather than collapsing into a standard flat `Array`.

# Arguments
- `hv`: The template `HeterogeneousVector`.
- `ElType`: The target type for all segments/fields within the new vector.

# Returns
- A `HeterogeneousVector` where every field's elements are of type `ElType`.

# Implementation Note
Uses `map` over the field names to recursively call `_similar_field` on each segment, 
preserving the original `NamedTuple` keys.
"""
function Base.similar(hv::HeterogeneousVector, ::Type{ElType}) where {ElType}
    names = propertynames(hv)
    new_fields = map(names) do name
        field = getfield(NamedTuple(hv), name)
        _similar_field(field, ElType)
    end
    return HeterogeneousVector(NamedTuple{names}(new_fields))
end

"""
    similar(hv::HeterogeneousVector; kwargs...)

Construct a new `HeterogeneousVector` with the same field names and structure as `hv`, 
optionally overriding the element types of specific fields.

This method allows for high-level, name-based type transformation. If a field name is 
provided as a keyword argument, the new vector will use the specified type for that 
segment. Fields not mentioned in `kwargs` will preserve their original element types.

# Arguments
- `hv::HeterogeneousVector`: The template vector providing the names and structure.
- `kwargs...`: Pairs of `fieldname = Type` used to redefine specific segments.

# Returns
- A `HeterogeneousVector` with uninitialized (or zeroed) data in the requested types.

# Errors
- Throws an `ArgumentError` if any key in `kwargs` does not match an existing field 
  name in `hv`. This prevents silent failures caused by typos in field names.

# Performance Note
This implementation avoids `Dict` allocations by operating directly on the `kwargs` 
NamedTuple, making it more efficient and "compiler-friendly" than dictionary-based lookups.

# Example
```jldoctest
julia> using HeterogeneousArrays, Unitful

julia> v = HeterogeneousVector(pos = [1.0, 2.0]u"m", id = [10, 20]);

julia> # Change 'id' to Float64 and 'pos' to a different unit/type
       v2 = similar(v, id = Float64, pos = Float32);

julia> eltype(v2.id)
Float64

julia> # Typos in field names will now trigger an error
       similar(v, poss = Float64)
ERROR: ArgumentError: Field 'poss' does not exist in HeterogeneousVector. Available fields: (:pos, :id)

"""
function Base.similar(hv::HeterogeneousVector{T, S}; kwargs...) where {T, S}
    names = propertynames(hv)
    kw_names = keys(kwargs)

    # Validate that all provided keywords exist as fields
    for k in kw_names
        if k !== :_ && !(k in names)
            throw(ArgumentError("Field '$k' does not exist in $(nameof(typeof(hv))). Available fields: $names"))
        end
    end

    # Construct the new fields
    new_fields_tuple = ntuple(length(names)) do i
        name = names[i]
        field = getfield(NamedTuple(hv), name)

        # If 'name' exists in kwargs, we get the value (the Type).
        # Otherwise, we use the existing field's type
        target_type = get(kwargs, name, nothing)

        if target_type !== nothing
            return _similar_field(field, target_type)
        else
            return _similar_field(field)
        end
    end

    return HeterogeneousVector(NamedTuple{names}(new_fields_tuple))
end

mutable struct BcInfo{BcStyle <: Broadcast.BroadcastStyle}
    f::Function
    Args::DataType # Structure of arguments, available both on runtime and compile-time
    # The expression needed to evaluated to get to the current node from the root broadcast to unpack
    # We let it be of the Any type for now as it can be both a Symbol or an Expr
    expr::Any
    # Only the two last fields need to be mutable
    current_arg::Int
    res_args::Vector{Any}
    function BcInfo(BcStyle, F, Args, expr)
        # No need for storing Axes, it is basically a placeholder for obtaining the other type parameters
        new{BcStyle}(F.instance, Args, expr, 0, Any[])
    end
    function BcInfo(BT::Type, expr)
        BcStyle, Axes, F, Args = BT.parameters
        new{BcStyle}(F.instance, Args, expr, 0, Any[])
    end
end

# In case you wonder: Why is the unpacking done in such a convoluted way? Isn't it better to use recursion?
# The answer is: Yes, it is far easier to use recursion here, but once the broadcasts get complicated enough,
# Julia gives up on optimizing out the unpacking, leaving the macro expansion to runtime, hampering performance.
@generated function unpack_broadcast(
        bc::Broadcast.Broadcasted{BcStyle, Axes, F, Args}, ::Val{field}
) where {BcStyle, Axes, F, Args, field}
    # Defines some constants
    generate_info(F, Args, arg_path) = BcInfo(BcStyle, F, Args, arg_path)
    bc_stack = Vector{BcInfo{BcStyle}}()
    push!(bc_stack, generate_info(F, Args, :bc))
    res_broadcast = nothing # We must declare this variable here in order to see changes after exiting the loop 
    while !isempty(bc_stack)
        bc_info = pop!(bc_stack)
        expr = bc_info.expr
        args_expr = :(getfield($expr, :args))
        res = bc_info.res_args
        if !(res_broadcast isa Nothing)
            # Adds the result from the child broadcast if it exists
            push!(res, res_broadcast)
        end
        arg_types = bc_info.Args.parameters
        nargs = length(arg_types)
        # A flag on whether we should jump back to the beginning of the loop
        # Needed because Julia does not support while-else, nor break statements for outer loops
        ArgT_is_bc = false
        while bc_info.current_arg < nargs
            bc_info.current_arg += 1
            i = bc_info.current_arg
            current_arg_expr = :(getfield($args_expr, $(i)))
            ArgT = arg_types[i]
            if ArgT <: Broadcast.Broadcasted{BcStyle}
                # A new broadcast is found
                # We first re-add the old broadcast to the stack
                push!(bc_stack, bc_info)
                # then construct the information for the new one
                new_info = BcInfo(ArgT, current_arg_expr)
                # and then push it to the stack
                push!(bc_stack, new_info)
                ArgT_is_bc = true
                break
            end

            if ArgT <: AbstractHeterogeneousVector
                current_arg_expr = :(getproperty($current_arg_expr, $(QuoteNode(field))))
            end
            push!(res, current_arg_expr)
        end
        if ArgT_is_bc
            # In case we have encountered a new broadcast, we start the process over again
            # one level deeper
            res_broadcast = nothing
        else
            # Otherwise, we are done handling the arguments and construct the resulting broadcast
            arg_tuple_expr = :(tuple($(res...)))
            res_broadcast = :(Broadcast.Broadcasted(getfield($expr, :f), $arg_tuple_expr))
        end
    end
    return res_broadcast
end

# Using the low-level functions Broadcast.broadcasted or Broadcast.Broadcasted incur considerable
# overhead due to some oddities in the Julia compiler when the arg tuple is not a bitset
function Base.copy(bc::Broadcast.Broadcasted{Broadcast.Style{AbstractHeterogeneousVector{Names}}}) where {Names}
    function map_fun(::Val{name}) where {name}
        bc_unpacked = unpack_broadcast(bc, Val(name))
        Broadcast.materialize(bc_unpacked)
    end
    res_args = map(map_fun, Val.(Names))
    HeterogeneousVector(NamedTuple{Names}(res_args))
end

# @inline Base.@constprop :aggressive function Base.copyto!(
#         dest::AbstractHeterogeneousVector{T, S},
#         bc::Broadcast.Broadcasted{
#             Broadcast.Style{AbstractHeterogeneousVector{Names}}, Axes, F, Args}
# ) where {T, S, Names, Axes, F, Args <: Tuple}
#     if fieldnames(S) != Names
#         throw(ArgumentError("Cannot copy to heterogeneous vector with different field names: $(fieldnames(S)) vs $(Names)"))
#     end
#     # Using value types to specialize map_fun is indeed an ugly solution
#     # Constant propagation should **usually** make this unnecessary, but
#     # benchmarking has shown there are cases where this does not happen (even with aggressive const propagation),
#     # causing type unstability and costly runtime dispatch
#     function map_fun(::Val{name}) where {name}
#         target_field = getfield(NamedTuple(dest), name)
#         bc_unpacked = unpack_broadcast(bc, Val(name))
#         if target_field isa Ref
#             target_field[] = Broadcast.materialize(bc_unpacked)
#         else
#             Broadcast.materialize!(target_field, bc_unpacked)
#         end
#     end
#     map(map_fun, Val.(Names))
#     dest
# end

@inline Base.@constprop :aggressive function Base.copyto!(
        dest::AbstractHeterogeneousVector{T, S},
        bc::Broadcast.Broadcasted{
            Broadcast.Style{AbstractHeterogeneousVector{Names}}, Axes, F, Args}
) where {T, S, Names, Axes, F, Args <: Tuple}
    if fieldnames(S) != Names
        throw(ArgumentError("Field name mismatch: $(fieldnames(S)) vs $(Names)"))
    end

    # Define map_fun as a named local function with an explicit @inline hint
    @inline function map_field(::Val{name}) where {name}
        target_field = getfield(NamedTuple(dest), name)
        bc_unpacked = unpack_broadcast(bc, Val(name))

        if target_field isa Ref
            # Use @inline here too
            target_field[] = Broadcast.materialize(bc_unpacked)
        else
            # This triggers the recursion for nested structures
            Broadcast.materialize!(target_field, bc_unpacked)
        end
        return nothing
    end

    # map over Val handles the unrolling
    map(map_field, Val.(Names))
    return dest
end

# Compute segment ranges for each field in the NamedTuple
# The results are zero-indexed ranges, i.e. the first field starts at 0
function _compute_segment_ranges(x::NamedTuple)
    # We need zero-based contiguous ranges for each field in order.
    # NOTE: Iterating a NamedTuple iterates its values, which is what we want for lengths.
    n = length(x)
    if n == 0
        return NamedTuple()
    end
    # Collect lengths without allocating intermediate vectors where possible.
    # map over NamedTuple returns a tuple, so we can splat into cumsum input.
    field_lengths = map(_field_length, x)  # tuple of Int
    # Build prefix sums starting with 0 (zero-based indexing for segments).
    # We avoid concatenations like [0; ...] by constructing a tuple directly.
    segment_ends = cumsum((0, field_lengths...))  # length n+1 tuple
    # Create the range for each field i: segment_ends[i] : segment_ends[i+1]-1
    ranges = ntuple(i -> begin
            s = segment_ends[i]
            e = segment_ends[i + 1] - 1
            s:e
        end, n)
    # Extract the compile-time field name tuple from the NamedTuple type for a fully-typed result.
    names = fieldnames(typeof(x))
    return NamedTuple{names}(ranges)
end

# Written specifically to deal with cases such as calculate_residuals!() where the destination is an ordinary Array
@inline Base.@constprop :aggressive function Base.copyto!(
        dest::AbstractArray,
        bc::Broadcast.Broadcasted{Broadcast.Style{AbstractHeterogeneousVector{Names}}}
) where {Names}
    hv = find_heterogeneous_vector(bc)
    dest_idx = firstindex(dest)
    segment_ranges = _compute_segment_ranges(NamedTuple(hv))
    function map_fun(::Val{name}) where {name}
        bc_unpacked = unpack_broadcast(bc, Val(name))
        segment_range = segment_ranges[name]
        dest_segment = view(dest, dest_idx .+ segment_range)
        Broadcast.materialize!(dest_segment, bc_unpacked)
    end
    map(map_fun, Val.(Names))
    return dest
end

# Show methods for AbstractHeterogeneousVector
Base.summary(hv::AbstractHeterogeneousVector) = string(typeof(hv), " with members:")
function Base.show(io::IO, m::MIME"text/plain", hv::AbstractHeterogeneousVector)
    show(io, m, NamedTuple(hv))
end

# Copy-catted from RecursiveArrayTools.jl/src/utils.jl
# From Iterators.jl. Moved here since Iterators.jl is not precompile safe anymore.

# Concatenate the output of n iterators
struct Chain{T <: Tuple}
    xss::T
end

# iteratorsize method defined at bottom because of how @generated functions work in 0.6 now

"""
    chain(xs...)

Iterate through any number of iterators in sequence.

```julia
julia> for i in chain(1:3, ['a', 'b', 'c'])
           @show i
       end
i = 1
i = 2
i = 3
i = 'a'
i = 'b'
i = 'c'
```
"""
chain(xss...) = Chain(xss)

Base.length(it::Chain{Tuple{}}) = 0
Base.length(it::Chain) = sum(length, it.xss)

Base.eltype(::Type{Chain{T}}) where {T} = typejoin([eltype(t) for t in T.parameters]...)

function Base.iterate(it::Chain)
    i = 1
    xs_state = nothing
    while i <= length(it.xss)
        xs_state = iterate(it.xss[i])
        xs_state !== nothing && return xs_state[1], (i, xs_state[2])
        i += 1
    end
    return nothing
end

function Base.iterate(it::Chain, state)
    i, xs_state = state
    xs_state = iterate(it.xss[i], xs_state)
    while xs_state == nothing
        i += 1
        i > length(it.xss) && return nothing
        xs_state = iterate(it.xss[i])
    end
    return xs_state[1], (i, xs_state[2])
end

Base.iterate(x::AbstractHeterogeneousVector) = iterate(Chain(values(NamedTuple(x))))
function Base.iterate(x::AbstractHeterogeneousVector, state)
    iterate(Chain(values(NamedTuple(x))), state)
end
end # module HeterogeneousArrays
