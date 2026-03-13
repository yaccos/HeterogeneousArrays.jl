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
        throw(DimensionMismatch("Number of types must match number of fields"))
    end
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
    for k in kw_names
        if k !== :_ && !(k in names)
            throw(ArgumentError("Field '$k' does not exist in $(nameof(typeof(hv))). Available fields: $names"))
        end
    end
    new_fields_tuple = ntuple(length(names)) do i
        name = names[i]
        field = getfield(NamedTuple(hv), name)
        target_type = get(kwargs, name, nothing)
        target_type !== nothing ? _similar_field(field, target_type) : _similar_field(field)
    end
    return HeterogeneousVector(NamedTuple{names}(new_fields_tuple))
end
