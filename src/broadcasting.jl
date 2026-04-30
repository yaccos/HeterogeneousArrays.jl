"""
    Base.BroadcastStyle(::Type{<:AbstractHeterogeneousVector}) -> BroadcastStyle

Define the broadcast style for HeterogeneousVector to enable type-stable broadcasting.

The HeterogeneousVector uses a custom broadcast style to ensure that broadcasting operations
preserve the heterogeneous structure and field names. When multiple HeterogeneousVectors are 
involved in a broadcast operation, they must have compatible field names.

# Broadcast Rules

1. **Single HeterogeneousVector with other types**: The broadcast result preserves the 
   HeterogeneousVector structure and field names.

2. **Multiple HeterogeneousVectors with matching field names**: All vectors must have identical 
   field names; operations proceed field-by-field in parallel.

3. **Multiple HeterogeneousVectors with different field names**: Throws an error to prevent 
   silent data corruption.

# Examples
```jldoctest
julia> using HeterogeneousArrays

julia> v = HeterogeneousVector(a = [1, 2], b = 3.0);

julia> w = HeterogeneousVector(a = [10, 20], b = 5.0);

julia> result = v .+ w;  # Element-wise addition preserves structure

julia> result.a
2-element Vector{Int64}:
 11
 22
```
"""
abstract type AbstractHeterogeneousVectorStyle{Names} <: Broadcast.AbstractArrayStyle{1} end

struct PureHeterogeneousVectorStyle{Names} <: AbstractHeterogeneousVectorStyle{Names} end

struct MixedHeterogeneousVectorStyle{Names} <: AbstractHeterogeneousVectorStyle{Names} end

function Base.BroadcastStyle(::Type{<:AbstractHeterogeneousVector{T, S}}) where {T, S}
    PureHeterogeneousVectorStyle{fieldnames(S)}()
end

# Broadcasting over HeterogeneousVectors with different fields yields an error
function Base.BroadcastStyle(::PureHeterogeneousVectorStyle{Names1},
        ::PureHeterogeneousVectorStyle{Names2}) where {Names1, Names2}
    error("Cannot broadcast heterogeneous vectors with different field names: $(Names1) vs $(Names2)")
end

# This specialization ensures that  
function Base.BroadcastStyle(::PureHeterogeneousVectorStyle{Names},
        ::PureHeterogeneousVectorStyle{Names}) where {Names}
    PureHeterogeneousVectorStyle{Names}()
end

function Base.BroadcastStyle(style::AbstractHeterogeneousVectorStyle,
        ::Base.Broadcast.BroadcastStyle)
    style
end

function Base.BroadcastStyle(::PureHeterogeneousVectorStyle{Names}, ::Base.Broadcast.AbstractArrayStyle{1}) where {Names}
    MixedHeterogeneousVectorStyle{Names}()
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

function Base.similar(bc::Broadcast.Broadcasted{AbstractHeterogeneousVectorStyle{Names}}) where {Names}
    hv = find_heterogeneous_vector(bc)
    similar(hv)
end

function Base.similar(
        bc::Broadcast.Broadcasted{AbstractHeterogeneousVectorStyle{Names}},
        ::Type{ElType}) where {Names, ElType}
    hv = find_heterogeneous_vector(bc)
    similar_x = map(NamedTuple(hv)) do field
        _similar_field(field, ElType)
    end
    HeterogeneousVector(similar_x)
end

mutable struct BcInfo{BcStyle <: Broadcast.BroadcastStyle}
    f::Function
    Args::DataType
    expr::Any
    current_arg::Int
    res_args::Vector{Any}
    function BcInfo(BcStyle, F, Args, expr)
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
        ::Broadcast.Broadcasted{BcStyle, Axes, F, Args}, ::Val{field}
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

"""
    Base.copy(bc::Broadcast.Broadcasted{Broadcast.Style{AbstractHeterogeneousVector}}) -> HeterogeneousVector

Materialize a broadcast operation into a new HeterogeneousVector.

When a broadcast expression involves a HeterogeneousVector, this method is called to 
allocate and fill the result. The operation is performed field-by-field, allowing 
type-stable operations on each segment independently.

# Arguments
- `bc`: A broadcasted expression tree rooted at a HeterogeneousVector

# Returns
A new HeterogeneousVector containing the broadcasted results, with all fields computed

# Examples
```jldoctest
julia> using HeterogeneousArrays

julia> v = HeterogeneousVector(a = [1, 2], b = 3.0);

julia> result = copy(Broadcast.broadcasted(+, v, 2));

julia> result.a
2-element Vector{Int64}:
 3
 4

julia> result.b
5.0
```
"""
function Base.copy(bc::Broadcast.Broadcasted{PureHeterogeneousVectorStyle{Names}}) where {Names}
    function map_fun(::Val{name}) where {name}
        bc_unpacked = unpack_broadcast(bc, Val(name))
        Broadcast.materialize(bc_unpacked)
    end
    res_args = map(map_fun, Val.(Names))
    HeterogeneousVector(NamedTuple{Names}(res_args))
end

"""
    Base.copyto!(dest::AbstractHeterogeneousVector, bc::Broadcast.Broadcasted) -> AbstractHeterogeneousVector

Materialize a broadcast operation in-place into a HeterogeneousVector.

The broadcast result is computed field-by-field and stored directly into the destination 
vector's existing storage. For array fields, this uses `Broadcast.materialize!()` for 
in-place operations. For scalar fields, the result is assigned to the wrapped value.

# Arguments
- `dest`: The destination HeterogeneousVector (will be modified)
- `bc`: A broadcasted expression tree

# Returns
The modified `dest` vector

# Errors
- Throws `ArgumentError` if the broadcast expression involves a HeterogeneousVector with 
  different field names than `dest`

# Examples
```jldoctest
julia> using HeterogeneousArrays

julia> v = HeterogeneousVector(a = [1.0, 2.0], b = 3.0);

julia> w = HeterogeneousVector(a = [10.0, 20.0], b = 5.0);

julia> copyto!(v, Broadcast.broadcasted(+, v, w));

julia> v.a
2-element Vector{Float64}:
 11.0
 22.0

julia> v.b
8.0
```
"""
@inline Base.@constprop :aggressive function Base.copyto!(
        dest::AbstractHeterogeneousVector{T, S},
        bc::Broadcast.Broadcasted{
            PureHeterogeneousVectorStyle{Names}, Axes, F, Args}
) where {T, S, Names, Axes, F, Args <: Tuple}
    if fieldnames(S) != Names
        throw(ArgumentError("Field name mismatch: $(fieldnames(S)) vs $(Names)"))
    end
    @inline function map_field(::Val{name}) where {name}
        target_field = getfield(NamedTuple(dest), name)
        bc_unpacked = unpack_broadcast(bc, Val(name))
        if target_field isa Ref
            target_field[] = Broadcast.materialize(bc_unpacked)
        else
            Broadcast.materialize!(target_field, bc_unpacked)
        end
        return nothing
    end
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

"""
    Base.copyto!(dest::AbstractArray, bc::Broadcast.Broadcasted{Broadcast.Style{AbstractHeterogeneousVector{Names}}})

Materialize a HeterogeneousVector broadcast result into a flat AbstractArray.

This is the "bridge" between structured heterogeneous data and standard numerical 
solvers. It allows computing residuals or norms from mixed-unit data and 
storing them in a plain, contiguous float array.

# Storage Layout
The result is flattened field-by-field according to the order in `Names`. 
For a vector with fields `pos` (length 2) and `time` (length 1):
- `dest[1:2]` contains results from `pos`
- `dest[3]` contains results from `time`

# Examples
```jldoctest
julia> using HeterogeneousArrays, Unitful

julia> v = HeterogeneousVector(pos = [1.0u"m", 2.0u"m"], time = 10.0u"s");

julia> v_proj = HeterogeneousVector(pos = [1.5u"m", 3.0u"m"], time = 5.0u"s");

julia> # The ODE solver provides a plain Vector{Float64}

julia> residuals = Vector{Float64}(undef, length(v));

julia> # This triggers the flattening copyto!

julia> residuals .= ustrip.(v .- v_proj);

julia> residuals
3-element Vector{Float64}:
 -0.5
 -1.0
  5.0
```
"""
@inline Base.@constprop :aggressive function Base.copyto!(
        dest::AbstractArray,
        bc::Broadcast.Broadcasted{PureHeterogeneousVectorStyle{Names}}
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

@generated function unpack_broadcast(
        ::Broadcast.Broadcasted{BcStyle, Axes, F, Args}, ::Val{field}, segment_range
) where {BcStyle <: MixedHeterogeneousVectorStyle, Axes, F, Args, field}
    # TODO
end

@inline Base.@constprop :aggressive function Base.copy(
    bc::Broadcast.Broadcasted{MixedHeterogeneousVectorStyle{Names}}
) where {Names}
    hv = find_heterogeneous_vector(bc)
    segment_ranges = _compute_segment_ranges(NamedTuple(hv))
    function map_fun(::Val{name}) where {name}
        segment_range = segment_ranges[name]
        bc_unpacked = unpack_broadcast(bc, Val(name), segment_range)
        # dest_segment = view(dest, dest_idx .+ segment_range)
        Broadcast.materialize(bc_unpacked)
    end
    res_args = map(map_fun, Val.(Names))
    HeterogeneousVector(NamedTuple{Names}(res_args))
    return dest
end

# Show methods for AbstractHeterogeneousVector
Base.summary(hv::AbstractHeterogeneousVector) = string(typeof(hv), " with members:")
function Base.show(io::IO, m::MIME"text/plain", hv::AbstractHeterogeneousVector)
    show(io, m, NamedTuple(hv))
end
