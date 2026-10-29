"""
    CollectionVectorStyle{Names} <: Broadcast.AbstractArrayStyle{1}

Broadcast style of a `CollectionVector` with field names `Names`.

A broadcast involving a `CollectionVector` is evaluated field by field: every
`CollectionVector` in the expression is replaced by one of its fields (the view returned by
`getproperty`) and the function is broadcast over that field. Scalars, `Ref`s and
0-dimensional arrays are passed unchanged. An ordinary 1D array is split into the slot
segments of the fields, so it needs one element per storage slot, and fields whose
elements occupy several slots each (e.g. complex fields) are rejected.

`CollectionVector`s with the same field names combine even if their units, storage types or
field lengths differ; mismatches are reported when the broadcast is evaluated. Different
field names, a tuple, an array with two or more dimensions, or a `HeterogeneousVector` in
the same expression throw an `ArgumentError`.
"""
struct CollectionVectorStyle{Names} <: Broadcast.AbstractArrayStyle{1} end

function Base.BroadcastStyle(::Type{<:CollectionVector{T, S}}) where {T, S}
    CollectionVectorStyle{keys(S)}()
end

function Base.BroadcastStyle(::CollectionVectorStyle{Names1},
        ::CollectionVectorStyle{Names2}) where {Names1, Names2}
    throw(ArgumentError("Cannot broadcast CollectionVectors with different field names: " *
                        "$(Names1) vs $(Names2)"))
end

function Base.BroadcastStyle(style::CollectionVectorStyle{Names},
        ::CollectionVectorStyle{Names}) where {Names}
    style
end

# Pairings with scalars, 0D and 1D arrays and other styles keep the CollectionVector style.
# The `Unknown`, `AbstractArrayStyle` and `DefaultArrayStyle` methods avoid method
# ambiguities with the generic rules in Base.Broadcast.
Base.BroadcastStyle(style::CollectionVectorStyle, ::Broadcast.BroadcastStyle) = style
Base.BroadcastStyle(style::CollectionVectorStyle, ::Broadcast.Unknown) = style
Base.BroadcastStyle(style::CollectionVectorStyle, ::Broadcast.AbstractArrayStyle{0}) = style
Base.BroadcastStyle(style::CollectionVectorStyle, ::Broadcast.AbstractArrayStyle{1}) = style
Base.BroadcastStyle(style::CollectionVectorStyle, ::Broadcast.DefaultArrayStyle{0}) = style
Base.BroadcastStyle(style::CollectionVectorStyle, ::Broadcast.DefaultArrayStyle{1}) = style

# Arrays of other dimensions are not supported.
function Base.BroadcastStyle(::CollectionVectorStyle,
        ::Broadcast.AbstractArrayStyle{N}) where {N}
    throw(ArgumentError("Cannot broadcast a CollectionVector with a $N-dimensional array; " *
                        "only scalars and 1D arrays are supported"))
end
function Base.BroadcastStyle(::CollectionVectorStyle,
        ::Broadcast.DefaultArrayStyle{N}) where {N}
    throw(ArgumentError("Cannot broadcast a CollectionVector with a $N-dimensional array; " *
                        "only scalars and 1D arrays are supported"))
end

function Base.BroadcastStyle(::CollectionVectorStyle, ::Broadcast.Style{Tuple})
    throw(ArgumentError("Cannot broadcast a CollectionVector with a Tuple; collect it into a Vector first"))
end

# A CollectionVector and a HeterogeneousVector do not combine, in either order.
function Base.BroadcastStyle(::CollectionVectorStyle{Names1},
        ::HeterogeneousVectorStyle{Names2}) where {Names1, Names2}
    throw(mixedcontainers(Names1, Names2))
end
function Base.BroadcastStyle(::HeterogeneousVectorStyle{Names2},
        ::CollectionVectorStyle{Names1}) where {Names1, Names2}
    throw(mixedcontainers(Names1, Names2))
end
function mixedcontainers(cvnames, hvnames)
    ArgumentError("Cannot broadcast a CollectionVector (fields $cvnames) together with a " *
                  "HeterogeneousVector (fields $hvnames)")
end

# Zero-based segment of each field in the storage of a CollectionVector of storage type `T`
# and shape `S`, which partitions ordinary 1D arrays in a broadcast: an `Int` for a scalar
# field, a `UnitRange` for an array field, and `nothing` for a field whose elements occupy
# several slots each.
function segments(::Type{T}, S::NamedTuple) where {T}
    NamedTuple{keys(S)}(segmentlist(T, values(S)))
end
segmentlist(::Type, ::Tuple{}) = ()
function segmentlist(::Type{T}, specs::Tuple) where {T}
    slot, ft = first(specs)
    segment = slotcount(T, ft) == 1 ? slot .- 1 : nothing
    return (segment, segmentlist(T, Base.tail(specs))...)
end

# The field names `names` as a tuple of `Val`s. The loops over the fields in this file recurse
# over such a tuple: inference widens a recursion that carries the names in a `Val{names}`
# from the third field on, but recursion over a tuple of `Val`s keeps things stable
# TODO: even if they stay type-stable, tuple recursions with more than 32 fields start to
# allocate and slow down. Potential solutions can be @generated functions or
# the UnrolledUtilities.jl package
valnames(::Val{names}) where {names} = ntuple(i -> Val(names[i]), Val(length(names)))

# First element of the per-field broadcast `bcf`. A scalar field (`Int` slot) gives a
# 0-dimensional broadcast, an array field (`UnitRange` slot) a 1D one.
function firstresult(bcf, ::Val{name}, ::Int) where {name}
    checkresult(name, Broadcast.materialize(bcf))
end
function firstresult(bcf, ::Val{name}, ::UnitRange{Int}) where {name}
    bci = Broadcast.instantiate(bcf)
    return checkresult(name, bci[first(eachindex(bci))])
end
function checkresult(name, q)
    Q = typeof(q)
    is_storable(Q) || throw(ArgumentError(
        "Field '$name': the broadcast produces elements of type $Q, which CollectionVector " *
        "does not support"))
    return q
end

# Storage type and shape of the result of broadcasting `bc`, whose CollectionVector inputs
# have storage type `T` and shape `S`. Each field's type and storage type come from the first
# element of its per-field broadcast. Field lengths come from `S`.
@inline function resultshape(::Type{T}, S::NamedTuple, bc) where {T}
    segs = segments(T, S)
    qs = firstfieldresults(bc, segs, S, valnames(Val(keys(S))))
    Tnew = storagetype(qs)
    return Tnew, NamedTuple{keys(S)}(resultspecs(Tnew, T, values(S), resultfieldtypes(qs), 0))
end

firstfieldresults(bc, segs, S, ::Tuple{}) = ()
@inline function firstfieldresults(bc, segs, S, names::Tuple)
    q = firstfieldresult(bc, segs, S, first(names))
    return (q, firstfieldresults(bc, segs, S, Base.tail(names))...)
end
@inline function firstfieldresult(bc, segs, S, ::Val{name}) where {name}
    bcf = unpack_broadcast(bc, Val(name), getfield(segs, name))
    return firstresult(bcf, Val(name), getfield(S, name)[1])
end

# Promotion of the storage types of the values `qs`
storagetype(::Tuple{}) = Union{}
storagetype(qs::Tuple) = promote_type(rawtype(typeof(first(qs))), storagetype(Base.tail(qs)))

# Field types of the values `qs`. They are singletons (a unit, `nothing`, ...), so every
# argument of `resultspecs` is a compile-time constant and inference knows the result shape.
# With `qs` itself as argument, inference loses the shape from about ten fields on.
resultfieldtypes(::Tuple{}) = ()
resultfieldtypes(qs::Tuple) = (field_type(first(qs)), resultfieldtypes(Base.tail(qs))...)

# Slot and field type of each field of the result, whose field types are `fts`
resultspecs(::Type, ::Type, ::Tuple{}, ::Tuple{}, offset) = ()
function resultspecs(
        ::Type{Tnew}, ::Type{T}, specs::Tuple, fts::Tuple, offset) where {Tnew, T}
    slot, ft = first(specs)
    ftnew = first(fts)
    n = slotcount(Tnew, ftnew) * fieldlength(T, slot, ft)
    spec = (resultslot(slot, offset, n), ftnew)
    return (spec, resultspecs(Tnew, T, Base.tail(specs), Base.tail(fts), offset + n)...)
end
fieldlength(::Type, ::Int, ft) = 1
fieldlength(::Type{T}, r::UnitRange{Int}, ft) where {T} = length(r) ÷ slotcount(T, ft)
resultslot(::Int, offset, n) = offset + 1
resultslot(::UnitRange{Int}, offset, n) = (offset + 1):(offset + n)

# Replace every array argument of the broadcast tree that may share memory with `dest` by a
# copy, except `dest` itself. Fields are written one after another, so an argument that
# aliases `dest` could otherwise be read after another field has already overwritten it.
function unaliasargs(dest, bc::Broadcast.Broadcasted{Style}) where {Style}
    Broadcast.Broadcasted{Style}(bc.f, map(arg -> unaliasargs(dest, arg), bc.args), bc.axes)
end
unaliasargs(dest, x) = dest === x ? x : Base.unalias(dest, x)

Base.unaliascopy(x::CollectionVector) = attach(shapeof(x), Base.unaliascopy(rawdata(x)))

# Segments for partitioning ordinary arrays come from a CollectionVector in the broadcast
# tree, or from the destination when the tree holds none (e.g. `x .= 0.0u"m"`).
segmentsource(cv::CollectionVector, dest) = cv
segmentsource(::Any, dest) = dest

"""
    Broadcast.materialize!(::CollectionVectorStyle, dest, bc::Broadcasted)

Evaluate `dest .= bc` when the combined style of `dest` and `bc` is `CollectionVectorStyle`.

Base passes `copyto!` the style of the right-hand side alone. This method passes
`CollectionVectorStyle` instead, so that the assignment is evaluated field by field even when
only `dest` is a `CollectionVector`, as in `x .= 0.0u"m"` or `x .= v` with an ordinary
vector `v`.
"""
@inline function Broadcast.materialize!(::CollectionVectorStyle{Names}, dest,
        bc::Broadcast.Broadcasted) where {Names}
    style = CollectionVectorStyle{Names}()
    return copyto!(dest, Broadcast.instantiate(
        Broadcast.Broadcasted(style, bc.f, bc.args, axes(dest))))
end

"""
    copy(bc::Broadcasted{<:CollectionVectorStyle}) -> CollectionVector

Evaluate a broadcast into a new `CollectionVector` with the field names and field lengths of
its `CollectionVector` arguments.

The field types follow the result type of the function, as for ordinary arrays: `x .* x` has
squared units, `ustrip.(x)` has unit-free fields, and `abs.(z)` makes a complex field real,
with half as many slots. The storage type is the promotion of the storage types of the field
results, e.g. `Bool` for `x .> zero.(x)`.

As in the `CollectionVector` constructor, the first element of each field's result sets the
field type and later elements are converted into it. An `ArgumentError` naming the field is
thrown when a later element does not convert, or when a result element has a type that
`CollectionVector` cannot store.

The first element of each field is computed twice: once to determine the field type and once
when the result is written.
"""
function Base.copy(bc::Broadcast.Broadcasted{CollectionVectorStyle{Names}}) where {Names}
    cv = find_container(bc)
    Tnew, Snew = resultshape(eltype(rawdata(cv)), shapeof(cv), bc)
    dest = attach(Snew, similar(rawdata(cv), Tnew, shape_length(Tnew, Snew)))
    return copyto!(dest, bc)
end

"""
    copyto!(dest::CollectionVector, bc::Broadcasted{<:CollectionVectorStyle}) -> dest

Evaluate a broadcast into `dest` field by field. `dest` must have the field names of the
broadcast.

Each field's results are converted into the element type of that field of `dest`, e.g. from
`cm` into `m`, from `Float32` into `Float64`, or from a real number into a complex field. An
error while evaluating or converting a field is rethrown as an `ArgumentError` whose message
names the field and includes the original error. Fields are written in order, so the fields
before the failing one already hold their new values.

Arguments that may share memory with `dest`, other than `dest` itself, are copied first, so
the right-hand side is read in full before any field is written. Ordinary 1D arrays are split
by the slot layout of a `CollectionVector` in the broadcast, or by that of `dest` when the
broadcast contains none.
"""
function Base.copyto!(dest::CollectionVector{<:Any, S},
        bc::Broadcast.Broadcasted{CollectionVectorStyle{Names}}) where {S, Names}
    keys(S) == Names || throw(ArgumentError(
        "Cannot broadcast into a CollectionVector with fields $(keys(S)) from a broadcast " *
        "over fields $Names"))
    src = segmentsource(find_container(bc), dest)
    segs = segments(eltype(rawdata(src)), shapeof(src))
    bcu = unaliasargs(dest, bc)
    broadcastfields!(dest, bcu, segs, valnames(Val(Names)))
    return dest
end

broadcastfields!(dest, bc, segs, ::Tuple{}) = nothing
@inline function broadcastfields!(dest, bc, segs, names::Tuple)
    broadcastfield!(dest, bc, segs, first(names))
    return broadcastfields!(dest, bc, segs, Base.tail(names))
end
@inline function broadcastfield!(dest, bc, segs, ::Val{name}) where {name}
    bcf = unpack_broadcast(bc, Val(name), getfield(segs, name))
    spec = getfield(shapeof(dest), name)
    return storefield!(dest, Val(name), spec[1], spec[2], bcf)
end
# Evaluate one field's broadcast into the destination, rethrowing any error with the
# field name.
function storefield!(dest, ::Val{name}, ::Int, ft, bcf) where {name}
    try
        setproperty!(dest, name, Broadcast.materialize(bcf))
    catch e
        throw(storeerror(name, eltype(rawdata(dest)), ft, e))
    end
    return nothing
end
function storefield!(dest, ::Val{name}, ::UnitRange{Int}, ft, bcf) where {name}
    try
        Broadcast.materialize!(getproperty(dest, name), bcf)
    catch e
        throw(storeerror(name, eltype(rawdata(dest)), ft, e))
    end
    return nothing
end
function storeerror(name, ::Type{T}, ft, e) where {T}
    ArgumentError("Field '$name': broadcasting into elements of type " *
                  "$(elementtype(T, ft)) failed. Cause: $(sprint(showerror, e))")
end

"""
    copyto!(dest::AbstractVector, bc::Broadcasted{<:CollectionVectorStyle}) -> dest

Evaluate a broadcast into an ordinary vector `dest`, as in `r .= ustrip.(x .- y)`.

`dest` is laid out like the storage of the out-of-place result (see `copy`): each field's
results go to that field's slots in the result's layout. Elements are converted into
`eltype(dest)`, so a dimensionful result throws a `Unitful.DimensionError` for a
`Vector{Float64}` destination. A field whose result elements occupy several slots each, such
as a complex field, throws an `ArgumentError` naming the field.

`dest` must have one element per storage slot of the result. With `.=`, Base also checks
`dest` against the length of the `CollectionVector` arguments, so the result must have as
many slots as the input. Otherwise a `DimensionMismatch` is thrown.

As for `copy`, the first element of each field is computed twice.
"""
function Base.copyto!(dest::AbstractVector,
        bc::Broadcast.Broadcasted{CollectionVectorStyle{Names}}) where {Names}
    cv = find_container(bc)
    T, S = eltype(rawdata(cv)), shapeof(cv)
    Tnew, Snew = resultshape(T, S, bc)
    n = shape_length(Tnew, Snew)
    length(dest) == n || throw(DimensionMismatch(
        "Destination has length $(length(dest)), the broadcast result has $n storage slots"))
    bcu = unaliasargs(dest, bc)
    segs, destsegs = segments(T, S), segments(Tnew, Snew)
    storesegments!(dest, bcu, segs, destsegs, valnames(Val(Names)))
    return dest
end

storesegments!(dest, bc, segs, destsegs, ::Tuple{}) = nothing
@inline function storesegments!(dest, bc, segs, destsegs, names::Tuple)
    storesegment!(dest, bc, segs, destsegs, first(names))
    return storesegments!(dest, bc, segs, destsegs, Base.tail(names))
end
@inline function storesegment!(dest, bc, segs, destsegs, ::Val{name}) where {name}
    bcf = unpack_broadcast(bc, Val(name), getfield(segs, name))
    Broadcast.materialize!(segmentview(dest, getfield(destsegs, name), name), bcf)
    return nothing
end
