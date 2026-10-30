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

# Storage type and field type of one field of the result, as a pair `(R, ft)`, from the
# per-field broadcast `bcf`. When inference gives a concrete element type `Q`, and the field
# type of a `Q` is inferred as a singleton type (a unit, `Nothing`, ...), both come from `Q`
# and no element is evaluated. Otherwise (e.g. a function whose result unit or realness
# depends on the value) they come from the first element of the field's result.
@inline function fieldresult(bcf, ::Val{name}, slot) where {name}
    Q = Broadcast.combine_eltypes(bcf.f, bcf.args)
    if isconcretetype(Q)
        checkresult(name, Q)
        FT = Base.promote_op(field_type, Q)
        Base.issingletontype(FT) && return (rawtype(Q), FT.instance)
    end
    q = firstresult(bcf, Val(name), slot)
    return (rawtype(typeof(q)), field_type(q))
end

# First element of the per-field broadcast `bcf`. A scalar field (`Int` slot) gives a
# 0-dimensional broadcast, an array field (`UnitRange` slot) a 1D one.
function firstresult(bcf, ::Val{name}, ::Int) where {name}
    q = Broadcast.materialize(bcf)
    checkresult(name, typeof(q))
    return q
end
function firstresult(bcf, ::Val{name}, ::UnitRange{Int}) where {name}
    bci = Broadcast.instantiate(bcf)
    q = bci[first(eachindex(bci))]
    checkresult(name, typeof(q))
    return q
end
function checkresult(name, ::Type{Q}) where {Q}
    is_storable(Q) || throw(ArgumentError(
        "Field '$name': the broadcast produces elements of type $Q, which CollectionVector " *
        "does not support"))
    return nothing
end

# Storage type and shape of the result of broadcasting `bc`, whose CollectionVector inputs
# have storage type `T` and shape `S`. Each field's storage type and field type come from
# `fieldresult`. Field lengths come from `S`.
@inline function resultshape(::Type{T}, S::NamedTuple, bc) where {T}
    segs = segments(T, S)
    rs = fieldresults(bc, segs, S, valnames(Val(keys(S))))
    Tnew = storagetype(rs)
    return Tnew, NamedTuple{keys(S)}(resultspecs(Tnew, T, values(S), rs, 0))
end

fieldresults(bc, segs, S, ::Tuple{}) = ()
@inline function fieldresults(bc, segs, S, names::Tuple)
    r = fieldresult(bc, segs, S, first(names))
    return (r, fieldresults(bc, segs, S, Base.tail(names))...)
end
@inline function fieldresult(bc, segs, S, ::Val{name}) where {name}
    bcf = unpack_broadcast(bc, Val(name), getfield(segs, name))
    return fieldresult(bcf, Val(name), getfield(S, name)[1])
end

# Promotion of the storage types in the field results `rs`
storagetype(::Tuple{}) = Union{}
storagetype(rs::Tuple) = promote_type(first(rs)[1], storagetype(Base.tail(rs)))

# Slot and field type of each field of the result, from the input specs and the field
# results `rs`
resultspecs(::Type, ::Type, ::Tuple{}, ::Tuple{}, offset) = ()
function resultspecs(
        ::Type{Tnew}, ::Type{T}, specs::Tuple, rs::Tuple, offset) where {Tnew, T}
    slot, ft = first(specs)
    ftnew = first(rs)[2]
    n = slotcount(Tnew, ftnew) * fieldlength(T, slot, ft)
    spec = (resultslot(slot, offset, n), ftnew)
    return (spec, resultspecs(Tnew, T, Base.tail(specs), Base.tail(rs), offset + n)...)
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

The field types are determined without evaluating the function when inference gives a
concrete result type for a field, as for `x .* x` or `ustrip.(x)`. When it does not, e.g. for a
function whose result unit depends on the value, the first element of that field is computed
twice: once to determine the field type and once when the result is written.
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
before the failing one already hold their new values. The failing field is found by
re-evaluating the broadcast field by field into a scratch vector, after `dest` has been
partly written. If no field fails on re-evaluation, the original error is rethrown unchanged.

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
    names = valnames(Val(Names))
    # One exception handler per assignment
    try
        broadcastfields!(dest, bcu, segs, names)
    catch e
        throw(failing_field_error(dest, bcu, segs, names, e))
    end
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
    return storefield!(dest, Val(name), spec[1], bcf)
end
# Evaluate one field's broadcast into the destination. A scalar field is assigned as a
# value, an array field is written in place through its view.
function storefield!(dest, ::Val{name}, ::Int, bcf) where {name}
    setproperty!(dest, name, Broadcast.materialize(bcf))
    return nothing
end
function storefield!(dest, ::Val{name}, ::UnitRange{Int}, bcf) where {name}
    Broadcast.materialize!(getproperty(dest, name), bcf)
    return nothing
end

# Error path of `copyto!`: redo the assignment field by field into a scratch vector and
# return an `ArgumentError` naming the first field that throws, or the original error `e`
# when none does. Runs once per failed broadcast.
function failing_field_error(dest, bc, segs, names, e)
    scratch = attach(shapeof(dest), similar(rawdata(dest)))
    for name in names
        err = field_error(scratch, bc, segs, name)
        err === nothing || return err
    end
    return e
end
function field_error(scratch, bc, segs, ::Val{name}) where {name}
    try
        broadcastfield!(scratch, bc, segs, Val(name))
    catch e
        ft = getfield(shapeof(scratch), name)[2]
        return ArgumentError("Field '$name': broadcasting into elements of type " *
                             "$(elementtype(eltype(rawdata(scratch)), ft)) failed. " *
                             "Cause: $(sprint(showerror, e))")
    end
    return nothing
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

As for `copy`, the first element of a field is computed twice when inference does not give a
concrete result type for it.
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
