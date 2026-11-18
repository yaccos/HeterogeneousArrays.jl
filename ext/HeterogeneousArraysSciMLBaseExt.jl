module HeterogeneousArraysSciMLBaseExt

# Implementation of `cv_wrapper` (documented in src/solver_wrapper.jl).
#
# `solve(prob, cv_wrapper(alg); kwargs...)` runs DiffEqBase's normal `solve` checks and then
# calls `SciMLBase.__solve(prob, ::CVWrapper; kwargs...)` below. There we
#   1. build the same problem on the unitless storage `rawdata(u0)`, with times in the unit
#      of `tspan`, and an ODE function (`RawRHS`) that re-attaches the units before calling
#      the user's function,
#   2. solve that plain-Vector problem with the wrapped algorithm (so Jacobian-based solvers
#      work),
#   3. return the solution with the units attached again (`CVSolution`).
#
# Notation: SU = shape of u (`shapeof(u0)`: field names, storage slots and units),
# SD = shape of du (units of u divided by the time unit), tu = time unit.

using HeterogeneousArrays: HeterogeneousArrays, CollectionVector, rawdata, attach, shapeof
using SciMLBase: SciMLBase
using Unitful: Unitful, NoUnits, unit, ustrip, dimension

## The wrapper type: only marks "solve on the raw storage" and holds the real algorithm
## (`nothing` = default algorithm).

struct CVWrapper{A} <: SciMLBase.AbstractODEAlgorithm
    alg::A
end
HeterogeneousArrays.cv_wrapper(alg = nothing) = CVWrapper(alg)

# Print as the call the user wrote, `cv_wrapper(Rodas5P(...))`, instead of the internal type
# name. Adds a `show` method for CVWrapper only; nothing else is affected.
Base.show(io::IO, w::CVWrapper) = print(io, "cv_wrapper(", something(w.alg, ""), ")")

# DiffEqBase checks these traits on the unitful problem before calling `__solve`:
# without the first it would reject the Quantity `tspan`.
SciMLBase.allows_arbitrary_number_types(::CVWrapper) = true
SciMLBase.isadaptive(w::CVWrapper) = w.alg === nothing || SciMLBase.isadaptive(w.alg)

## Units

time_unit(t::Unitful.AbstractQuantity) = unit(t)
time_unit(::Real) = NoUnits

# Shape of the n-th time derivative of u: same slots, each unit divided by tu^n.
# A dimensionless result is stored as a plain number (`nothing`), as in the constructor.
derivative_shape(SU, tu, n = 1) = map(((slot, ft),) -> (slot, dunit(ft, tu^n)), SU)
dunit(ft::Unitful.Units, d) = plain(ft / d)
dunit(::Nothing, d) = plain(inv(d))
plain(u) = u == NoUnits ? nothing : u

# A time (or collection of times) as plain number(s) in the unit `tu`. Throws if the
# dimension does not match `tspan`; `name` is the keyword argument, for the error message.
function strip_time(tu, x, name)
    dimension(x) == dimension(tu) || throw(ArgumentError(
        "cv_wrapper: `$name` is $x, but `tspan` has unit $tu. They must have the same dimension."))
    return float(ustrip(tu, x))
end
function strip_time(tu, x::Union{AbstractArray, Tuple}, name)
    map(xi -> strip_time(tu, xi, name), x)
end
strip_time(tu, ::Nothing, name) = nothing

## The ODE function on raw storage
#
# Callable struct with the shapes as type parameters, so `attach` is resolved at compile
# time: wrapping the solver's arrays costs nothing and allocates nothing. This also works
# when the arrays hold ForwardDiff.Dual numbers, so autodiff Jacobians work.

struct RawRHS{IIP, SU, SD, F, TU} <: Function
    f::F    # the user's ODE function
    tu::TU  # time unit, to give `t` its unit back
end

# In-place: the user writes into a unitful view of the solver's `du`. Compatible units are
# converted on assignment, a wrong dimension throws a DimensionError.
function (r::RawRHS{true, SU, SD})(du, u, p, t) where {SU, SD}
    r.f(attach(SD, du), attach(SU, u), p, t * r.tu)
    return nothing
end
# Out-of-place: copy the returned CollectionVector into the units of SD.
function (r::RawRHS{false, SU, SD})(u, p, t) where {SU, SD}
    du = attach(SD, similar(u))
    copyto!(du, r.f(attach(SU, u), p, t * r.tu))
    return rawdata(du)
end

## Translating the problem and the keyword arguments

const TIME_KWARGS = (:dt, :dtmin, :dtmax, :saveat, :tstops, :d_discontinuities)

# abstol per field, as raw numbers in the units of u0. A NamedTuple or CollectionVector is
# copied field by field into the shape of u0 (`copyto!` converts units and checks that the
# fields match). A plain number is rejected: its unit would be ambiguous.
function raw_abstol(SU, u0raw, tol)
    tol isa Real && throw(ArgumentError(
        "cv_wrapper: `abstol = $tol` has no unit. Give it per field with units, as a " *
        "NamedTuple like `(x = 1e-6u\"m\", v = 1e-6u\"m/s\")` or a CollectionVector."))
    tol isa NamedTuple && (tol = CollectionVector(tol))
    out = attach(SU, similar(u0raw))
    copyto!(out, tol)
    return rawdata(out)
end

function raw_kwargs(SU, u0raw, tu; kwargs...)
    kw = Dict{Symbol, Any}(kwargs)
    # Callbacks would be called with the raw integrator (no units): refuse them for now.
    cb = get(kw, :callback, nothing)
    cb === nothing ||
        cb isa SciMLBase.CallbackSet &&
        isempty(cb.continuous_callbacks) &&
        isempty(cb.discrete_callbacks) ||
        throw(ArgumentError("cv_wrapper does not support callbacks yet"))
    for k in TIME_KWARGS
        haskey(kw, k) && (kw[k] = strip_time(tu, kw[k], k))
    end
    haskey(kw, :abstol) && (kw[:abstol] = raw_abstol(SU, u0raw, kw[:abstol]))
    return NamedTuple(kw)  # all other keyword arguments are passed on unchanged
end

function SciMLBase.__solve(prob::SciMLBase.AbstractODEProblem, w::CVWrapper, args...;
        kwargs...)
    prob.u0 isa CollectionVector || throw(ArgumentError(
        "cv_wrapper needs a CollectionVector as u0, got a $(typeof(prob.u0))"))
    SU = shapeof(prob.u0)
    tu = time_unit(prob.tspan[1])
    iip = SciMLBase.isinplace(prob)
    # Same problem on the raw storage. `p` is passed unchanged (the user's function gets
    # it as-is), and the specialization of the user's ODEFunction is kept.
    rhs = RawRHS{iip, SU, derivative_shape(SU, tu), typeof(prob.f.f), typeof(tu)}(prob.f.f, tu)
    f = SciMLBase.ODEFunction{iip, SciMLBase.specialization(prob.f)}(rhs)
    tspan = strip_time(tu, prob.tspan, :tspan)
    rawprob = SciMLBase.ODEProblem{iip}(f, copy(rawdata(prob.u0)), tspan, prob.p)

    rawkw = raw_kwargs(SU, rawprob.u0, tu; kwargs...)
    algs = w.alg === nothing ? () : (w.alg,)
    rawsol = SciMLBase.solve(rawprob, algs..., args...; rawkw...)
    return CVSolution(rawsol, SU, tu)
end

# The integrator interface (`init`, `step!`, ...) is not supported: it would need an
# integrator wrapper with units (also needed for callbacks). Without this method `init`
# would fall back to DiffEqBase's default and fail with an unrelated error.
function SciMLBase.__init(::SciMLBase.AbstractODEProblem, ::CVWrapper, args...; kwargs...)
    throw(ArgumentError("cv_wrapper supports `solve` only; the integrator interface " *
                        "(`init`, `step!`, ...) is not implemented yet"))
end

## The solution with units
#
# `sol.u` are CollectionVector views over the raw saved states (no copies), `sol.t` has
# units, and every other property (retcode, stats, ...) comes from the raw solution `sol.raw`.

struct CVSolution{SU, R, TU, UV, TV}
    raw::R
    tu::TU
    u::UV
    t::TV
end
function CVSolution(raw, SU, tu)
    u = [attach(SU, x) for x in raw.u]
    t = raw.t .* tu
    return CVSolution{SU, typeof(raw), typeof(tu), typeof(u), typeof(t)}(raw, tu, u, t)
end

function Base.getproperty(sol::CVSolution, name::Symbol)
    name in fieldnames(CVSolution) ? getfield(sol, name) :
    getproperty(getfield(sol, :raw), name)
end

# Interpolation with a unitful time (or a vector of times): `sol(t)` gives u(t),
# `sol(t, Val{n})` the n-th derivative, in units of u / tu^n.
function (sol::CVSolution{SU})(t, ::Type{Val{n}} = Val{0}) where {SU, n}
    t isa AbstractVector && return [sol(ti, Val{n}) for ti in t]
    x = sol.raw(strip_time(sol.tu, t, :t), Val{n})
    return attach(derivative_shape(SU, sol.tu, n), x)
end

end # module
