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
#   3. return the solution with the units attached again: an `ODESolution` on the user's
#      problem, interpolated through `CVInterpolation`.
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

# Shape of the n-th time derivative of u: same slots, each unit divided by tu^n. The type
# of `tu^n` depends on the value of `n`, so `n` enters through a `Val` and the whole map
# is computed at compile time. A dimensionless result is stored as a plain number
# (`nothing`), as in the constructor.
derivative_shape(SU, tu, n = 1) = derivative_shape(SU, tu, Val(n))
derivative_shape(SU, tu, ::Val{n}) where {n} = map(((slot, ft),) -> (slot, dunit(ft, tu^n)), SU)
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
# Callable struct with the shapes and the time unit as type parameters, so `attach` is
# resolved at compile time: wrapping the solver's arrays costs nothing and allocates
# nothing, and this also works for ForwardDiff.Dual arrays (autodiff Jacobians)

struct RawRHS{IIP, SU, SD, F, TU} <: Function
    f::F    # the user's ODE function
end

# In-place: the user writes into a unitful view of the solver's `du`. Compatible units are
# converted on assignment, a wrong dimension throws a DimensionError.
function (r::RawRHS{true, SU, SD, F, TU})(du, u, p, t) where {SU, SD, F, TU}
    r.f(attach(SD, du), attach(SU, u), p, t * TU)
    return nothing
end
# Out-of-place: pass through f's own storage when it already has the shape and units of SD;
# otherwise copy into SD, which converts units and checks the fields. Pass-through is what a
# plain out-of-place function already promises (fresh vector per call, solver owns it).
function (r::RawRHS{false, SU, SD, F, TU})(u, p, t) where {SU, SD, F, TU}
    du = r.f(attach(SU, u), p, t * TU)
    data = rawdata(du)
    if shapeof(du) == SD && typeof(data) === typeof(u)
        return data  # f's own storage is already what the solver expects
    end
    out = attach(SD, similar(u))
    copyto!(out, du)
    return rawdata(out)
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
    tu = time_unit(first(prob.tspan))
    iip = SciMLBase.isinplace(prob)
    # Same problem on the raw storage. `p` is passed unchanged (the user's function gets
    # it as-is), and the specialization of the user's ODEFunction is kept.
    rhs = RawRHS{iip, SU, derivative_shape(SU, tu), typeof(prob.f.f), tu}(prob.f.f)
    f = SciMLBase.ODEFunction{iip, SciMLBase.specialization(prob.f)}(rhs)
    tspan = strip_time(tu, prob.tspan, :tspan)
    rawprob = SciMLBase.ODEProblem{iip}(f, copy(rawdata(prob.u0)), tspan, prob.p)

    rawkw = raw_kwargs(SU, rawprob.u0, tu; kwargs...)
    rawsol = SciMLBase.__solve(rawprob, w.alg, args...; rawkw...)
    return SciMLBase.build_solution(rawsol, w, prob, SU, tu)
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
# A regular `ODESolution` on the user's problem, built like OrdinaryDiffEq builds its own.
# `sol.u` are views over the raw states, `sol.original` is the raw solution, and `sol.k` is
# `nothing` (stage units depend on the solver). `CVInterpolation` handles the units in `sol(t)`.

struct CVInterpolation{SU, TU, R} <: SciMLBase.AbstractDiffEqInterpolation
    raw::R  # the raw solution
end
CVInterpolation(raw, SU, tu) = CVInterpolation{SU, tu, typeof(raw)}(raw)

SciMLBase.interp_summary(ci::CVInterpolation) = SciMLBase.interp_summary(ci.raw.interp)

# `sol(t, Val{n}; idxs)` lands here: u(t), or its n-th derivative in units of u / tu^n.
# `idxs` (an index or a vector of indices) picks elements of that state.
function (ci::CVInterpolation{SU, TU})(t::Number, idxs, ::Type{Val{n}}, p,
        continuity) where {SU, TU, n}
    x = ci.raw(strip_time(TU, t, :t), Val{n}; continuity)
    u = attach(derivative_shape(SU, TU, Val(n)), x)
    return idxs === nothing ? u : u[idxs]
end
function (ci::CVInterpolation)(t::AbstractVector{<:Number}, idxs, deriv, p, continuity)
    return SciMLBase.DiffEqArray([ci(ti, idxs, deriv, p, continuity) for ti in t], t)
end
function (ci::CVInterpolation)(v, t, idxs, deriv, p, continuity)
    throw(ArgumentError("cv_wrapper: in-place interpolation `sol(v, t)` is not supported"))
end

function SciMLBase.build_solution(rawsol::SciMLBase.AbstractODESolution, w::CVWrapper,
        prob::SciMLBase.AbstractODEProblem, SU, tu)
    u = [attach(SU, x) for x in rawsol.u]
    t = rawsol.t .* tu
    return SciMLBase.build_solution(prob, w, t, u;
        interp = CVInterpolation(rawsol, SU, tu), dense = rawsol.dense,
        alg_choice = rawsol.alg_choice, calculate_error = false, stats = rawsol.stats,
        retcode = rawsol.retcode, original = rawsol)
end

end # module
