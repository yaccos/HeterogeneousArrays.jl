using Unitful
using HeterogeneousArrays
using ComponentArrays
using RecursiveArrayTools

r0 = [1131.34, -2282.34, 6672.42]u"km"
v0 = [-5.64, 4.30, 2.42]u"km/s"

u0_basic = [r0; v0]
u0_recursive = ArrayPartition(r0, v0)
u0_component = ComponentVector(r = r0, v = v0)
u0_het = HeterogeneousVector(r = r0, v = v0)

u0_basic
print(u0_basic)

u0_recursive
print(u0_recursive)

u0_component
print(u0_component)

u0_het
print(u0_het)

# Helper functions
_show_unwrap(x::Ref) = x[]
_show_unwrap(x) = x
_show_pairs(hv) = (k => _show_unwrap(v) for (k, v) in pairs(NamedTuple(hv)))

# numeric type without the unit noise
_numstr(::Type{T}) where {T} = string(T)
_numstr(::Type{Q}) where {Q <: Unitful.AbstractQuantity} = string(Unitful.numtype(Q))
_unit(v) = unit(eltype(v))
_unitstr(v) = _unit(v) == Unitful.NoUnits ? "" : " [$(_unit(v))]"

_kindstr(v::AbstractArray) = "$(length(v))-element $(_numstr(eltype(v)))$(_unitstr(v))"
_kindstr(v) = "$(_numstr(typeof(v)))$(_unitstr(v))"

# value column: numbers only, unit shown once
function _valstr(v::AbstractArray{<:Unitful.AbstractQuantity})
    "$(sprint(show, ustrip.(v); context = :compact => true)) $(_unit(v))"
end
_valstr(v::AbstractArray) = sprint(show, v; context = :compact => true)
_valstr(v) = repr(v)

# Option A

function Base.show(io::IO, ::MIME"text/plain", hv::AbstractHeterogeneousVector)
    ps = collect(_show_pairs(hv))
    println(io, nameof(typeof(hv)), " (", length(hv), "-element flattened view)")
    pad = maximum(length ∘ string ∘ first, ps; init = 0)
    for (i, (k, v)) in enumerate(ps)
        branch = i == lastindex(ps) ? "└─ " : "├─ "
        println(io, "  ", branch, rpad(string(k), pad), " = ", repr(v))
    end
end

u0_het

# Option B, thanks Revise.jl

function Base.show(io::IO, hv::AbstractHeterogeneousVector)
    print(io, nameof(typeof(hv)), "(")
    for (i, (k, v)) in enumerate(_show_pairs(hv))
        i > 1 && print(io, ", ")
        print(io, k, " = ", repr(v))
    end
    print(io, ")")
end

u0_het
print(u0_het)

# Option C

function Base.show(io::IO, ::MIME"text/plain", hv::AbstractHeterogeneousVector)
    ps = collect(_show_pairs(hv))
    println(io, nameof(typeof(hv)), " with ", length(ps), " members, ",
        length(hv), " elements total:")
    kpad = maximum(length ∘ string ∘ first, ps; init = 0)
    tags = ["[" * _kindstr(v) * "]" for (_, v) in ps]
    tpad = maximum(length, tags; init = 0)
    for ((k, v), tag) in zip(ps, tags)
        println(io, "  ", rpad(string(k), kpad), "  ", rpad(tag, tpad), "  ", _valstr(v))
    end
end

u0_het

using DynamicQuantities

hv_dynamic = HeterogeneousVector(v = [2.2, 9.2, 3.145]us"km/s", r = [29.0, 205.5, 30.2]us"km")

using Unitful
hv_unstable = HeterogeneousVector(
    v = [2.2Unitful.u"m", 9.2Unitful.u"s", 3.145Unitful.u"kg"], r = 6.4145Unitful.u"V")

# Evil examples
using Unitful
c = CollectionVector(a = [4.5u"m", 9.5u"kg"], b = [3.134, -5, 2]u"rad", c = 8.15u"A")
HeterogeneousVector(a = [4.5u"m", 9.5u"kg"], b = [3.134, -5, 2]u"rad", c = 8.15u"A")

struct FakeQuantity{T} <: Number
    value::T
    unit::Symbol
end

d = CollectionVector(a = [FakeQuantity(2.2, :kms), FakeQuantity(9.2, :kms)])

import DynamicQuantities
d_dynamic = CollectionVector(v = [2.2, 9.2, 3.145]DynamicQuantities.us"km/s",
    r = [29.0, 205.5, 30.2]DynamicQuantities.us"km")
HeterogeneousArrays.rawdata(d_dynamic)
u_raw = HeterogeneousArrays.rawdata(d_dynamic)
u_reconstructed = HeterogeneousArrays.attach(typeof(u).paramaters[2], u_raw)
S = typeof(d_dynamic).parameters[2]
u_reconstructed = HeterogeneousArrays.attach(S, u_raw)

d_dynamic_nonstable = CollectionVector(
    v = [
        2.2DynamicQuantities.us"km/s", 9.2DynamicQuantities.us"m", 3.1459DynamicQuantities.us"s"],
    r = [29.0, 205.5, 30.2]DynamicQuantities.us"km")
HeterogeneousArrays.rawdata(d_dynamic_nonstable)

CollectionVector(a = [4.5u"m", 9.5u"ft"], b = [3.134, -5, 2]u"rad", c = 8.15u"A")

u = CollectionVector(a = [4.5u"m", 9.5u"ft"], b = [3.134, -5, 2]u"rad", c = 8.15)

u_raw = HeterogeneousArrays.rawdata(u)

function get_S(x::CollectionVector{T, S, D, E}) where {T, S, D, E}
    S
end

S = get_S(u)

u_reconstructed = HeterogeneousArrays.attach(S, u_raw)

u = CollectionVector(a = Any[4.5u"m", 9.5u"ft"], b = Any[3.134u"rad", -5.2u"deg"])

u_raw = HeterogeneousArrays.rawdata(u)

function get_S(x::CollectionVector{T, S, D, E}) where {T, S, D, E}
    S
end

S = get_S(u)

u_reconstructed = HeterogeneousArrays.attach(S, u_raw)

u = CollectionVector(a = [], b = Any[3.134u"rad", -5.2u"deg"])

struct orbital_mechanics_variables{DistanceType, SpeedType}
    r::Vector{DistanceType}
    v::Vector{SpeedType}
end

isbitstype(typeof(2.2DQ.us"km/s")) # false
isbitstype(typeof(2.2DQ.u"km/s")) # true

a = 2.2DQ.u"km/s"
as = 2.2DQ.us"km/s"

d_dynamic_nonstable = CollectionVector(
    v = [
        2.2DynamicQuantities.u"km/s", 9.2DynamicQuantities.u"m", 3.1459DynamicQuantities.u"s"],
    r = [29.0, 205.5, 30.2]DynamicQuantities.u"km")

2.2DQ.u"km/s" isa Number

similar([4.5u"m", 9.5u"kg"])

zero([4.5u"m", 9.5u"kg"])

import DifferentialEquations as DE
using LinearAlgebra
function f_raw_allocating(y, μ, t)
    r = view(y, 1:3)
    v = view(y, 4:6)
    r_mag = norm(r)
    dr = v
    dv = -μ .* r ./ r_mag^3
    return [dr; dv]
end

r0_raw = [1131.34, -2282.34, 6672.42]
v0_raw = [-5.64, 4.30, 2.42]
u0_raw = [r0_raw; v0_raw]
Δt_raw = 3600.0*100
tspan_raw = (0.0, Δt_raw)
μ_raw = 398600.44

prob = DE.ODEProblem(f_raw_allocating, u0_raw, tspan_raw, μ_raw)

DE.solve(prob; alg = DE.Tsit5(), adaptive = true, dt = Δt_raw)
n_objects = 3

r0_unitful = r0_raw * Unitful.u"km"
v0_unitful = v0_raw * Unitful.u"km/s"
μ_unitful = μ_raw * Unitful.u"km^3/s^2"
Δt_unitful = Δt_raw * Unitful.u"s"
u0_unitful = [r0_unitful; v0_unitful]

tspan_unitful = (0.0 * Unitful.u"s", Δt_unitful)
prob = DE.ODEProblem(f_raw_allocating, u0_unitful, tspan_unitful, μ_unitful)

DE.solve(prob; alg = DE.Tsit5(), adaptive = true, dt = Δt_raw)

zero(real(ustrip(eltype(u0_unitful))))

u = CollectionVector(a = Any[402.24f0, 41.2])

u = CollectionVector(a = Any[402.24f0, 41.2f0])
