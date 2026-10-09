import RecursiveArrayTools
using StaticArrays: FieldVector, SVector

r0_raw = [1131.34, -2282.34, 6672.42]
v0_raw = [-5.64, 4.30, 2.42]
μ_raw = 398600.44
Δt_raw = 3600.0
n_objects = 3

@kwdef struct OrbitFieldVector{T} <: FieldVector{2, SVector{n_objects, T}}
    r::SVector{n_objects, T}
    v::SVector{n_objects, T}
end

RecursiveArrayTools.recursivecopy(u::OrbitFieldVector) = copy(u)

u0 = OrbitFieldVector(r=SVector{n_objects}(r0_raw), v = SVector{n_objects}(v0_raw))
copy(u0)
deepcopy(u0)
RecursiveArrayTools.recursivecopy(u0)
