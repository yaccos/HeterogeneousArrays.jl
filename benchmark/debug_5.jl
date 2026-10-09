import OrdinaryDiffEq as ODE
case = build_case(:heterogeneousvector, :flexunits, :inplace)
prob, dt = case
case = (prob=prob, dt=dt)
abstol=ComponentVector(r=1e-5*oneunit.(r0_flex),v=1e-5*oneunit.(r0_flex))
reltol = ComponentVector(r=1e-5*oneunit.(v0_flex),v=1e-5*oneunit.(v0_flex))
ODE.solve(case.prob; alg = DE.Tsit5(), adaptive = true,abstol=abstol,
reltol = reltol, dt = case.dt)