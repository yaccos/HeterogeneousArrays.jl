cases
cases[5]

case = cases[5]

DE.solve(case.prob; alg = DE.Tsit5(), adaptive = true,abstol=ComponentVector(r=1e-5*oneunit.(r0_flex),v=1e-5*oneunit.(r0_flex)),
reltol = ComponentVector(r=1e-5*oneunit.(v0_flex),v=1e-5*oneunit.(v0_flex)), dt = case.dt)
