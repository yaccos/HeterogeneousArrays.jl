import RecursiveArrayTools
case = build_case(:fieldvector, :unitful, :allocating)
prob, dt = case
case = (prob=prob, dt=dt)
#abstol = Base.similar(prob.u0)
#reltol = Base.similar(prob.u0)
abstol = 1e-5
reltol = 1e-5
RecursiveArrayTools.recursivecopy(u::OrbitFieldVector) = copy(u)
DE.solve(case.prob; alg = DE.Tsit5(), adaptive = true, abstol=abstol, reltol = reltol, dt = case.dt)
