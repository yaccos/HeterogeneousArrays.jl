unitful_allocating_case = cases[3]
sol = DE.solve(unitful_allocating_case.prob; alg = DE.Tsit5(), abstol=ComponentVector(r=1e-5*oneunit.(r0_unitful),v=1e-5*oneunit.(v0_unitful)), adaptive = true, dt = unitful_allocating_case.dt)

skipped = NamedTuple{(:array_label, :unit_label, :interface_label, :reason)}[]
for case in [unitful_allocating_case]
    try
        DE.solve(case.prob; alg = DE.Tsit5(), adaptive = false, dt = case.dt)

        trial = @benchmark DE.solve($(case.prob); alg = DE.Tsit5(), dt = $(case.dt)) samples=100

        t_min = minimum(trial).time / 1e6
        stderror_ms = (std(trial.times) / sqrt(length(trial.times))) / 1e6
        allocs = trial.allocs
        memory = trial.memory
        mem_str = memory < 1024 ? "$(memory) B" : "$(round(memory/1024, digits=1)) KiB"

        println(rpad(case.array_label, 22),
            rpad(case.unit_label, 14),
            rpad(case.interface_label, 18),
            lpad(format_val(t_min), 12),
            lpad(format_val(stderror_ms), 15),
            lpad(allocs, 12),
            lpad(mem_str, 15))
    catch err
        push!(skipped, (array_label = case.array_label, unit_label = case.unit_label, interface_label = case.interface_label, reason = sprint(showerror, err)))
    end
end
unitful_allocating_case
DE.solve(case.prob; alg = DE.Tsit5(), adaptive = false, dt = case.dt)

if !isempty(skipped)
    println("Skipped incompatible combinations:")
    for item in skipped
        println("- ", item.array_label, " / ", item.unit_label, " / ", item.interface_label, ": ", item.reason)
    end
end

for case in [unitful_allocating_case]
    DE.solve(case.prob; alg = DE.Tsit5(), adaptive = true, dt = case.dt)

    trial = @benchmark DE.solve($(case.prob); alg = DE.Tsit5(), adaptive = true,  dt = $(case.dt)) samples=100

    t_min = minimum(trial).time / 1e6
    stderror_ms = (std(trial.times) / sqrt(length(trial.times))) / 1e6
    allocs = trial.allocs
    memory = trial.memory
    mem_str = memory < 1024 ? "$(memory) B" : "$(round(memory/1024, digits=1)) KiB"

    println(rpad(case.array_label, 22),
        rpad(case.unit_label, 14),
        rpad(case.interface_label, 18),
        lpad(format_val(t_min), 12),
        lpad(format_val(stderror_ms), 15),
        lpad(allocs, 12),
        lpad(mem_str, 15))
end
