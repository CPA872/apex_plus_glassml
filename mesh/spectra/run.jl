include("spectra.jl")
D = traffic_matrix(5)
display(D)
P, w = SPECTRA(D)
display(P)
display(w)
makespan, schedule = alg3(w; s=4)
display(makespan)
display(schedule)

# Save the schedule to a file for plotting
open("schedule.json", "w") do f
    JSON.print(f, schedule)
end

# Run the Python script to generate the plot
run(`python3 plot_schedule.py`)
# P, w = wwfa(D)
# display(P)
# display(w)
# makespan = alg2(w; s=4)
# display(makespan)