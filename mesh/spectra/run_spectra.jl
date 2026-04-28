using JSON
include(joinpath(@__DIR__, "spectra.jl"))

if length(ARGS) < 2
    println("Usage: julia run_spectra.jl <input.json> <output.json>")
    exit(1)
end

input_path = ARGS[1]
output_path = ARGS[2]

data = JSON.parsefile(input_path)

# Convert list of lists to Matrix{Float64}
# JSON arrays are row-major, hcat makes them column-major vectors, so we transpose
D_raw = data["D"]
D = Matrix(Float64.(hcat(D_raw...))')

G = Int(data["G"])
delta = Float64(data["delta"])

# (1) Run SPECTRA
P, w = SPECTRA(D)

# (2) Run alg3
makespan, schedule = alg3(w; s=G, delta=delta)

output = Dict(
    "P" => P,
    "schedule" => schedule
)

open(output_path, "w") do f
    JSON.print(f, output)
end