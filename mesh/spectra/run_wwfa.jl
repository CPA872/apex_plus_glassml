using JSON
include(joinpath(@__DIR__, "spectra.jl"))

if length(ARGS) < 2
    println("Usage: julia run_wwfa.jl <input.json> <output.json>")
    exit(1)
end

input_path = ARGS[1]
output_path = ARGS[2]

data = JSON.parsefile(input_path)

# Convert list of lists to Matrix{Float64}
# JSON arrays are row-major, hcat makes them column-major vectors, so we transpose
D_raw = data["D"]
D = Matrix(Float64.(hcat(D_raw...))')

# Run WWFA
P, w = wwfa(D)

output = Dict(
    "P" => P,
    "w" => w
)

open(output_path, "w") do f
    JSON.print(f, output)
end