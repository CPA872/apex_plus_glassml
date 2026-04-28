#!/usr/bin/env julia
"""
Generate deterministic random test cases for SPECTRA and WWFA.
Saves inputs and reference outputs to tests/ directory.
"""

using JSON
include(joinpath(@__DIR__, "spectra.jl"))

const TEST_DIR = joinpath(@__DIR__, "tests")
mkpath(TEST_DIR)

# Fixed seed for reproducibility
rng_test = MersenneTwister(42)

# ── helpers ──────────────────────────────────────────────────────────

function rand_dense_matrix(rng, n)
    D = rand(rng, n, n)
    D .= D ./ max(maximum(sum(D, dims=1)), maximum(sum(D, dims=2)))
    return D
end

function rand_sparse_matrix(rng, n; density=0.3)
    D = zeros(n, n)
    for i in 1:n, j in 1:n
        if rand(rng) < density
            D[i, j] = rand(rng)
        end
    end
    D .= D ./ max(maximum(sum(D, dims=1)), maximum(sum(D, dims=2)), 1e-12)
    return D
end

function rand_doubly_stochastic(rng, n; iters=20)
    D = rand(rng, n, n)
    for _ in 1:iters
        D .= D ./ sum(D, dims=2)
        D .= D ./ sum(D, dims=1)
    end
    return D
end

function rand_skewed_matrix(rng, n; hot_fraction=0.2)
    D = zeros(n, n)
    hot_cols = sort(shuffle(rng, 1:n)[1:max(1, round(Int, n * hot_fraction))])
    for i in 1:n, j in 1:n
        if j in hot_cols
            D[i, j] = rand(rng) * 5.0
        else
            D[i, j] = rand(rng) * 0.2
        end
    end
    D .= D ./ max(maximum(sum(D, dims=1)), maximum(sum(D, dims=2)), 1e-12)
    return D
end

function rand_permutation_sum(rng, n; k=4)
    I_n = Matrix{Float64}(I, n, n)
    D = zeros(n, n)
    for _ in 1:k
        perm = I_n[shuffle(rng, 1:n), :]
        D .+= perm .* rand(rng)
    end
    return D
end

# ── invariant checks ─────────────────────────────────────────────────

function check_permutation(P, n)
    for i in 1:n
        @assert sum(P[i, :]) == 1 "Row $i does not sum to 1"
        @assert sum(P[:, i]) == 1 "Col $i does not sum to 1"
    end
    for i in 1:n, j in 1:n
        @assert P[i, j] in (0, 1) "P[$i,$j] = $(P[i,j]) not in {0,1}"
    end
end

function check_coverage(D, P_list, w_list; tol=1e-6)
    n = size(D, 1)
    reconstructed = zeros(n, n)
    for (P, w) in zip(P_list, w_list)
        reconstructed .+= w .* P
    end
    for i in 1:n, j in 1:n
        if reconstructed[i, j] < D[i, j] - tol
            return false, "Coverage violation at [$i,$j]: reconstructed=$(reconstructed[i,j]) < D=$(D[i,j])"
        end
    end
    return true, "OK"
end

function check_schedule(schedule, w_list, s; tol=1e-6)
    for (job_key, bins) in schedule
        job = isa(job_key, AbstractString) ? parse(Int, job_key) : Int(job_key)
        total = 0.0
        for (bin_key, interval) in bins
            bin = isa(bin_key, AbstractString) ? parse(Int, bin_key) : Int(bin_key)
            @assert 1 <= bin <= s "Bin $bin out of range [1, $s]"
            duration = interval[2] - interval[1]
            @assert duration >= -tol "Negative duration for job $job on bin $bin"
            total += duration
        end
        if abs(total - w_list[job]) > tol
            return false, "Job $job: scheduled duration $total != weight $(w_list[job])"
        end
    end
    return true, "OK"
end

function compute_makespan_from_schedule(schedule)
    max_end = 0.0
    for (_, bins) in schedule
        for (_, interval) in bins
            max_end = max(max_end, interval[2])
        end
    end
    return max_end
end

# ── test case definitions ────────────────────────────────────────────

struct TestCase
    name::String
    D::Matrix{Float64}
    s_values::Vector{Int}
    delta_values::Vector{Float64}
end

function generate_cases(rng)
    cases = TestCase[]

    # 1. Small matrices
    for n in [4, 6, 8]
        push!(cases, TestCase("dense_$(n)x$(n)", rand_dense_matrix(rng, n), [2, 4], [0.01, 0.05]))
    end

    # 2. Medium matrices
    for n in [16, 32]
        push!(cases, TestCase("dense_$(n)x$(n)", rand_dense_matrix(rng, n), [4, 8], [0.01, 0.04]))
    end

    # 3. Large matrices
    for n in [64, 72]
        push!(cases, TestCase("dense_$(n)x$(n)", rand_dense_matrix(rng, n), [4, 16], [0.01, 0.04]))
    end

    # 4. Sparse matrices
    for (n, dens) in [(8, 0.3), (16, 0.2), (32, 0.15), (64, 0.1), (72, 0.1)]
        push!(cases, TestCase("sparse_$(n)x$(n)_d$(Int(dens*100))", rand_sparse_matrix(rng, n; density=dens), [4], [0.01]))
    end

    # 5. Doubly stochastic
    for n in [8, 16, 32, 64, 72]
        push!(cases, TestCase("dstoch_$(n)x$(n)", rand_doubly_stochastic(rng, n), [4], [0.01]))
    end

    # 6. Skewed (MoE-like)
    for n in [16, 32, 64, 72]
        push!(cases, TestCase("skewed_$(n)x$(n)", rand_skewed_matrix(rng, n; hot_fraction=0.2), [4, 8], [0.01]))
    end

    # 7. Permutation sums (exact decomposition should be possible)
    for (n, k) in [(8, 3), (16, 5), (32, 8), (64, 12), (72, 14)]
        push!(cases, TestCase("permsum_$(n)x$(n)_k$(k)", rand_permutation_sum(rng, n; k=k), [4], [0.01]))
    end

    # 8. Edge cases
    # Single nonzero per row/col (already a permutation)
    I_perm = Matrix{Float64}(I, 8, 8)[shuffle(rng, 1:8), :] .* 0.5
    push!(cases, TestCase("single_perm_8x8", I_perm, [2, 4], [0.01]))

    # Near-zero matrix
    push!(cases, TestCase("near_zero_8x8", rand_dense_matrix(rng, 8) .* 1e-6, [4], [0.01]))

    return cases
end

# ── main ─────────────────────────────────────────────────────────────

function main()
    cases = generate_cases(rng_test)

    summary = Dict{String, Any}[]
    total = 0
    passed = 0
    failed = 0

    for case in cases
        for s in case.s_values
            for delta in case.delta_values
                total += 1
                test_id = "$(case.name)_s$(s)_d$(Int(delta*100))"
                n = size(case.D, 1)
                print("[$total] $test_id ($(n)x$(n)) ... ")

                # Save input
                input_data = Dict("D" => [case.D[i, :] for i in 1:n], "G" => s, "delta" => delta)
                input_path = joinpath(TEST_DIR, "$(test_id)_input.json")
                open(input_path, "w") do f; JSON.print(f, input_data); end

                # Run SPECTRA
                t_spectra = @elapsed begin
                    P_s, w_s = SPECTRA(case.D)
                    makespan_s, schedule_s = alg3(w_s; s=s, delta=delta)
                end

                # Run WWFA
                t_wwfa = @elapsed begin
                    P_w, w_w = wwfa(case.D)
                end

                # Validate SPECTRA
                errors = String[]
                try
                    for (idx, P) in enumerate(P_s)
                        check_permutation(P, n)
                    end
                catch e
                    push!(errors, "SPECTRA perm invalid: $e")
                end

                ok, msg = check_coverage(case.D, P_s, w_s)
                if !ok; push!(errors, "SPECTRA coverage: $msg"); end

                ok, msg = check_schedule(schedule_s, w_s, s)
                if !ok; push!(errors, "SPECTRA schedule: $msg"); end

                # Validate WWFA
                try
                    for (idx, P) in enumerate(P_w)
                        check_permutation(P, n)
                    end
                catch e
                    push!(errors, "WWFA perm invalid: $e")
                end

                ok, msg = check_coverage(case.D, P_w, w_w)
                if !ok; push!(errors, "WWFA coverage: $msg"); end

                if isempty(errors)
                    passed += 1
                    println("PASS  spectra: k=$(length(w_s)) makespan=$(round(makespan_s, digits=6)) ($(round(t_spectra*1000, digits=1))ms)  wwfa: k=$(length(w_w)) ($(round(t_wwfa*1000, digits=1))ms)")
                else
                    failed += 1
                    println("FAIL")
                    for e in errors; println("  ✗ $e"); end
                end

                # Save reference outputs
                output_data = Dict(
                    "test_id" => test_id,
                    "n" => n,
                    "s" => s,
                    "delta" => delta,
                    "spectra" => Dict(
                        "P" => P_s,
                        "w" => w_s,
                        "makespan" => makespan_s,
                        "schedule" => schedule_s,
                        "k" => length(w_s),
                        "time_s" => t_spectra,
                    ),
                    "wwfa" => Dict(
                        "P" => P_w,
                        "w" => w_w,
                        "k" => length(w_w),
                        "time_s" => t_wwfa,
                    ),
                    "errors" => errors,
                )
                output_path = joinpath(TEST_DIR, "$(test_id)_output.json")
                open(output_path, "w") do f; JSON.print(f, output_data); end

                push!(summary, Dict(
                    "test_id" => test_id, "n" => n, "s" => s, "delta" => delta,
                    "spectra_k" => length(w_s), "spectra_makespan" => makespan_s,
                    "wwfa_k" => length(w_w),
                    "spectra_time" => t_spectra, "wwfa_time" => t_wwfa,
                    "pass" => isempty(errors),
                ))
            end
        end
    end

    # Save summary
    open(joinpath(TEST_DIR, "summary.json"), "w") do f
        JSON.print(f, summary, 2)
    end

    println("\n═══════════════════════════════════")
    println("Total: $total  Passed: $passed  Failed: $failed")
    println("Reference outputs saved to $TEST_DIR")
    println("═══════════════════════════════════")
end

main()
