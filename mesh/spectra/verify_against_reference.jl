#!/usr/bin/env julia
"""
Re-run all test cases and compare against saved reference outputs.
Checks that decomposition coverage and scheduling invariants still hold,
and that makespans have not regressed.
"""

using JSON
include(joinpath(@__DIR__, "spectra.jl"))

const TEST_DIR = joinpath(@__DIR__, "tests")
const MAKESPAN_TOL = 1e-6  # allow tiny floating-point drift
const COVERAGE_TOL = 1e-6

function load_matrix(raw)
    n = length(raw)
    return Matrix(Float64.(hcat(raw...))')
end

function check_permutation(P, n)
    for i in 1:n
        sum(P[i, :]) == 1 || return false, "Row $i sum != 1"
        sum(P[:, i]) == 1 || return false, "Col $i sum != 1"
    end
    for i in 1:n, j in 1:n
        P[i, j] in (0, 1) || return false, "P[$i,$j] not binary"
    end
    return true, "OK"
end

function check_coverage(D, P_list, w_list; tol=COVERAGE_TOL)
    n = size(D, 1)
    R = zeros(n, n)
    for (P, w) in zip(P_list, w_list)
        R .+= w .* P
    end
    for i in 1:n, j in 1:n
        if R[i, j] < D[i, j] - tol
            return false, "[$i,$j]: $(R[i,j]) < $(D[i,j])"
        end
    end
    return true, "OK"
end

function main()
    input_files = filter(f -> endswith(f, "_input.json"), readdir(TEST_DIR))
    sort!(input_files)

    total = 0
    passed = 0
    failed = 0
    regressions = 0

    for input_file in input_files
        test_id = replace(input_file, "_input.json" => "")
        output_file = joinpath(TEST_DIR, "$(test_id)_output.json")

        isfile(output_file) || continue

        total += 1
        ref = JSON.parsefile(output_file)
        inp = JSON.parsefile(joinpath(TEST_DIR, input_file))
        D = load_matrix(inp["D"])
        n = size(D, 1)
        s = inp["G"]
        delta = inp["delta"]

        print("[$total] $test_id ... ")

        errors = String[]

        # Re-run SPECTRA
        P_s, w_s = SPECTRA(D)
        makespan_s, schedule_s = alg3(w_s; s=s, delta=delta)

        # Re-run WWFA
        P_w, w_w = wwfa(D)

        # Check SPECTRA permutations
        for (idx, P) in enumerate(P_s)
            ok, msg = check_permutation(P, n)
            ok || push!(errors, "SPECTRA perm $idx: $msg")
        end

        # Check SPECTRA coverage
        ok, msg = check_coverage(D, P_s, w_s)
        ok || push!(errors, "SPECTRA coverage: $msg")

        # Check WWFA permutations
        for (idx, P) in enumerate(P_w)
            ok, msg = check_permutation(P, n)
            ok || push!(errors, "WWFA perm $idx: $msg")
        end

        # Check WWFA coverage
        ok, msg = check_coverage(D, P_w, w_w)
        ok || push!(errors, "WWFA coverage: $msg")

        # Compare makespan against reference
        ref_makespan = ref["spectra"]["makespan"]
        if makespan_s > ref_makespan + MAKESPAN_TOL
            regressions += 1
            push!(errors, "Makespan regression: $(round(makespan_s, digits=6)) > ref $(round(ref_makespan, digits=6))")
        end

        if isempty(errors)
            passed += 1
            delta_ms = makespan_s - ref_makespan
            sign_str = delta_ms >= 0 ? "+" : ""
            println("PASS  makespan=$(round(makespan_s, digits=6)) ($(sign_str)$(round(delta_ms, digits=6)) vs ref)")
        else
            failed += 1
            println("FAIL")
            for e in errors; println("  ✗ $e"); end
        end
    end

    println("\n═══════════════════════════════════")
    println("Total: $total  Passed: $passed  Failed: $failed  Regressions: $regressions")
    println("═══════════════════════════════════")

    return failed == 0
end

exit(main() ? 0 : 1)
