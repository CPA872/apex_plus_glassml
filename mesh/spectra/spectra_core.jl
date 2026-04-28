#=
spectra_core.jl — Lean wrapper around the core SPECTRA/WWFA/alg3 algorithms.

Replaces PyCall/scipy dependency with Hungarian.jl for the assignment solver,
enabling in-process use from Python via juliacall (PythonCall).

Does NOT modify any algorithm logic — just provides the minimal dependencies.
=#

using DataStructures
using Random, LinearAlgebra, Statistics
using Hungarian: hungarian as _hungarian_jl

# Match the calling convention from spectra.jl:
#   hungarian(cost_matrix) -> (assignment_vector, nothing)
# Hungarian.jl returns (assignment, cost), same shape.
hungarian = D -> (_hungarian_jl(D)[1], nothing)

rng = MersenneTwister(0)

# ── Data structures ──────────────────────────────────────────────────

mutable struct Bin3
    id::Int
    load::Float64
    balls::BinaryMaxHeap{Tuple{Float64, Int}}
end

# ── alg3: Schedule k permutations across s parallel planes ───────────

function alg3(w; s=8, delta=0.01, epsilon=1e-2)
    pq2 = BinaryMinMaxHeap{Tuple{Float64, Int}}()
    bin_map = Dict{Int, Bin3}();

    for i in 1:s
        bin = Bin3(i, 0.0, BinaryMaxHeap{Tuple{Float64, Int}}())
        bin_map[i] = bin
        priority = 0.0
        push!(pq2, (priority, i))
    end

    for (idx, ball) in enumerate(w)
        (min_priority, bin_id) = popmin!(pq2)
        bin = bin_map[bin_id]

        bin.load += delta

        push!(bin.balls, (ball, idx))
        bin.load += ball

        push!(pq2, (bin.load, bin_id))
    end

    while s > 1
        (min_priority, min_bin_id) = popmin!(pq2)
        (max_priority, max_bin_id) = popmax!(pq2)

        min_bin = bin_map[min_bin_id]
        max_bin = bin_map[max_bin_id]

        if max_bin.load - min_bin.load <= delta
            break
        end

        middle = (max_bin.load + min_bin.load + delta) / 2
        a, a_idx = first(max_bin.balls)
        runtime_a1 = max_bin.load - middle
        if a <= runtime_a1
            break
        end

        a, a_idx = pop!(max_bin.balls)
        a = a - runtime_a1
        push!(max_bin.balls, (a, a_idx))
        push!(min_bin.balls, (runtime_a1, a_idx))

        max_bin.load = max_bin.load - runtime_a1
        min_bin.load = min_bin.load + delta + runtime_a1

        push!(pq2, (max_bin.load, max_bin_id))
        push!(pq2, (min_bin.load, min_bin_id))
    end

    max_bin = findmax([(bin.load, bin.id) for bin in values(bin_map)])
    makespan = max_bin[1][1]

    schedule = Dict{Int, Dict{Int, Vector{Float64}}}()

    for i in 1:s
        bin = bin_map[i]
        current_time = 0.0

        while !isempty(bin.balls)
            weight, job_id = pop!(bin.balls)
            start_t = current_time + delta
            end_t = start_t + weight

            if !haskey(schedule, job_id)
                schedule[job_id] = Dict{Int, Vector{Float64}}()
            end
            schedule[job_id][i] = [start_t, end_t]

            current_time = end_t
        end
    end

    return makespan, schedule
end

# ── SPECTRA: Sparsity-aware decomposition ────────────────────────────

function SPECTRA(D::Matrix{Float64}, desc="none", tol=0.0)
    n = size(D, 1)
    D_rem = copy(D)
    P = Matrix{Int16}[]
    w = Float64[]

    init = ifelse.(D .<= tol, 0.0, 0.0)
    init2 = ifelse.(D .<= tol, 1e6, 0.0)
    D_support = ifelse.(D .<= tol, 0, 1)

    max_nnz_row = maximum([count(!iszero, D_support[i, :]) for i in 1:n])
    max_nnz_col = maximum([count(!iszero, D_support[:, j]) for j in 1:n])

    k = max(max_nnz_row, max_nnz_col)

    while true
        should_break = true
        cost_matrix = copy(init)

        cRows = Set{Int}()
        cCols = Set{Int}()

        for i in 1:n
            if count(x -> x > 0, D_support[i, :]) >= k
                push!(cRows, i)
            end
        end

        for j in 1:n
            if count(x -> x > 0, D_support[:, j]) >= k
                push!(cCols, j)
            end
        end

        for i in 1:n, j in 1:n
            if init2[i, j] < 1
                if (i in cRows || j in cCols) && D_support[i, j] == 0
                    cost_matrix[i, j] = 1e9
                elseif D_rem[i, j] > tol
                    cost_matrix[i, j] = -D_rem[i, j]
                    if D_support[i, j] == 1
                        should_break = false
                    end
                else
                    cost_matrix[i, j] = 0.0
                end
            elseif i in cRows || j in cCols
                cost_matrix[i, j] = 1e9
            end
        end

        if should_break
            break
        end

        assignment, _ = hungarian(cost_matrix)

        M = zeros(n, n)
        α = Inf
        for i in 1:n
            j = assignment[i]
            M[i, j] = 1
            D_support[i, j] = 0
            if D_rem[i, j] > tol
                α = min(α, D_rem[i, j])
            end
        end

        D_rem .-= α * M

        push!(P, copy(M))
        push!(w, α)

        k -= 1
    end

    for step in 1:length(w)
        values = [D_rem[I] for I in findall(P[step] .== 1)]

        max_pos = maximum(values)
        if max_pos > 0
            w[step] += max_pos
            D_rem .-= max_pos * P[step]
        end
    end

    perm = sortperm(w, rev=true)
    w_sorted = w[perm]
    P_sorted = P[perm]

    return P_sorted, w_sorted
end

# ── WWFA: Wavefront Arbiter decomposition ────────────────────────────

function wwfa(D::Matrix{Float64}, ε=1e-12, coverage=false, tol=1e-15)
    ε = max(ε,1e-14);
    n = size(D, 1)
    D_rem = copy(D)
    P = Matrix{Int16}[]
    w = Float64[]

    shift_reg = 1

    D_support = nothing
    max_nnz_row = nothing
    max_nnz_col = nothing
    k = nothing
    if coverage
        D_support = ifelse.(D .<= tol, 0, 1)
        max_nnz_row = maximum([count(!iszero, D_support[i, :]) for i in 1:n])
        max_nnz_col = maximum([count(!iszero, D_support[:, j]) for j in 1:n])
        k = max(max_nnz_row, max_nnz_col)
    end

    while(sqrt(sum(D_rem.^2)) > ε)
        if coverage
            if iszero(D_support)
                break
            end
        end

        D_rem2 = copy(D_rem)

        if coverage
            cRows = Set{Int}()
            cCols = Set{Int}()

            for i in 1:n
                if count(x -> x > 0, D_support[i, :]) >= k
                    push!(cRows, i)
                end
            end

            for j in 1:n
                if count(x -> x > 0, D_support[:, j]) >= k
                    push!(cCols, j)
                end
            end

            for i in 1:n
                for j in 1:n
                    if (i in cRows || j in cCols) && D_support[i, j] == 0
                        D_rem2[i, j] = -1e9
                    end
                end
            end
        end

        M = zeros(n,n)
        alpha = 2.0
        rows = Set{Int}()
        cols = Set{Int}()
        total_found = 0
        for i in 1:n
            for j in 1:n
                if j in rows
                    continue
                end
                ind = mod(shift_reg-j, n) + 1
                if ind in cols
                    continue
                end
                if D_rem2[j, ind] <= tol
                    if !coverage
                        D_rem[j, ind] = 0.0
                    end
                else
                    M[j, ind] = 1
                    total_found += 1
                    if D_rem2[j, ind] < alpha
                        alpha = D_rem2[j, ind]
                    end
                    push!(rows, j)
                    push!(cols, ind)
                end
            end

            shift_reg += 1
            if shift_reg > n
                shift_reg -= n
            end
            if total_found == n
                break
            end
        end

        miss_rows = Set{Int}(1:n)
        miss_cols = Set{Int}(1:n)
        setdiff!(miss_rows, rows)
        setdiff!(miss_cols, cols)

        while !isempty(miss_rows)
            r = pop!(miss_rows)
            c = pop!(miss_cols)
            M[r,c] = 1
        end

        D_rem .= max.(D_rem .- alpha .* M, 0)

        push!(w, alpha)
        push!(P, M)

        if coverage
            D_support .*= 1 .- M
            k -= 1
        end
    end

    # Note: refine() requires Gurobi/JuMP — skip for coverage=false (default)
    if coverage
        error("coverage=true requires Gurobi (use spectra.jl directly)")
    end

    perm = sortperm(w, rev=true)
    w_sorted = w[perm]
    P_sorted = P[perm]

    return P_sorted, w_sorted
end
