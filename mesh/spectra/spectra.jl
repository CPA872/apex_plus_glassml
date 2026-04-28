#using BirkhoffDecomposition
using SparseArrays
using DataStructures
using Random, LinearAlgebra, Statistics
using JuMP
#using HiGHS
using Gurobi
# const GRB_ENV = Gurobi.Env()
using BenchmarkTools
using CSV
using DataFrames
using Graphs
#using Hungarian

using JSON
using PyCall
torch = pyimport("torch")
scipy = pyimport("scipy")

hungarian = D -> (scipy.optimize.linear_sum_assignment(D)[2] .+ 1, nothing)

rng = MersenneTwister(0)

function perturb_doubly_stochastic(A::AbstractMatrix{<:Real}; noise_std=0.003, iters=50)
    A = float.(A)
    mask = A .> 0.0

    # Generate noise matrix
    noise = zeros(size(A))
    noise[mask] .= randn(rng, sum(mask)) .* noise_std

    # Scale noise to prevent negative values in B
    abs_noise = abs.(noise[mask])
    scale = min.(1.0, 0.999 * A[mask] ./ (abs_noise .+ 1e-12))
    noise[mask] .= noise[mask] .* scale

    # Center rows and columns iteratively
    m = float.(mask)
    row_num_nonzeros = sum(m, dims=2)
    col_num_nonzeros = sum(m, dims=1)
    for _ in 1:iters
        noise .-= (sum(noise, dims=2) ./ row_num_nonzeros) .* m
        noise .-= (sum(noise, dims=1) ./ col_num_nonzeros) .* m
    end

    return A .+ noise
end

function traffic_matrix(n::Int; nL=4, nS=12, cL=0.7, noise_std=0.003)
    cS = 1 - cL
    A = Matrix{Float64}(I, n, n)

    p1 = zeros(n, n)
    p1_coef = cL / nL
    for _ in 1:nL
        perm = A[shuffle(rng, 1:n), :]
        p1 .+= perm .* p1_coef
    end

    p2 = zeros(n, n)
    p2_coef = cS / nS
    for _ in 1:nS
        perm = A[shuffle(rng, 1:n), :]
        p2 .+= perm .* p2_coef
    end

    D = p1 .+ p2
    if noise_std > 0
        D = perturb_doubly_stochastic(D; noise_std=noise_std)
    end

    return D
end

mutable struct Bin
    id::Int
    load::Float64
    num_balls::Int
end

mutable struct Bin2
    id::Int
    load::Float64
    balls::BinaryMaxHeap{Float64}
end

mutable struct Bin3
    id::Int
    load::Float64
    balls::BinaryMaxHeap{Tuple{Float64, Int}}
end



function alg1(w; s=8, delta=0.01, epsilon=1e-2)
    # P, w = greedy_partial_decomposition(D, epsilon)
    # Custom priority queue
    pq = BinaryMinHeap{Tuple{Float64, Int}}()  # (priority, bin_id)
    bin_map = Dict{Int, Bin}();
    
    for i in 1:s
        bin = Bin(i, 0.0, 0)
        bin_map[i] = bin
        priority = 0.0  # initial load
        push!(pq, (priority, i))
    end

    for ball in w
        (min_priority, bin_id) = pop!(pq)
        bin = bin_map[bin_id]
    
        #if bin.num_balls > 0 # DELTA
        bin.load += delta
        #end
    
        # push!(bin.balls, ball)
        bin.load += ball
        bin.num_balls += 1
    
        # ??? push!(pq, (bin.load + (bin.num_balls > 0 ? delta : 0.0), bin_id))
        push!(pq, (bin.load, bin_id))
    end

    max_bin = findmax([(bin.load, bin.id) for bin in values(bin_map)])
    return max_bin[1][1]
end

function alg2(w; s=8, delta=0.01, epsilon=1e-2)
    # P, w = greedy_partial_decomposition(D, epsilon)
    # Custom priority queue
    pq2 = BinaryMinMaxHeap{Tuple{Float64, Int}}()  # (priority, bin_id)
    bin_map = Dict{Int, Bin2}();
    
    for i in 1:s
        bin = Bin2(i, 0.0, BinaryMaxHeap{Float64}())
        bin_map[i] = bin
        priority = 0.0  # initial load
        push!(pq2, (priority, i))
    end

    for ball in w
        (min_priority, bin_id) = popmin!(pq2)
        bin = bin_map[bin_id]
    
        #if !isempty(bin.balls) # DELTA
        bin.load += delta
        #end
    
        push!(bin.balls, ball)
        bin.load += ball
    
        # ??? push!(pq, (bin.load + (bin.num_balls > 0 ? delta : 0.0), bin_id))
        push!(pq2, (bin.load, bin_id))
    end

    while s > 1
        (min_priority, min_bin_id) = popmin!(pq2)
        (max_priority, max_bin_id) = popmax!(pq2)
    
        min_bin = bin_map[min_bin_id]
        max_bin = bin_map[max_bin_id]
    
        # condition to exit
        if max_bin.load - min_bin.load <= delta
            break
        end
    
        middle = (max_bin.load + min_bin.load + delta) / 2
        a = first(max_bin.balls)
        runtime_a1 = max_bin.load - middle
        if a <= runtime_a1
            break
        end
    
        # a = popfirst!(max_bin.balls) - runtime_a1
        a = pop!(max_bin.balls) - runtime_a1
        push!(max_bin.balls, a)
        push!(min_bin.balls, runtime_a1)
    
        # move a1 to B; --------------
    
        max_bin.load = max_bin.load - runtime_a1
        min_bin.load = min_bin.load + delta + runtime_a1
    
        push!(pq2, (max_bin.load, max_bin_id))
        push!(pq2, (min_bin.load, min_bin_id))
    end

    max_bin = findmax([(bin.load, bin.id) for bin in values(bin_map)])
    return max_bin[1][1]
end

function alg3(w; s=8, delta=0.01, epsilon=1e-2)
    # P, w = greedy_partial_decomposition(D, epsilon)
    # Custom priority queue
    pq2 = BinaryMinMaxHeap{Tuple{Float64, Int}}()  # (priority, bin_id)
    bin_map = Dict{Int, Bin3}();
    
    for i in 1:s
        bin = Bin3(i, 0.0, BinaryMaxHeap{Tuple{Float64, Int}}())
        bin_map[i] = bin
        priority = 0.0  # initial load
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
    
        # condition to exit
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
    
        # move a1 to B; --------------
    
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
        
        # Extract all balls to determine intervals
        # Note: Heap order is roughly largest first.
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

function algMILP(w; s=8, delta=0.01, epsilon=1e-2)
    return 0.0
    alpha = w
    n = length(alpha)        # number of balls
    m = s                    # number of bins

    model = Model(Gurobi.Optimizer)
    set_silent(model)
    set_attribute(model, "TimeLimit", 30)
    # set_attribute(model, "Presolve", 0)

    @variable(model, a[1:n, 1:m] >= 0)       # a(i, k)
    @variable(model, b[1:n, 1:m], Bin)       # b(i, k): binary assignment
    @variable(model, T[1:m] >= 0)            # T(k)
    @variable(model, makespan >= 0)         # Makespan

    # 1. sum a(i, k) == a(i)
    @constraint(model, [i=1:n], sum(a[i, k] for k=1:m) == alpha[i])

    # 2. a(i, k) <= b(i, k)
    @constraint(model, [i=1:n, k=1:m], a[i, k] <= alpha[i] * b[i, k])

    # 3. T(k) = sum a(i, k) + delta * (sum b(i, k) - 1)
    @constraint(model, [k=1:m], T[k] == sum(a[i, k] for i=1:n) + delta * (sum(b[i, k] for i=1:n))) # DELTA

    # 4. Each bin must be assigned at least one ball
    @constraint(model, [k=1:m], sum(b[i, k] for i=1:n) >= 1) 

    # 5. Makespan >= T(k)
    @constraint(model, [k=1:m], makespan >= T[k])

    # Minimize Makespan
    @objective(model, Min, makespan)

    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        #println("Optimal Makespan: ", value(makespan))
        #A = value.(a)
        #B = value.(b)
        #println("Assignment matrix A:\n", A)
        return value(makespan)
    else
        #println("Solver did not find an optimal solution.")
        return value(makespan)
    end
end

function naive(w; s=8, delta=0.01, epsilon=1e-2)
    #Dk = D ./ s
    #P, w = greedy_partial_decomposition(Dk, epsilon / s)
    makespan = sum(w) + (length(w)) * delta # DELTA
    return makespan
end

function compute_D1(D::Matrix{Float64})
    n, m = size(D)
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    set_attribute(model, "TimeLimit", 30)

    # Define decision variables with element-wise upper bounds
    @variable(model, D1[1:n, 1:m] >= 0)
    for i in 1:n, j in 1:m
        @constraint(model, D1[i, j] <= D[i, j])
    end

    # Row sum constraints
    for i in 1:n
        @constraint(model, sum(D1[i, j] for j in 1:m) == 0.5 * sum(D[i, j] for j in 1:m))
    end

    # Column sum constraints
    for j in 1:m
        @constraint(model, sum(D1[i, j] for i in 1:n) == 0.5 * sum(D[i, j] for i in 1:n))
    end

    # Dummy objective (LP is feasibility-based)
    #@objective(model, Min, 0)
    @objective(model, Max, sum(rand() * D1[i,j] for i in 1:n, j in 1:m))

    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        D1_val = value.(D1)
        D2_val = D .- D1_val
        return D1_val, D2_val
    else
        D1_val = zeros(n, m)
        try
            D1_val = value.(D1)
        catch e
            println("Could not 2-way split D matrix")
        end
        D2_val = D .- D1_val
        return D1_val, D2_val
    end
end

"""
    find_alternating_cycle(G, start, visited, path, parent)

Recursive DFS to find a simple cycle in bipartite graph G starting from `start`.
"""
function find_alternating_cycle(G, start, visited, path, parent)
    visited[start] = true
    push!(path, start)

    for neighbor in neighbors(G, start)
        if neighbor == parent
            continue
        elseif neighbor in path
            # Found a cycle
            cycle_start = findfirst(x -> x == neighbor, path)
            return path[cycle_start:end]
        elseif !visited[neighbor]
            result = find_alternating_cycle(G, neighbor, visited, path, start)
            if result !== nothing
                return result
            end
        end
    end

    pop!(path)
    return nothing
end

"""
    improve_D1_via_alternating_cycles(D, D1)

Refines D1 using alternating cycles in the bipartite loose-edge graph.
"""
function improve_D1_via_alternating_cycles(D::Matrix{Float64})
    D1 = D ./ 2
    n = size(D, 1)

    while true
        G = SimpleGraph(2n)
        for i in 1:n, j in 1:n
            if 0 < D1[i, j] < D[i, j]
                add_edge!(G, i, n + j)  # uᵢ to vⱼ
            end
        end

        # Try to find a cycle starting from any node
        found = false
        for start in shuffle(1:2n)
            visited = falses(2n)
            path = Int[]
            cycle = find_alternating_cycle(G, start, visited, path, -1)
            if cycle !== nothing
                found = true
                # Extract edges and compute alternating +/−
                #edges = [(cycle[k], cycle[k+1]) for k in 1:2:length(cycle)-1]
                edges = [(cycle[k], cycle[k+1]) for k in 1:length(cycle)-1]
                push!(edges, (cycle[length(cycle)], cycle[1]))

                # Compute η (max change allowed on all edges)
                etas = Float64[]
                #sign = +1
                flip_sign = rand(Bool) ? 1.0 : -1.0
                sign = flip_sign
                for (u, v) in edges
                    i = u <= n ? u : v
                    j = u > n ? u - n : v - n
                    current = D1[i, j]
                    limit = sign > 0 ? D[i, j] - current : current
                    push!(etas, limit)
                    sign *= -1
                end
                η = minimum(etas)

                # Update D1 along cycle
                sign = flip_sign
                for (u, v) in edges
                    i = u <= n ? u : v
                    j = u > n ? u - n : v - n
                    D1[i, j] += sign * η
                    sign *= -1
                end

                break  # Restart with updated G
            end
        end

        if !found
            break  # No more alternating cycles
        end
    end

    return D1, D .- D1
end


function compute_lessLP(D::Matrix{Float64}, s)
    # Inner recursive function
    function recurse_split(D::Matrix{Float64}, depth::Int)
        if depth >= s
            return [D]
        else
            #D1, D2 = compute_D1(D)
            D1, D2 = improve_D1_via_alternating_cycles(D)
            return vcat(recurse_split(D1, depth * 2),
                        recurse_split(D2, depth * 2))
        end
    end

    return recurse_split(D, 1)
end

function greedy_partial_decomposition(D::Matrix{Float64}, epsilon::Float64=1e-2, tol::Float64=1e-6; max_steps=1000)
    n = size(D, 1)
    D_rem = copy(D)
    #schedule = []
    #P = []
    #w = []
    P = Matrix{Int16}[]
    w = Float64[]

    for step in 1:max_steps
        # Stop if done
        #if maximum(D_rem) < tol
        if sqrt(sum(D_rem.^2)) < epsilon
            break
        end

        # Build the bipartite graph: only use edges with positive demand
        cost_matrix = fill(1e6, n, n)  # effectively block zeros
        for i in 1:n, j in 1:n
            if D_rem[i, j] > tol
                cost_matrix[i, j] = -D_rem[i, j]  # maximize weight
           end
        end

        #cost_matrix = -D_rem

        # Hungarian algorithm gives a full assignment, but we mask out the invalid ones
        assignment, _ = hungarian(cost_matrix)

        # Build partial matching: ignore any assignments from blocked (large cost)
        M = zeros(n, n)
        α = Inf
        for i in 1:n
            j = assignment[i]
            if j != 0 && D_rem[i, j] > tol
                M[i, j] = 1
                α = min(α, D_rem[i, j])
            end
        end

        if α == Inf  # no valid matches
            #println("early break")
            break
        end

        #push!(schedule, (copy(M), α))
        push!(P, copy(M))
        push!(w, α)

        # Subtract α * M from D_rem
        for i in 1:n, j in 1:n
            if M[i, j] == 1
                D_rem[i, j] -= α
            end
        end
    end

    #return schedule
    perm = sortperm(w, rev=true)
    w_sorted = w[perm]
    P_sorted = P[perm]

    return P_sorted, w_sorted
    #return P, w
end

function max_weight_matching(D::Matrix{Float64})
    cost_matrix = -D  # Hungarian solves min-cost, so negate for max-weight
    assignment, _ = hungarian(cost_matrix)
    matches = [(i, j) for (i, j) in enumerate(assignment) if j != 0]
    return matches
end

function bff(D::Matrix{Float64}, δ::Float64 = 0.01, ε::Float64 = 1e-2)
    n = size(D, 1)
    D_rem = copy(D)
    Ia = Set(1:n)  # Available inputs
    Oa = Set(1:n)  # Available outputs
    schedule = []

    # Initial matching
    M = max_weight_matching(D_rem)
    for (i, j) in M
        push!(schedule, (0.0, i, j, D[i, j]))  # (start_time, input, output, duration)
        D_rem[i, j] = 0.0
        delete!(Ia, i)
        delete!(Oa, j)
    end

    # Timeline of events (priority queue: (time, input/output, is_input))
    events = PriorityQueue{Tuple{Float64, Int, Bool}, Float64}()

    for (start, i, j, dur) in schedule
        finish = start + dur
        events[(finish, j, false)] = finish  # output j becomes free
        events[(finish + δ, i, true)] = finish + δ  # input i becomes free after reconfig
    end

    while !isempty(events) && sqrt(sum(D_rem.^2)) > ε
        time, port, is_input = dequeue!(events)
        if is_input
            i = port
            #max_val, j_best = findmax(D_rem[i, :])
            max_val = 0.0
            j_best = 0
            for out_port in Oa
                if D_rem[i, out_port] > max_val
                    max_val = D_rem[i, out_port]
                    j_best = out_port
                end
            end
            
            if max_val > 0
                push!(schedule, (time, i, j_best, max_val))
                D_rem[i, j_best] = 0.0
                delete!(Oa, j_best)
                events[(time + max_val, j_best, false)] = time + max_val
                events[(time + max_val + δ, i, true)] = time + max_val + δ
            else
                push!(Ia, i)
            end
        else
            j = port
            #max_val, i_best = findmax(D_rem[:, j])
            max_val = 0.0
            i_best = 0
            for in_port in Ia
                if D_rem[in_port, j] > max_val
                    max_val = D_rem[in_port, j]
                    i_best = in_port
                end
            end
            
            if max_val > 0
                push!(schedule, (time, i_best, j, max_val))
                D_rem[i_best, j] = 0.0
                delete!(Ia, i_best)
                events[(time + max_val, j, false)] = time + max_val
                events[(time + max_val + δ, i_best, true)] = time + max_val + δ
            else
                push!(Oa, j)
            end
        end

        if isempty(events)
            if is_input
                push!(schedule, (time, port, 0, 0.0))
            else
                push!(schedule, (time, 0, port, 0.0))
            end
        end
    end

    while !isempty(events)
        time, port, is_input = dequeue!(events)
        if is_input
            push!(schedule, (time, port, 0, 0.0))
        else
            push!(schedule, (time, 0, port, 0.0))
        end
    end

    return schedule, D_rem
end

mutable struct Bin_BFF
    id::Int
    load::Float64
    num_balls::Int
    submatrix::Matrix{Float64}
end

mutable struct Bin2_BFF
    id::Int
    load::Float64
    balls::BinaryHeap{Tuple{Float64, Matrix{Int16}}}
end

function alg1_BFF(w, P; s=8, delta=0.01, epsilon=1e-2)
    # P, w = greedy_partial_decomposition(D, epsilon)
    # Custom priority queue
    n = size(P[1], 1)
    pq = BinaryMinHeap{Tuple{Float64, Int}}()  # (priority, bin_id)
    bin_map = Dict{Int, Bin_BFF}();
    
    for i in 1:s
        bin = Bin_BFF(i, 0.0, 0, zeros(Float64, n, n))
        bin_map[i] = bin
        priority = 0.0  # initial load
        push!(pq, (priority, i))
    end

    for (ball, perm) in zip(w, P)
        (min_priority, bin_id) = pop!(pq)
        bin = bin_map[bin_id]
    
        #if bin.num_balls > 0 # DELTA
        bin.load += delta
        #end
    
        # push!(bin.balls, ball)
        bin.load += ball
        bin.num_balls += 1
        bin.submatrix .+= ball .* Float64.(perm)
    
        # ??? push!(pq, (bin.load + (bin.num_balls > 0 ? delta : 0.0), bin_id))
        push!(pq, (bin.load, bin_id))
    end

    makespan = -1.0
    total_count = 0
    for i in 1:s
        makespan = max(bff(bin_map[i].submatrix, delta, epsilon / s)[1][end][1], makespan)
        total_count += count(x -> x > 1e-8, bin_map[i].submatrix)
    end
    return makespan, total_count
end

function alg2_BFF(w, P; s=8, delta=0.01, epsilon=1e-2)
    # P, w = greedy_partial_decomposition(D, epsilon)
    # Custom priority queue
    n = size(P[1], 1)
    pq2 = BinaryMinMaxHeap{Tuple{Float64, Int}}()  # (priority, bin_id)
    bin_map = Dict{Int, Bin2_BFF}();
    
    for i in 1:s
        bin = Bin2_BFF(i, 0.0, BinaryHeap{Tuple{Float64, Matrix{Int16}}}(Base.ReverseOrdering(Base.By(first))))
        bin_map[i] = bin
        priority = 0.0  # initial load
        push!(pq2, (priority, i))
    end

    for (ball, perm) in zip(w, P)
        (min_priority, bin_id) = popmin!(pq2)
        bin = bin_map[bin_id]
    
        #if !isempty(bin.balls) # DELTA
        bin.load += delta
        #end
    
        push!(bin.balls, (ball, perm))
        bin.load += ball
    
        # ??? push!(pq, (bin.load + (bin.num_balls > 0 ? delta : 0.0), bin_id))
        push!(pq2, (bin.load, bin_id))
    end

    while s > 1
        (min_priority, min_bin_id) = popmin!(pq2)
        (max_priority, max_bin_id) = popmax!(pq2)
    
        min_bin = bin_map[min_bin_id]
        max_bin = bin_map[max_bin_id]
    
        # condition to exit
        if max_bin.load - min_bin.load <= delta
            break
        end
    
        middle = (max_bin.load + min_bin.load + delta) / 2
        a, a_perm = first(max_bin.balls)
        runtime_a1 = max_bin.load - middle
        if a <= runtime_a1
            break
        end
    
        #### a = popfirst!(max_bin.balls) - runtime_a1
        a, a_perm = pop!(max_bin.balls)
        a = a - runtime_a1
        #a = pop!(max_bin.balls) - runtime_a1
        push!(max_bin.balls, (a, a_perm))
        push!(min_bin.balls, (runtime_a1, a_perm))
    
        # move a1 to B; --------------
    
        max_bin.load = max_bin.load - runtime_a1
        min_bin.load = min_bin.load + delta + runtime_a1
    
        push!(pq2, (max_bin.load, max_bin_id))
        push!(pq2, (min_bin.load, min_bin_id))
    end
    
    makespan = -1.0
    total_count = 0
    for i in 1:s
        submatrix = zeros(Float64, n, n)
        balls = bin_map[i].balls
        while !isempty(balls)
            ball, perm = pop!(balls)
            submatrix .+= ball .* Float64.(perm)
        end
        makespan = max(bff(submatrix, delta, epsilon / s)[1][end][1], makespan)
        total_count += count(x -> x > 1e-8, submatrix)
    end
    return makespan, total_count
end

function alg1_BFF_deltaless(w, P; s=8, delta=0.01, epsilon=1e-2)
    # P, w = greedy_partial_decomposition(D, epsilon)
    # Custom priority queue
    n = size(P[1], 1)
    pq = BinaryMinHeap{Tuple{Float64, Int}}()  # (priority, bin_id)
    bin_map = Dict{Int, Bin_BFF}();
    
    for i in 1:s
        bin = Bin_BFF(i, 0.0, 0, zeros(Float64, n, n))
        bin_map[i] = bin
        priority = 0.0  # initial load
        push!(pq, (priority, i))
    end

    for (ball, perm) in zip(w, P)
        (min_priority, bin_id) = pop!(pq)
        bin = bin_map[bin_id]
    
        # push!(bin.balls, ball)
        bin.load += ball
        bin.num_balls += 1
        bin.submatrix .+= ball .* Float64.(perm)
    
        # ??? push!(pq, (bin.load + (bin.num_balls > 0 ? delta : 0.0), bin_id))
        push!(pq, (bin.load, bin_id))
    end

    makespan = -1.0
    total_count = 0
    for i in 1:s
        makespan = max(bff(bin_map[i].submatrix, delta, epsilon / s)[1][end][1], makespan)
        total_count += count(x -> x > 1e-8, bin_map[i].submatrix)
    end
    return makespan, total_count
end

function alg2_BFF_deltaless(w, P; s=8, delta=0.01, epsilon=1e-2, tol=1e-6)
    # P, w = greedy_partial_decomposition(D, epsilon)
    # Custom priority queue
    n = size(P[1], 1)
    pq2 = BinaryMinMaxHeap{Tuple{Float64, Int}}()  # (priority, bin_id)
    bin_map = Dict{Int, Bin2_BFF}();
    
    for i in 1:s
        bin = Bin2_BFF(i, 0.0, BinaryHeap{Tuple{Float64, Matrix{Int16}}}(Base.ReverseOrdering(Base.By(first))))
        bin_map[i] = bin
        priority = 0.0  # initial load
        push!(pq2, (priority, i))
    end

    for (ball, perm) in zip(w, P)
        (min_priority, bin_id) = popmin!(pq2)
        bin = bin_map[bin_id]
    
        push!(bin.balls, (ball, perm))
        bin.load += ball
    
        # ??? push!(pq, (bin.load + (bin.num_balls > 0 ? delta : 0.0), bin_id))
        push!(pq2, (bin.load, bin_id))
    end

    while s > 1
        (min_priority, min_bin_id) = popmin!(pq2)
        (max_priority, max_bin_id) = popmax!(pq2)
    
        min_bin = bin_map[min_bin_id]
        max_bin = bin_map[max_bin_id]
    
        # condition to exit
        if max_bin.load - min_bin.load <= tol
            break
        end
    
        middle = (max_bin.load + min_bin.load) / 2
        a, a_perm = first(max_bin.balls)
        runtime_a1 = max_bin.load - middle
        if a <= runtime_a1
            break
        end
    
        #### a = popfirst!(max_bin.balls) - runtime_a1
        a, a_perm = pop!(max_bin.balls)
        a = a - runtime_a1
        #a = pop!(max_bin.balls) - runtime_a1
        push!(max_bin.balls, (a, a_perm))
        push!(min_bin.balls, (runtime_a1, a_perm))
    
        # move a1 to B; --------------
    
        max_bin.load = max_bin.load - runtime_a1
        min_bin.load = min_bin.load + runtime_a1
    
        push!(pq2, (max_bin.load, max_bin_id))
        push!(pq2, (min_bin.load, min_bin_id))
    end

    makespan = -1.0
    total_count = 0
    for i in 1:s
        submatrix = zeros(Float64, n, n)
        balls = bin_map[i].balls
        while !isempty(balls)
            ball, perm = pop!(balls)
            submatrix .+= ball .* Float64.(perm)
        end
        makespan = max(bff(submatrix, delta, epsilon / s)[1][end][1], makespan)
        total_count += count(x -> x > 1e-8, submatrix)
    end
    return makespan, total_count
end

function isaac1(D::Matrix{Float64}, desc="none")
    n = size(D, 1)
    D_rem = copy(D)
    D_accum = zeros(n,n)
    P = Matrix{Int16}[]
    w = Float64[]

    while true
        should_break = true
        cost_matrix = zeros(n, n)
        for i in 1:n, j in 1:n
            if D_rem[i, j] > 0
                cost_matrix[i, j] = -D_rem[i, j]  # maximize weight
                should_break = false
           end
        end

        if should_break
            break
        end

        #cost_matrix = -D_rem

        # Hungarian algorithm gives a full assignment, but we mask out the invalid ones
        assignment, _ = hungarian(cost_matrix)

        M = zeros(n, n)
        α = -1
        for i in 1:n
            j = assignment[i]
            #if j != 0 && D_rem[i, j] > 0
            M[i, j] = 1
            α = max(α, D_rem[i, j])
            D_rem[i,j] = 0.0
            #end
        end

        #push!(schedule, (copy(M), α))
        push!(P, copy(M))
        push!(w, α)

        D_accum .+= α * M
    end

    D_accum .-= D    

    if desc != "none"
        if desc == "false"
            perm = sortperm(w, rev=false)
            w = w[perm]
            P = P[perm]
        else
            perm = sortperm(w, rev=true)
            w = w[perm]
            P = P[perm]
        end
    end

    #println(w)
    
    for step in 1:length(w)
        values = [D_accum[I] for I in findall(P[step] .== 1)] 

        min_pos = minimum(values)
        if min_pos > 0
            w[step] -= min_pos
            D_accum .-= min_pos * P[step]
        end

    end

    #return schedule
    perm = sortperm(w, rev=true)
    w_sorted = w[perm]
    P_sorted = P[perm]

    return P_sorted, w_sorted
    #return P, w
end

function SPECTRA(D::Matrix{Float64}, desc="none", tol=0.0)
    n = size(D, 1)
    D_rem = copy(D)
    #D_accum = zeros(n,n)
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
            #if D_rem[i, j] > 0
            if init2[i, j] < 1
                # cost_matrix[i, j] = -D_rem[i, j]  # maximize weight
                
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

        #cost_matrix = -D_rem

        # Hungarian algorithm gives a full assignment, but we mask out the invalid ones
        assignment, _ = hungarian(cost_matrix)

        M = zeros(n, n)
        α = Inf
        for i in 1:n
            j = assignment[i]
            #if j != 0 && D_rem[i, j] > 0
            M[i, j] = 1
            D_support[i, j] = 0
            if D_rem[i, j] > tol
                α = min(α, D_rem[i, j])
            end
            #D_rem[i,j] = 0.0
            #end
        end

        D_rem .-= α * M

        #push!(schedule, (copy(M), α))
        push!(P, copy(M))
        push!(w, α)

        #D_accum .+= α * M
        k -= 1
    end

    #D_accum .-= D    

    #println(w)
    
    for step in 1:length(w)
        values = [D_rem[I] for I in findall(P[step] .== 1)] 

        max_pos = maximum(values)
        if max_pos > 0
            w[step] += max_pos
            D_rem .-= max_pos * P[step]
        end

    end

    #return schedule
    perm = sortperm(w, rev=true)
    w_sorted = w[perm]
    P_sorted = P[perm]

    return P_sorted, w_sorted
    #return P, w
end

function SPECTRA_MILP(D::Matrix{Float64}, desc="none", tol=0.0)
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
            #if D_rem[i, j] > 0
            if init2[i, j] < 1
                # cost_matrix[i, j] = -D_rem[i, j]  # maximize weight
                
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

        #cost_matrix = -D_rem

        # Hungarian algorithm gives a full assignment, but we mask out the invalid ones
        assignment, _ = hungarian(cost_matrix)

        M = zeros(n, n)
        α = Inf
        for i in 1:n
            j = assignment[i]
            #if j != 0 && D_rem[i, j] > 0
            M[i, j] = 1
            D_support[i, j] = 0
            if D_rem[i, j] > tol
                α = min(α, D_rem[i, j])
            end
            #D_rem[i,j] = 0.0
            #end
        end

        D_rem .-= α * M

        #push!(schedule, (copy(M), α))
        push!(P, copy(M))
        push!(w, α)

        k -= 1
    end

    k = length(P)

    model = Model(Gurobi.Optimizer)
    set_silent(model)
    set_attribute(model, "TimeLimit", 30)
    # set_attribute(model, "Presolve", 0)

    # Variables: α[i] ∈ [0, 1]
    @variable(model, 0 <= α[1:k])

    # Constraints: elementwise sum α[i] * P[i] ≥ D
    for i in 1:n, j in 1:n
        @constraint(model, sum(α[l] * P[l][i, j] for l in 1:k) >= D[i, j])
    end

    # Objective: minimize total alpha sum
    @objective(model, Min, sum(α))

    optimize!(model)

    w = [value(α[i]) for i in 1:length(α)]
    perm = sortperm(w, rev=true)
    w_sorted = w[perm]
    P_sorted = P[perm]

    if termination_status(model) == MOI.OPTIMAL
        return P_sorted, w_sorted
    else
        return P_sorted, w_sorted
    end
end

function less_pct(Ds::Vector{Matrix{Float64}}, t::Float64, D::Matrix{Float64}, delta::Float64, decomp)
    n = size(D, 1)
    #d = [decomp(Dsplit) for Dsplit in Ds]
    #makespan_LESS_ = maximum([sum(wsplit) + delta*(length(wsplit)-1) for wsplit in wLESS_])
    D_accum = zeros(n, n)
    #ones = ifelse.(D .<= 0.0, 0, 1)
    #target = t
    
    for Dsplit in Ds
        P, w = decomp(Dsplit)
        target = t
        for i in 1:length(w) # DELTA
            if delta >= target
                break
            end
            target -= delta

            if w[i] >= target
                D_accum .+= target * P[i]
                break
            end
            target -= w[i]
            D_accum .+= w[i] * P[i]
            
        end
    end

    for i in 1:n
        for j in 1:n
            if D[i,j] <= 0.0
                D_accum[i,j] = 0.0
            else
                #D_accum[i,j] = min(D_accum[i,j], D[i,j])
                D_accum[i,j] = min(D_accum[i,j], D[i,j])
            end
            
        end
    end
    
    return 100 * sum(D_accum) / sum(D)
    #return log(sum(D_accum)) - log(sum(D))
    #return sum(D_accum) / sum(D)
    #return sum(D_accum), sum(D)
end

function less_bff_pct(Ds::Vector{Matrix{Float64}}, t::Float64, D2::Matrix{Float64}, δ::Float64, decomp)
    n = size(D2, 1)
    #makespan_LESS_ = maximum([sum(wsplit) + delta*(length(wsplit)-1) for wsplit in wLESS_])
    D_accum = zeros(n, n)
    #ones = ifelse.(D .<= 0.0, 0, 1)
    #target = t

    for D in Ds
        D_rem = copy(D)
        Ia = Set(1:n)  # Available inputs
        Oa = Set(1:n)  # Available outputs
        schedule = []
    
        # Initial matching
        M = max_weight_matching(D_rem)
        for (i, j) in M
            push!(schedule, (0.0, i, j, D[i, j]))  # (start_time, input, output, duration)
            if D[i, j] + δ <= t # DELTA
                D_accum[i, j] += D[i, j]
            elseif t - δ > 0
                D_accum[i, j] += t - δ
            end
            D_rem[i, j] = 0.0
            delete!(Ia, i)
            delete!(Oa, j)
        end
    
        # Timeline of events (priority queue: (time, input/output, is_input))
        events = PriorityQueue{Tuple{Float64, Int, Bool}, Float64}()
    
        for (start, i, j, dur) in schedule
            finish = start + dur
            events[(finish, j, false)] = finish  # output j becomes free
            events[(finish + δ, i, true)] = finish + δ  # input i becomes free after reconfig
        end
    
        while !isempty(events) && sqrt(sum(D_rem.^2)) > 0.0
            time, port, is_input = dequeue!(events)
            if is_input
                i = port
                #max_val, j_best = findmax(D_rem[i, :])
                max_val = 0.0
                j_best = 0
                for out_port in Oa
                    if D_rem[i, out_port] > max_val
                        max_val = D_rem[i, out_port]
                        j_best = out_port
                    end
                end
                
                if max_val > 0
                    push!(schedule, (time, i, j_best, max_val))
                    if time + max_val + δ <= t # DELTA
                        D_accum[i, j_best] += max_val
                    elseif t - time - δ > 0
                        D_accum[i, j_best] += t - time - δ
                    end
                    D_rem[i, j_best] = 0.0
                    delete!(Oa, j_best)
                    events[(time + max_val, j_best, false)] = time + max_val
                    events[(time + max_val + δ, i, true)] = time + max_val + δ
                else
                    push!(Ia, i)
                end
            else
                j = port
                #max_val, i_best = findmax(D_rem[:, j])
                max_val = 0.0
                i_best = 0
                for in_port in Ia
                    if D_rem[in_port, j] > max_val
                        max_val = D_rem[in_port, j]
                        i_best = in_port
                    end
                end
                
                if max_val > 0
                    push!(schedule, (time, i_best, j, max_val))
                    if time + max_val + δ <= t # DELTA
                        D_accum[i_best, j] += max_val
                    elseif t - time - δ > 0
                        D_accum[i_best, j] += t - time - δ
                    end
                    D_rem[i_best, j] = 0.0
                    delete!(Ia, i_best)
                    events[(time + max_val, j, false)] = time + max_val
                    events[(time + max_val + δ, i_best, true)] = time + max_val + δ
                else
                    push!(Oa, j)
                end
            end
        end
    
        
    end

    
    return 100 * sum(D_accum) / sum(D2)
    #return log(sum(D_accum)) - log(sum(D))
    #return sum(D_accum) / sum(D)
    #return sum(D_accum), sum(D)
end

function getBirkhoffStepSize(x_star,x,y)
    return minimum((x_star.-x).*y .- (y.-1));
    #return minimum((x_star.-x).*y .- (y.-1) .- ((x_star .> 0.0) .- 1));
end

function getBirkhoffStepSize2(x_star,x,y)
    # BROKEN CODE
    #return minimum((x_star.-x).*y .- (y.-1) .- ((x_star .> 0.0) .- 1));


    #Possible fix (though still susceptible if selected perm is entirely 0)
    return minimum((x_star.-x).*y .- (y.-1) .- (((x_star.-x) .> 0.0) .- 1));
end

struct polytope
    A;
    b;
    l;
    u;
    model;
    x;
end

function LP(c)

    #@objective(P.model, Min, c'* P.x)
    #optimize!(P.model)
    #return value.(P.x)

    d = size(c, 1)
    n = Int(sqrt(d))
    weight_matrix = reshape(c, n, n)
    assignment, _ = hungarian(weight_matrix)
    M = zeros(n,n)
    for i in 1:n
        j = assignment[i]
        M[i, j] = 1
    end

    return reshape(M, d)

end

function randomDoublyStochasticMatrix(n; num_perm = n^2)

    M = zeros(n,n)
    α = rand(num_perm)
    α = α / sum(α)

    for i=1:num_perm
        perm = randperm(n)
        for j=1:n
            M[perm[j],j] += α[i]
        end
    end

    return M
end

function randomPermutations(n; num_perm = n^2)

    M = zeros(n,n)
    α = rand(num_perm)
    permutations = zeros(n,num_perm);

    α = α / sum(α)

    for i=1:num_perm
        perm = randperm(n)
        permutations[:,i] = perm
        for j=1:n
            M[perm[j],j] += α[i]
        end
    end

    return M, permutations, α
end

function birkhoffPolytope(n)

    M = zeros(n*n,2*n);
    for i=1:n
        M[(i-1)*n*n + (i-1)*n + 1 : (i-1)*n*n + (i-1)*n + n ] = ones(n,1);
    end
    for i=1:n
        for j=1:n
            M[n*n*n + (i-1)*n*n + (j-1)*n + i] = 1;
        end
    end

    A = M';
    b = ones(2*n);

    model = Model(Gurobi.Optimizer)
    #set_optimizer_attribute(model, "solver", "simplex")
    # set_optimizer_attribute(model, "output_flag", false)
    set_silent(model)
    @variable(model, 0 <= x[1:n*n] <= 1)
    @constraint(model, A * x .== b)

    return polytope(A,b,0,1,model,x)

end

function getEPBplus(x_star, x, max_rep, ε, α_init)

    n = sqrt(size(x_star,1))
    d  = size(x_star,1);
    i = 1;
    y = 0;
    α = 0;
    α = α_init

    while(i <= max_rep)

        z = Int16.(x_star - x .> ε)
        s = getBirkhoffStepSize(x_star,x,z)
        beta = (s + ε/d)*0.5

        c = -ones(d) + beta ./ (x_star - x .+ ε/d)
        b = ((n/ε)*1000).*Int16.(x_star - x .<= α)
        y_iter = LP(c + b);
        y_z = x;
        if(c'*y_iter > c'*y_z)
            #println("HEY")
            #y_iter = y_z
        end

        α_iter = getBirkhoffStepSize(x_star, x, y_iter);

        if(α < α_iter)# && (y_iter'*(x_star.> 0) >= sqrt(d)))
            α = α_iter;
            y = y_iter;
        else
            if i == 1
                return y_iter;
            else
                return y;
            end
        end
        i = i + 1;
    end

    return y
end

function randomDoublySubStochasticMatrix(n; num_perm = n^2)
    num_perm = rand(1:num_perm)
    M = zeros(n,n)
    α = rand(num_perm)
    α = α / sum(α)

    for i=1:num_perm
        perm = randperm(n)
        randSubset = randperm(n)
        randSubsetSize = rand(0:n)
        for j in randSubset[1:randSubsetSize]
            M[perm[j],j] += α[i]
        end
    end

    return M
end

function vonN(D)
    Dnew = copy(D)
    n = size(D, 1)

    rows = [sum(D[i, :]) for i in 1:n]
    cols = [sum(D[:, i]) for i in 1:n]

    subRows = Set((i, rows[i]) for i in 1:n if rows[i] < 1)
    subCols = Set((i, cols[i]) for i in 1:n if cols[i] < 1)

    i = 1
    while !isempty(subRows) && !isempty(subCols) 
        r, rSum = pop!(subRows)
        c, cSum = pop!(subCols)
        
        addp = 1
        if rSum > cSum
            addp = 1 - rSum
            push!(subCols, (c, cSum + addp))
        else
            addp = 1 - cSum
            push!(subRows, (r, rSum + addp))
        end
        Dnew[r, c] += addp
        
    end

    return Dnew
end

function expandStochastic(D)
    n = size(D, 1)
    diff = n - sum(D)
    if diff - floor(diff) < 1e-13
        diff = floor(diff)
    end
    subd = Int(ceil(diff))
    N = n + subd
    Dnew = fill(-1.0, (N, N))
    Dnew[1:n, 1:n] = copy(D)

    col = n + 1
    row = 1
    while col <= N && row <= n
        firstS = 1 - sum(Dnew[row, 1:col-1])
        secondS = 1 - sum(Dnew[1:row-1, col])

        if firstS < secondS
            Dnew[row, col] = firstS
            Dnew[row, 1+col:N] .= 0.0
            row += 1
        elseif secondS < firstS
            Dnew[row, col] = secondS
            Dnew[1+row:N, col] .= 0.0
            col += 1
        else
            Dnew[row, col] = firstS
            Dnew[row, 1+col:N] .= 0.0
            Dnew[1+row:N, col] .= 0.0
            row += 1
            col += 1
        end
    end

    # repeat of first block but transposed
    row = n + 1
    col = 1
    while row <= N && col <= n
        secondS = 1 - sum(Dnew[row, 1:col-1])
        firstS = 1 - sum(Dnew[1:row-1, col])

        if firstS < secondS
            Dnew[row, col] = firstS
            Dnew[1+row:N, col] .= 0.0   
            col += 1
        elseif secondS < firstS
            Dnew[row, col] = secondS
            Dnew[row, 1+col:N] .= 0.0
            row += 1
        else
            Dnew[row, col] = firstS
            Dnew[row, 1+col:N] .= 0.0
            Dnew[1+row:N, col] .= 0.0
            row += 1
            col += 1
        end
    end

    for r in 1+n:N
        for c in 1+n:N
            if Dnew[r,c] == 0.0
                continue
            end
            Dnew[r,c] = min(1-sum(Dnew[1:r-1, c]), 1-sum(Dnew[r, 1:c-1]))
        end
    end

    return Dnew
end

"""
function getBirkhoffStepSize1(x_star,x,y,z=-1.0)
    return minimum((x_star.-x).*y .- (y.-1) .- ((x_star .>= z) .- 1));
end

function getEPBplus1(x_star, x, max_rep, ε, α_init)

    n = sqrt(size(x_star,1))
    d  = size(x_star,1);
    i = 1;
    y = 0;
    α = 0;
    α = α_init

    while(i <= max_rep)

        z = Int16.(x_star - x .> ε)
        s = getBirkhoffStepSize(x_star,x,z)
        beta = (s + ε/d)*0.5

        c = -ones(d) + beta ./ (x_star - x .+ ε/d)
        b = (n/ε).*Int16.(x_star - x .<= α)
        y_iter = LP(c + b);
        y_z = x;
        if(c'*y_iter > c'*y_z)
            y_iter = y_z
        end

        α_iter = getBirkhoffStepSize1(x_star, x, y_iter, α);

        if(α < α_iter)# && (y_iter'*(x_star.> 0) >= sqrt(d)))
            α = α_iter;
            y = y_iter;
        else
            return y;
        end
        i = i + 1;
    end

    return y
end
"""

function getEPBplusL2(x_star, x, max_rep, ε, α_init)

    n = sqrt(size(x_star,1))
    d  = size(x_star,1);
    i = 1;
    y = 0;
    α = 0;
    α = α_init

    while(i <= max_rep)

        z = Int16.(x_star - x .> ε)
        s = getBirkhoffStepSize(x_star,x,z)
        beta = (s + ε/d)*0.5

        #c = -ones(d) + beta ./ (x_star - x .+ ε/d)
        c = x .- x_star
        b = ((n/ε)*1000).*Int16.(x_star - x .<= α)
        y_iter = LP(c + b);
        y_z = x;
        if(c'*y_iter > c'*y_z)
            #println("HEY")
            #y_iter = y_z
        end

        α_iter = getBirkhoffStepSize(x_star, x, y_iter);

        if(α < α_iter)# && (y_iter'*(x_star.> 0) >= sqrt(d)))
            α = α_iter;
            y = y_iter;
        else
            if i == 1
                return y_iter;
            else
                return y;
            end
        end
        i = i + 1;
    end

    return y
end

function refine(X, P, w, ε=1e-12)
    k = length(w)
    n = size(X,1)

    coverage = zeros(n,n)
    for p in P
        coverage .+= p
    end
    D = X .* (coverage .> 0.0)
    
    model = Model(Gurobi.Optimizer)
    #set_silent(model)
    set_silent(model)
    set_attribute(model, "TimeLimit", 30)
    # set_attribute(model, "Presolve", 0)
    
    # Variables: α[i] ∈ [0, 1]
    @variable(model, 0 <= α[1:k] <= 1)
    
    # Constraints: elementwise sum α[i] * P[i] ≥ D
    for i in 1:n, j in 1:n
        @constraint(model, sum(α[l] * P[l][i, j] for l in 1:k) >= D[i, j])
    end
    
    # Objective: minimize total alpha sum
    @objective(model, Min, sum(α))
    
    optimize!(model)
    
    w = [value(α[i]) for i in 1:length(α)]


    delInds = []    
    for i in 1:length(w)
        if w[i] < 1e-12
            push!(delInds, i)
        end
    end
    
    deleteat!(w, delInds)
    deleteat!(P, delInds)
    
    #perm = sortperm(w, rev=true)
    w_sorted = w#[perm]
    P_sorted = P#[perm]
    
    if termination_status(model) == MOI.OPTIMAL
        return P_sorted, w_sorted
    else
        return P_sorted, w_sorted
    end
end

function birkdecomp(X, ε=1e-12, L2=false, coverage=false; max_rep = 1)

    n = size(X,1);                                  # get size of Birkhoff polytope
    x_star = reshape(X, n*n);                       # reshape doubly stochastic to vector
    #B = birkhoffPolytope(n);                        # Birkhoff polytope
    ε = max(ε,1e-14);                               # fix the maximum minimum precision
    max_iter = (n-1)^2 + 1;

    x = zeros(n*n);                                 # initial point

    extreme_points = zeros(n*n,max_iter);           # extreme points matrix
    θ = zeros(max_iter);                            # weights vector
    approx = Inf;                                   # approximation error
    i = 1;                                          # iteration index

    log_approx = zeros(max_iter)

    support = x_star .> 0.0
    # Algorithms
    while(approx > ε)# && !iszero(support))
        if (coverage && iszero(support))
            break
        end
        α_init = (1 - sum(θ[1:i-1])) / (n*n)

        y = nothing
        if L2
            y = getEPBplusL2(x_star, x, max_rep, ε, α_init)
        else
            y = getEPBplus(x_star, x, max_rep, ε, α_init)
        end
        #y = getEPBplus(x_star, x, max_rep, ε, α_init)
        θi = getBirkhoffStepSize(x_star,x,y)
        x = x .+ θi.*y;
        θ[i] = θi;

        approx = sqrt(sum((abs.(x_star.-x)).^2));
        log_approx[i] = approx
        extreme_points[:,i] = abs.(y);

        support .*= 1 .- abs.(y)
        
        i = i + 1;
        
    end

    α = θ[1:i-1]
    P = [reshape(extreme_points[:, j], n, n) for j in 1:i-1]

    if coverage
        return refine(X, P, α, ε)
    end
    
    return P, α
end

function birkdecomp1(X, ε=1e-12, L2=false, coverage=false; max_rep = 1)

    n = size(X,1);                                  # get size of Birkhoff polytope
    x_star = reshape(X, n*n);                       # reshape doubly stochastic to vector
    #B = birkhoffPolytope(n);                        # Birkhoff polytope
    ε = max(ε,1e-14);                               # fix the maximum minimum precision
    #max_iter = (n-1)^2 + 1;
    max_iter = n^2;

    x = zeros(n*n);                                 # initial point

    extreme_points = zeros(n*n,max_iter);           # extreme points matrix
    θ = zeros(max_iter);                            # weights vector
    approx = Inf;                                   # approximation error
    i = 1;                                          # iteration index

    log_approx = zeros(max_iter)

    support = x_star .> 0.0
    # Algorithms
    while(approx > ε)# && !iszero(support))
        if (coverage && iszero(support))
            break
        end
        α_init = sum(x_star.-x) / (n*n)
        #α_init = sum(x_star.-x) / (n^4)
        #α_init = minimum((x_star.-x) .- (((x_star.-x).> 0.0) .- 1)) / 200.0

        y = nothing
        if L2
            y = getEPBplusL2(x_star, x, max_rep, ε, α_init)
        else
            y = getEPBplus(x_star, x, max_rep, ε, α_init)
        end
        #y = getEPBplus(x_star, x, max_rep, ε, α_init)
        y = y.*(x_star .- x .>= α_init)
        θi = getBirkhoffStepSize(x_star,x,y)
        #println(iszero(y))
        #println(θi)
        x = x .+ θi.*y;
        θ[i] = θi;

        approx = sqrt(sum((abs.(x_star.-x)).^2));
        log_approx[i] = approx
        extreme_points[:,i] = abs.(y);

        support .*= 1 .- abs.(y)
        
        i = i + 1;
    end

    α = θ[1:i-1]
    P = [reshape(extreme_points[:, j], n, n) for j in 1:i-1]

    if coverage
        return refine(X, P, α, ε)
    end
    
    return P, α
end

function birkdecomp2(X, ε=1e-12, L2=false, coverage=false; max_rep = 1)
    D2 = expandStochastic(X)
    P_,w_ = birkdecomp(D2, ε, L2; max_rep = max_rep)

    n = size(X, 1)
    P = Matrix{Float64}[]
    w = Float64[]

    for i in 1:length(w_)
        p = P_[i][1:n, 1:n]
        if !iszero(p)
            push!(P, p)
            push!(w, w_[i])
        end
    end

    if coverage
        support = X .> 0.0
        i = 0
        for p in P
            if iszero(support)
                break
            end
            support .*= 1 .- p
            
            i += 1
        end
        return refine(X, P[1:i], w[1:i], ε)
    end
    
    return P,w
end

function birkdecomp3(X, ε=1e-12, L2=false, coverage=false; max_rep = 1)
    D2 = vonN(X)
    P,w_ = birkdecomp(D2, ε, L2, coverage; max_rep = max_rep)

    return refine(X, P, w_, ε)
end

function birkdecomp4(X, ε=1e-12, L2=false, coverage=false; max_rep = 1)
    D2 = vonN(X)
    P,w = birkdecomp(D2, ε, L2, coverage; max_rep = max_rep)
    return P,w
end

function birkdecomp5(X, ε=1e-12, L2=false, coverage=false; max_rep = 1)

    n = size(X,1);                                  # get size of Birkhoff polytope
    x_star = reshape(X, n*n);                       # reshape doubly stochastic to vector
    #B = birkhoffPolytope(n);                        # Birkhoff polytope
    ε = max(ε,1e-14);                               # fix the maximum minimum precision
    #max_iter = (n-1)^2 + 1;
    max_iter = n^2;

    x = zeros(n*n);                                 # initial point

    extreme_points = zeros(n*n,max_iter);           # extreme points matrix
    θ = zeros(max_iter);                            # weights vector
    approx = Inf;                                   # approximation error
    i = 1;                                          # iteration index

    log_approx = zeros(max_iter)

    support = x_star .> 0.0
    # Algorithms
    while(approx > ε)# && !iszero(support))
        if (coverage && iszero(support))
            break
        end
        α_init = sum(x_star.-x) / (1e12)
        #α_init = sum(x_star.-x) / (n^4)
        #α_init = minimum((x_star.-x) .- (((x_star.-x).> 0.0) .- 1)) / 200.0

        y = nothing
        if L2
            y = getEPBplusL2(x_star, x, max_rep, ε, α_init)
        else
            y = getEPBplus(x_star, x, max_rep, ε, α_init)
        end
        #y = getEPBplus(x_star, x, max_rep, ε, α_init)
        y = y.*(x_star .- x .>= α_init)
        θi = getBirkhoffStepSize(x_star,x,y)
        x = x .+ θi.*y;
        θ[i] = θi;

        approx = sqrt(sum((abs.(x_star.-x)).^2));
        log_approx[i] = approx
        extreme_points[:,i] = abs.(y);

        support .*= 1 .- abs.(y)
        
        i = i + 1;
        
    end

    α = θ[1:i-1]
    P = [reshape(extreme_points[:, j], n, n) for j in 1:i-1]

    if coverage
        return refine(X, P, α, ε)
    end
    
    return P, α
end

function birkdecomp6(X, ε=1e-12, L2=false, coverage=false; max_rep = 1)

    n = size(X,1);                                  # get size of Birkhoff polytope
    x_star = reshape(X, n*n);                       # reshape doubly stochastic to vector
    #B = birkhoffPolytope(n);                        # Birkhoff polytope
    ε = max(ε,1e-14);                               # fix the maximum minimum precision
    #max_iter = (n-1)^2 + 1;
    max_iter = n^2;

    x = zeros(n*n);                                 # initial point

    extreme_points = zeros(n*n,max_iter);           # extreme points matrix
    θ = zeros(max_iter);                            # weights vector
    approx = Inf;                                   # approximation error
    i = 1;                                          # iteration index

    log_approx = zeros(max_iter)

    support = x_star .> 0.0
    # Algorithms
    while(approx > ε)# && !iszero(support))
        if (coverage && iszero(support))
            break
        end
        α_init = sum(x_star.-x) / (n*n)
        #α_init = sum(x_star.-x) / (n^4)
        #α_init = minimum((x_star.-x) .- (((x_star.-x).> 0.0) .- 1)) / 200.0

        y = nothing
        if L2
            y = getEPBplusL2(x_star, x, max_rep, ε, α_init)
        else
            y = getEPBplus(x_star, x, max_rep, ε, α_init)
        end
        #y = getEPBplus(x_star, x, max_rep, ε, α_init)
        y2 = y.*(x_star .- x .< α_init)
        y = y.*(x_star .- x .>= α_init)
        θi = getBirkhoffStepSize(x_star,x,y)
        #println(iszero(y))
        #println(θi)
        x = x .+ θi.*y .+ ((x_star.-x).*y2);
        θ[i] = θi;

        y = y .+ y2

        approx = sqrt(sum((abs.(x_star.-x)).^2));
        log_approx[i] = approx
        extreme_points[:,i] = abs.(y);

        support .*= 1 .- abs.(y)
        
        i = i + 1;
        
    end

    α = θ[1:i-1]
    P = [reshape(extreme_points[:, j], n, n) for j in 1:i-1]

    if coverage
        return refine(X, P, α, ε)
    end
    
    return P, α
end

function birkdecomp7(X, ε=1e-12, L2=false, coverage=false; max_rep = 1)
    D2 = vonN(X)
    
    n = size(X,1);                                  # get size of Birkhoff polytope
    x_star = reshape(X, n*n);                       # reshape doubly stochastic to vector
    shadow_star = reshape(D2, n*n);
    #B = birkhoffPolytope(n);                        # Birkhoff polytope
    ε = max(ε,1e-14);                               # fix the maximum minimum precision
    #max_iter = (n-1)^2 + 1;
    max_iter = n^2;

    x = zeros(n*n);                                 # initial point
    shadow = zeros(n*n);

    extreme_points = zeros(n*n,max_iter);           # extreme points matrix
    θ = zeros(max_iter);                            # weights vector
    shadowθ = [0.0];
    approx = Inf;                                   # approximation error
    i = 1;                                          # iteration index

    log_approx = zeros(max_iter)

    support = x_star .> 0.0
    # Algorithms
    while(approx > ε)# && !iszero(support))
        if (coverage && iszero(support))
            break
        end
        #α_init = (1 - sum(θ[1:i-1])) / (n*n)
        α_init = (1 - sum(shadowθ)) / (n*n)

        y = nothing
        if L2
            #y = getEPBplusL2(x_star, x, max_rep, ε, α_init)
            y = getEPBplusL2(shadow_star, shadow, max_rep, ε, α_init)
        else
            #y = getEPBplus(x_star, x, max_rep, ε, α_init)
            y = getEPBplus(shadow_star, shadow, max_rep, ε, α_init)
        end
        #y = getEPBplus(x_star, x, max_rep, ε, α_init)
        if iszero((x_star.-x).*y)
            θi = getBirkhoffStepSize(shadow_star,shadow,y)
            #x = x .+ θi.*y;
            shadow = shadow .+ θi.*y;
            #θ[i] = θi;
            push!(shadowθ, θi)
            continue
        end
 
        θi1 = minimum((x_star.-x).*y .- (y.-1) .- (((x_star.-x) .> 0.0) .- 1))
        θi2 = getBirkhoffStepSize(shadow_star,shadow,y)
        θi = min(θi1, θi2)

        shadow = shadow .+ θi.*y;
        push!(shadowθ, θi)

        #y = y.*(x_star .- x .>= α_init)
        x = x .+ θi.*y.*(x_star .- x .> 0.0);
        θ[i] = θi;

        approx = sqrt(sum((abs.(x_star.-x)).^2));
        log_approx[i] = approx
        extreme_points[:,i] = abs.(y);

        support .*= 1 .- abs.(y)
        
        i = i + 1;
        
    end

    α = θ[1:i-1]
    P = [reshape(extreme_points[:, j], n, n) for j in 1:i-1]

    if coverage
        return refine(X, P, α, ε)
    end
    
    return P, α
end


function greedyMWM(D::Matrix{Float64}, ε=1e-12, coverage=false, tol=0.0)
    ε = max(ε,1e-14);
    n = size(D, 1)
    D_rem = copy(D)
    P = Matrix{Int16}[]
    w = Float64[]

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
        for i in 1:n
            m, index = findmax(D_rem2)
            row, col = Tuple(index)
            if m <= tol
                D_rem2[row,col]=0.0
            elseif m < alpha
                alpha = m
            end
            M[row, col] = 1
            D_rem2[row, :] .= -1.0
            D_rem2[:, col] .= -1.0
        end

        D_rem .= max.(D_rem .- alpha .* M, 0)

        push!(w, alpha)
        push!(P, M)

        if coverage
            D_support .*= 1 .- M
            k -= 1
        end
    end

    if coverage
        P, w = refine(D, P, w, ε)
    end
    
    perm = sortperm(w, rev=true)
    w_sorted = w[perm]
    P_sorted = P[perm]

    return P_sorted, w_sorted
end

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

    if coverage
        P, w = refine(D, P, w, ε)
    end
    
    perm = sortperm(w, rev=true)
    w_sorted = w[perm]
    P_sorted = P[perm]

    return P_sorted, w_sorted
end

function eclipse(D::Matrix{Float64}, ε=1e-12, coverage=false, tol=0.0, delta=1e-3)
    ε = max(ε,1e-14);
    n = size(D, 1)
    D_rem = copy(D)
    P = Matrix{Int16}[]
    w = Float64[]

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
        
        M = zeros(n,n)
        alpha = 2.0

        H = sort(unique(D_rem))
        i_lb = 1
        i_ub = length(H)

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
        
        while i_lb < i_ub
            i = div(i_lb+i_ub, 2)
            D1 = min.(D_rem2, H[i])
            D2 = min.(D_rem2, H[i+1])

            #M1, s1 = hungarian(-D1)
            #M2, s2 = hungarian(-D2)

            M1, _ = hungarian(-D1)
            M2, _ = hungarian(-D2)
            s1 = sum(-D1[r, M1[r]] for r in 1:n)
            s2 = sum(-D2[r, M2[r]] for r in 1:n)
            
            v1 = -s1 / (H[i] + delta)
            v2 = -s2 / (H[i+1] + delta)


            if i_lb + 1 >= i_ub
                if v1 <= v2
                    alpha = H[i+1]
                    for z in 1:n
                        M[z, M2[z]] = 1
                    end
                else
                    alpha = H[i]
                    for z in 1:n
                        M[z, M1[z]] = 1
                    end
                end
                break
            end
            
            if v1 <= v2
                i_lb = i
            else
                i_ub = i
            end
            
            """
            if v1 < v2
                i_lb = i
            elseif v1 > v2
                i_ub = i
            else
                alpha = H[i]
                for z in 1:n
                    M[z, M1[z]] = 1
                end
                break
            end
            """
        end

        """
        alpha = H[i_lb]
        D1 = min.(D_rem2, H[i_lb])

        M1, _ = hungarian(-D1)
        for z in 1:n
            M[z, M1[z]] = 1
        end
        """
        
        #D_rem .= max.(D_rem .- alpha .* M, 0)
        D_rem .= D_rem .- min.(alpha .* M, D_rem)

        push!(w, alpha)
        push!(P, M)

        if coverage
            D_support .*= 1 .- M
            k -= 1
        end
    end

    if coverage
        P, w = refine(D, P, w, ε)
    end
    
    perm = sortperm(w, rev=true)
    w_sorted = w[perm]
    P_sorted = P[perm]

    return P_sorted, w_sorted
end

function generateClusters(n::Int, D::Matrix{Float64})
    n2 = size(D, 1)

    if n == n2
        return copy(D)
    end

    D2 = zeros(n,n)

    """
    for i in 1:n
        for j in 1:n
            if ceil(i / 8) == ceil(j / 8)
                num = 8 * (Int(ceil(i / 8)) - 1)
                D2[i, j] = D[i-num, j-num]
            end
        end
    end
    """
    for i in 1:n2:n
        if (n-i+1) < n2
            D2[i:n,i:n] = D[1:n-i+1,1:n-i+1]
        else
            D2[i:i+n2-1,i:i+n2-1] = D
        end
    end

    return D2
end

function generate4b()
    D = [
        0.00  4.14  3.21  3.27  3.21  3.25  3.25  3.21;
        3.20  0.00  4.10  3.19  3.17  3.17  3.20  3.17;
        3.21  3.16  0.00  4.16  3.15  3.17  3.20  3.16;
        3.27  3.19  3.22  0.00  4.16  3.22  3.25  3.19;
        3.21  3.17  3.15  3.23  0.00  4.13  3.22  3.17;
        3.25  3.17  3.17  3.22  3.19  0.00  4.17  3.18;
        3.25  3.20  3.20  3.25  3.22  3.23  0.00  4.14;
        4.15  3.17  3.16  3.19  3.17  3.18  3.20  0.00
    ]
    return D
end

function generate11b()
    D = zeros(32,32)
    for i in 0:31
        for j in 0:31
            if i != j && div(i,2) == div(j,2)
                D[i+1,j+1] += 680.0 * 125 / 1024   # TP
            end
            if abs(i-j) == 8
                D[i+1,j+1] += 16.0 * 125 / 1024    # PP
            end
            if i % 8 < 6
                if j - i == 2
                    D[i+1,j+1] += 1.0 * 741 / 1024    # DP
                end
            elseif i - j == 6 
                D[i+1,j+1] += 1.0 * 741 / 1024    # DP
            end
            if abs(i-j) == 24
                D[i+1,j+1] += 1.0 * 96 / 1024    # EmbTableSyn
            end
        end
    end
    return D
end

function generateHotI3b()
    TP = 300.0
    DP = 5.26
    PP = 7.02
    
    D = zeros(8,8)
    for i in 1:7
        D[i, i+1] = TP
    end
    D[8, 1] = TP

    D = generateClusters(48, D)

    for i in 1:40
        D[i, i+8] = DP
    end
    for i in 41:48
        D[i, i-40] = DP
    end

    D = generateClusters(192, D)
    
    for i in 1:144
        D[i, i+48] = PP
        D[i+48, i] = PP
    end
    return D
end

function generateHotIMoE()
    TP = 7.0
    DP = 0.97
    PP = 0.0224
    
    D = zeros(8,8)
    for i in 1:7
        D[i, i+1] = TP
    end
    D[8, 1] = TP

    D = generateClusters(128, D)

    for i in 1:120
        D[i, i+8] = DP
    end
    for i in 121:128
        D[i, i-120] = DP
    end

    for i in 1:128
        for j in 1:128
            if i != j && D[i, j] == 0.0
                D[i, j] = PP
            end
        end
    end
    
    return D
end

function normMatrix(D::Matrix{Float64}; noise_std=0.0)
    n = size(D, 1)
    maxrow = maximum([sum(D[i, :]) for i in 1:n])
    maxcol = maximum([sum(D[:, j]) for j in 1:n])

    maxsum = max(maxrow, maxcol)

    D_copy = D ./ maxsum
    
    if noise_std > 0.0
        mask = D_copy .> 0.0
        noise = zeros(size(D_copy))
        noise[mask] .= randn(rng, sum(mask)) .* noise_std
        D_copy .= abs.(D_copy .+ noise)
        
        maxrow = maximum([sum(D_copy[i, :]) for i in 1:n])
        maxcol = maximum([sum(D_copy[:, j]) for j in 1:n])

        maxsum = max(maxrow, maxcol)

        D_copy = D_copy ./ maxsum

    end

    return D_copy
end

function testWithParams!(results, s, delta, epsilon, n=100; ai=1,num_flows=16,pct_small=0.3)
    nL = div(num_flows, 4)
    nS = 3 * nL
    cS = pct_small
    cL = 1 - cS

    #trials_ = [20, 0]
    #trials_ = [1, 0]
    #trials_ = [10, 0]
    #trials_ = [100, 0]
    #trials_ = [3, 0]
    trials_ = [50, 0]

    #D_trials_ = [traffic_matrix(n; nL=nL, nS=nS, cL=cL, noise_std=0.0) for i in 1:trials]
    
    #D_ = traffic_matrix(n; nL=nL, nS=nS, cL=cL, noise_std=0)
    #Dk_ = D_ ./ s

    for (index, noise) in enumerate([false, true])
    
        trials = trials_[index]
    
        if trials <= 0
            continue
        end

        Dtest = nothing
        if ai == 1
            Dtest = normMatrix(generate4b())
        elseif ai == 2
            Dtest = normMatrix(generateClusters(64, generate4b()))
        elseif ai == 3
            Dtest = normMatrix(generateClusters(128, generate4b()))
        elseif ai == 4
            Dtest = normMatrix(generate11b())
        elseif ai == 5
            Dtest = normMatrix(generateClusters(64, generate11b()))
        elseif ai == 6
            Dtest = normMatrix(generateClusters(128, generate11b()))
        elseif ai == 7
            Dtest = normMatrix(generateHotI3b())
        elseif ai == 8
            Dtest = normMatrix(generateHotIMoE())
        elseif ai == 9
            Dtest = zeros(8,8)
        elseif ai == 10 # Qwen
            Dtest = zeros(64,64)
        elseif ai == 11
            Dtest = normMatrix(generate11b())
        elseif ai == 12 
            Dtest = normMatrix(generateClusters(64, generate11b()))
        elseif ai == 13 
            Dtest = normMatrix(generateClusters(128, generate11b()))
        elseif ai == 14 # Qwen
            Dtest = zeros(128,128)
        else
            continue
        end

        n = size(Dtest, 1)
        D_trials = []
        if ai < 9
            D_trials = [copy(Dtest) for i in 1:trials]
        elseif ai == 9
            for i in 1:trials
                temp = zeros(8,8)
                step_ = rand(1:1200)
                for j in 1:8
                    #data = torch.load("moe_traffics/traffic_rank_" * string(j-1) * "_step_" * string((500+i)*2-1) * ".pt", map_location="cpu").numpy() .+ torch.load("moe_traffics/traffic_rank_" * string(j-1) * "_step_" * string((500+i)*2) * ".pt", map_location="cpu").numpy()
                    data = torch.load("moe_traffics2/traffic_rank_" * string(j-1) * "_step_" * string(step_) * ".pt", map_location="cpu").numpy()
                    temp[j, :] = data
                end
                push!(D_trials, normMatrix(temp))
            end
        elseif ai == 10
            data = JSON.parsefile("ori/traffic_matrix_64.json")
            for iterat2 in 1:div(trials, 3)
                for iterat in ["iteration_1", "iteration_2", "iteration_3"]
                    raw = data["traffic_matrices"][iterat]
                    temp = Float64.(stack(raw)')
                    push!(D_trials, normMatrix(temp))
                    #push!(D_trials, perturb_doubly_stochastic(normMatrix(temp); noise_std=0.003)) 
                end
            end
            for iterat2 in 1:trials-div(trials,3)*3
                raw = data["traffic_matrices"][string("iteration_", iterat2)]
                temp = Float64.(stack(raw)')
                push!(D_trials, normMatrix(temp))
                #push!(D_trials, perturb_doubly_stochastic(normMatrix(temp); noise_std=0.003)) 
            end
        elseif ai == 11
            D_trials = [normMatrix(generate11b(); noise_std=0.003) for i in 1:trials]
        elseif ai == 12
            D_trials = [normMatrix(generateClusters(64, generate11b()); noise_std=0.003) for i in 1:trials]
        elseif ai == 13
            D_trials = [normMatrix(generateClusters(128, generate11b()); noise_std=0.003) for i in 1:trials]
        elseif ai == 14
            data = JSON.parsefile("ori/traffic_matrix_64.json")
            for iterat2 in 1:div(trials, 3)
                for iterat in ["iteration_1", "iteration_2", "iteration_3"]
                    raw = data["traffic_matrices"][iterat]
                    temp = Float64.(stack(raw)')
                    push!(D_trials, normMatrix(generateClusters(128, temp)))
                    #push!(D_trials, perturb_doubly_stochastic(normMatrix(temp); noise_std=0.003)) 
                end
            end
            for iterat2 in 1:trials-div(trials,3)*3
                raw = data["traffic_matrices"][string("iteration_", iterat2)]
                temp = Float64.(stack(raw)')
                push!(D_trials, normMatrix(generateClusters(128, temp)))
                #push!(D_trials, perturb_doubly_stochastic(normMatrix(temp); noise_std=0.003)) 
            end
        end
    
        #for decomp_num in [0,1,2,3,4,5,6,7]
        #for decomp_num in [1,2,3,4,5,6,7,8,9]
        for decomp_num in [1, 6]
            for sorted in [false]
                decomp = nothing
                """
                if decomp_num == 1
                    decomp = x -> isaac1(x, sorted ? "true" : "none")
                elseif decomp_num == 2
                    decomp = x -> isaac2(x, sorted ? "false" : "none")
                
                if decomp_num == 0
                    decomp = isaacLP
                    #if sorted
                    #    break
                    #end
                elseif decomp_num == 1
                    decomp = x -> birkdecomp1(x, epsilon, false, true)
                elseif decomp_num == 2
                    decomp = x -> birkdecomp2(x, epsilon, false, true)
                elseif decomp_num == 3
                    decomp = x -> birkdecomp3(x, epsilon, false, true)
                elseif decomp_num == 4
                    decomp = x -> birkdecomp4(x, epsilon, false, true)
                elseif decomp_num == 5
                    decomp = x -> birkdecomp5(x, epsilon, false, true)
                elseif decomp_num == 6
                    decomp = x -> birkdecomp6(x, epsilon, false, true)
                else
                    decomp = x -> birkdecomp7(x, epsilon, false, true)
                end
                """
                if decomp_num == 1
                    decomp = SPECTRA_MILP
                elseif decomp_num == 2
                    decomp = x -> birkdecomp4(x, epsilon)
                elseif decomp_num == 3
                    decomp = x -> birkdecomp4(x, epsilon, false, true)
                elseif decomp_num == 4
                    decomp = x -> greedyMWM(x, epsilon, false)
                elseif decomp_num == 5
                    decomp = x -> wwfa(x, epsilon, false)
                elseif decomp_num == 6
                    decomp = x -> eclipse(x, epsilon, false, 0.0, delta)
                elseif decomp_num == 7
                    decomp = x -> greedyMWM(x, epsilon, true)
                elseif decomp_num == 8
                    decomp = x -> wwfa(x, epsilon, true)
                elseif decomp_num == 9
                    decomp = x -> eclipse(x, epsilon, true, 0.0, delta)
                else
                    decomp = SPECTRA_MILP
                end
                #else
                #    decomp = x -> birkdecomp(x; max_rep=2)
                #end
                println("-------------------------------------------")
                if noise
                    println("WITH N")
                else 
                    println("WITHOUT N")
                end
                println("All times here in milliseconds")
            
                birk_runtime = Float64[]
                birk_less_runtime = Float64[]
                bff_less_runtime = Float64[]
            
                alg1_runtime = Float64[]
                alg1_makespan = Float64[]
                alg2_runtime = Float64[]
                alg2_makespan = Float64[]
                algMILP_runtime = Float64[]
                algMILP_makespan = Float64[]
                naive_runtime = Float64[]
                naive_makespan = Float64[]
                less_runtime = Float64[]
                less_makespan = Float64[]
            
                less_BFF_makespan = Float64[]
                alg1_BFF_runtime = Float64[]
                alg1_BFF_makespan = Float64[]
                alg2_BFF_runtime = Float64[]
                alg2_BFF_makespan = Float64[]
                alg1_BFF_delta0_runtime = Float64[]
                alg1_BFF_delta0_makespan = Float64[]
                alg2_BFF_delta0_runtime = Float64[]
                alg2_BFF_delta0_makespan = Float64[]
            
                alg1_BFF_nz = Int[]
                alg2_BFF_nz = Int[]
                alg1_BFF_delta0_nz = Int[]
                alg2_BFF_delta0_nz = Int[]
                less_nz = Int[]
    
                lesspct = Float64[]
                lessBFFpct = Float64[]

                k = Float64[]
                sum_w = Float64[]
            
                println("Number of Trials: ", trials)
                println()
                    
                for i in 1:trials
                    println("Begin Trial ", i)
                    D = D_trials[i]
                    Dk = D ./ s
            
                    P, w = (nothing, nothing)
                    P2, w2 = (nothing, nothing)
            
                    start_ = time_ns()
                    Ds = compute_lessLP(D, s)
                    less_runtime_temp = (time_ns() - start_) / 1e6
                    #cpu_times = [(@timed compute_lessLP(D, s)).time for _ in 1:time_trials]
                    #less_runtime_temp = median(cpu_times)*1000
            
                    wLESS = nothing
                    try 
                        #r = @benchmark greedy_partial_decomposition($D, $epsilon)
                        #push!(birk_runtime, median(r).time / 1e6)
                        #push!(birk_runtime, 0.0)
                        
                        start_ = time_ns()
                        P, w = decomp(D)
                        push!(birk_runtime, (time_ns() - start_) / 1e6)
                        push!(k, length(w))
                        push!(sum_w, sum(w))
                        #cpu_times = [(@timed decomp(D)).time for _ in 1:time_trials]
                        #push!(birk_runtime, median(cpu_times)*1000)
                        P2, w2 = (nothing, nothing)#decomp(Dk)
            
                        start_ = time_ns()
                        wLESS = [decomp(Dsplit)[2] for Dsplit in Ds] #UNCOMMENT if you want
                        push!(birk_less_runtime, (time_ns() - start_) / 1e6)
                        #cpu_times = [(@timed ([decomp(Dsplit)[2] for Dsplit in Ds])).time for _ in 1:time_trials]
                        #push!(birk_less_runtime, median(cpu_times)*1000)
                        push!(less_nz, sum([count(x -> x > 1e-8, Dsplit) for Dsplit in Ds]))
                    catch e
                        println("-------------------------------------------")
                        println("ERROR occured: ", e)
                        println("-------------------------------------------")
                        continue
                    end
            
                    #r = @benchmark alg1($w; s=$s, delta=$delta, epsilon=$epsilon)
                    #r = @benchmark alg2($w; s=$s, delta=$delta, epsilon=$epsilon)
                    #r = @benchmark algMILP($w; s=$s, delta=$delta, epsilon=$epsilon)
                    #push!(algMILP_runtime, median(r).time / 1e6)
                    #r = @benchmark naive($w2; s=$s, delta=$delta, epsilon=$epsilon)
                    
                    start_ = time_ns()
                    makespan = alg1(w; s=s, delta=delta, epsilon=epsilon)
                    push!(alg1_runtime, (time_ns() - start_) / 1e6)
                    push!(alg1_makespan, makespan)
            
                    start_ = time_ns()
                    makespan = alg2(w; s=s, delta=delta, epsilon=epsilon)
                    push!(alg2_runtime, (time_ns() - start_) / 1e6)
                    #cpu_times = [(@timed alg2(w; s=s, delta=delta, epsilon=epsilon)).time for _ in 1:time_trials]
                    #push!(alg2_runtime, median(cpu_times)*1000)
                    push!(alg2_makespan, makespan)
    
                    push!(lesspct, less_pct(Ds, makespan, D, delta, decomp))
                    push!(lessBFFpct, 0.0)#push!(lessBFFpct, less_bff_pct(Ds, makespan, D, delta, decomp))
            
                    start_ = time_ns()
                    makespan = 0.0#algMILP(w; s=s, delta=delta, epsilon=epsilon)
                    push!(algMILP_runtime, (time_ns() - start_) / 1e6)
                    push!(algMILP_makespan, makespan)
                    #push!(algMILP_runtime, 0.0)
                    #push!(algMILP_makespan, 0.0)
                    
                    start_ = time_ns()
                    makespan = 0.0#naive(w2; s=s, delta=delta, epsilon=epsilon)
                    push!(naive_runtime, (time_ns() - start_) / 1e6)
                    push!(naive_makespan, makespan)
            
                    makespan = maximum([sum(wsplit) + delta*(length(wsplit)) for wsplit in wLESS]) # DELTA #UNCOMMENT if you want
                    #makespan = 0.0
                    push!(less_runtime, less_runtime_temp)
                    push!(less_makespan, makespan)
            
                    start_ = time_ns()
                    makespan = maximum([bff(Dsplit, delta, epsilon / s)[1][end][1] for Dsplit in Ds])
                    push!(bff_less_runtime, (time_ns() - start_) / 1e6)
                    push!(less_BFF_makespan, makespan)
            
                    start_ = time_ns()
                    makespan, nz = 0.0, 0#alg1_BFF(w, P; s=s, delta=delta, epsilon=epsilon)
                    push!(alg1_BFF_runtime, (time_ns() - start_) / 1e6)
                    push!(alg1_BFF_makespan, makespan)
                    push!(alg1_BFF_nz, nz)
            
                    start_ = time_ns()
                    makespan, nz = alg2_BFF(w, P; s=s, delta=delta, epsilon=epsilon)
                    push!(alg2_BFF_runtime, (time_ns() - start_) / 1e6)
                    push!(alg2_BFF_makespan, makespan)
                    push!(alg2_BFF_nz, nz)
            
                    start_ = time_ns()
                    makespan, nz = 0.0, 0#alg1_BFF_deltaless(w, P; s=s, delta=delta, epsilon=epsilon)
                    push!(alg1_BFF_delta0_runtime, (time_ns() - start_) / 1e6)
                    push!(alg1_BFF_delta0_makespan, makespan)
                    push!(alg1_BFF_delta0_nz, nz)
            
                    start_ = time_ns()
                    makespan, nz = 0.0, 0#alg2_BFF_deltaless(w, P; s=s, delta=delta, epsilon=epsilon)
                    push!(alg2_BFF_delta0_runtime, (time_ns() - start_) / 1e6)
                    push!(alg2_BFF_delta0_makespan, makespan)
                    push!(alg2_BFF_delta0_nz, nz)
                end
            
                if isempty(alg1_runtime)
                    println()
                    println("PROBLEM: Trials did not produce any results!")
                    println("-------------------------------------------")
                    return
                end
            
                println()
                println("Average runtimes(ms) and standard deviation:")
                println("Greedy Decomp: ", mean(birk_runtime), " ", std(birk_runtime))
                println("Greedy Decomp LESS: ", mean(birk_less_runtime), " ", std(birk_less_runtime))
                println("BFF LESS: ", mean(bff_less_runtime), " ", std(bff_less_runtime))
            
                println("Alg1: ", mean(alg1_runtime), " ", std(alg1_runtime))
                println("Alg2: ", mean(alg2_runtime), " ", std(alg2_runtime))
                println("AlgMILP: ", mean(algMILP_runtime), " ", std(algMILP_runtime))
                println("Naive: ", mean(naive_runtime), " ", std(naive_runtime))
                println("LESS: ", mean(less_runtime), " ", std(less_runtime))
                println("Alg1 BFF: ", mean(alg1_BFF_runtime), " ", std(alg1_BFF_runtime))
                println("Alg2 BFF: ", mean(alg2_BFF_runtime), " ", std(alg2_BFF_runtime))
                println("Alg1 BFF deltaless: ", mean(alg1_BFF_delta0_runtime), " ", std(alg1_BFF_delta0_runtime))
                println("Alg2 BFF deltaless: ", mean(alg2_BFF_delta0_runtime), " ", std(alg2_BFF_delta0_runtime))
                println()
                println("Average makespan and standard deviation:")
                println("Alg1: ", mean(alg1_makespan), " ", std(alg1_makespan))
                println("Alg2: ", mean(alg2_makespan), " ", std(alg2_makespan))
                println("AlgMILP: ", mean(algMILP_makespan), " ", std(algMILP_makespan))
                println("Naive: ", mean(naive_makespan), " ", std(naive_makespan))
                println("LESS: ", mean(less_makespan), " ", std(less_makespan))
                println("Alg1 BFF: ", mean(alg1_BFF_makespan), " ", std(alg1_BFF_makespan))
                println("Alg2 BFF: ", mean(alg2_BFF_makespan), " ", std(alg2_BFF_makespan))
                println("Alg1 BFF deltaless: ", mean(alg1_BFF_delta0_makespan), " ", std(alg1_BFF_delta0_makespan))
                println("Alg2 BFF deltaless: ", mean(alg2_BFF_delta0_makespan), " ", std(alg2_BFF_delta0_makespan))
            
                println("-------------------------------------------")
            
                push!(results, (
                    s=s, delta=delta, epsilon=epsilon, noise=noise, 
                    num_flows=num_flows, pct_small=pct_small, 
            
                    birk_runtime=mean(birk_runtime), birk_runtime_std=std(birk_runtime),
                    birk_less_runtime=mean(birk_less_runtime), birk_less_runtime_std=std(birk_less_runtime),
                    bff_less_runtime=mean(bff_less_runtime), bff_less_runtime_std=std(bff_less_runtime),
                    alg1_runtime=mean(alg1_runtime), alg1_runtime_std=std(alg1_runtime),
                    alg2_runtime=mean(alg2_runtime), alg2_runtime_std=std(alg2_runtime),
                    algMILP_runtime=mean(algMILP_runtime), algMILP_runtime_std=std(algMILP_runtime),
                    naive_runtime=mean(naive_runtime), naive_runtime_std=std(naive_runtime),
                    less_runtime=mean(less_runtime), less_runtime_std=std(less_runtime),
                    alg1_BFF_runtime=mean(alg1_BFF_runtime), alg1_BFF_runtime_std=std(alg1_BFF_runtime),
                    alg2_BFF_runtime=mean(alg2_BFF_runtime), alg2_BFF_runtime_std=std(alg2_BFF_runtime),
                    alg1_BFF_delta0_runtime=mean(alg1_BFF_delta0_runtime), alg1_BFF_delta0_runtime_std=std(alg1_BFF_delta0_runtime),
                    alg2_BFF_delta0_runtime=mean(alg2_BFF_delta0_runtime), alg2_BFF_delta0_runtime_std=std(alg2_BFF_delta0_runtime),
            
                    alg1_makespan=mean(alg1_makespan), alg1_makespan_std=std(alg1_makespan),
                    alg2_makespan=mean(alg2_makespan), alg2_makespan_std=std(alg2_makespan),
                    algMILP_makespan=mean(algMILP_makespan), algMILP_makespan_std=std(algMILP_makespan),
                    naive_makespan=mean(naive_makespan), naive_makespan_std=std(naive_makespan),
                    less_makespan=mean(less_makespan), less_makespan_std=std(less_makespan),
                    less_BFF_makespan=mean(less_BFF_makespan), less_BFF_makespan_std=std(less_BFF_makespan),
                    alg1_BFF_makespan=mean(alg1_BFF_makespan), alg1_BFF_makespan_std=std(alg1_BFF_makespan),
                    alg2_BFF_makespan=mean(alg2_BFF_makespan), alg2_BFF_makespan_std=std(alg2_BFF_makespan),
                    alg1_BFF_delta0_makespan=mean(alg1_BFF_delta0_makespan), alg1_BFF_delta0_makespan_std=std(alg1_BFF_delta0_makespan),
                    alg2_BFF_delta0_makespan=mean(alg2_BFF_delta0_makespan), alg2_BFF_delta0_makespan_std=std(alg2_BFF_delta0_makespan),
            
                    n=n,
                    alg1_BFF_nz=round(Int, mean(alg1_BFF_nz)),
                    alg2_BFF_nz=round(Int, mean(alg2_BFF_nz)),
                    alg1_BFF_delta0_nz=round(Int, mean(alg1_BFF_delta0_nz)),
                    alg2_BFF_delta0_nz=round(Int, mean(alg2_BFF_delta0_nz)),
                    less_nz=round(Int, mean(less_nz)),
    
                    decomp_num=decomp_num,
                    sorted=sorted,
    
                    lesspct=mean(lesspct),
                    lessBFFpct=mean(lessBFFpct),

                    k=mean(k),
                    sum_w=mean(sum_w),

                    ai=ai
                ))
            end
        end
    end
end

# #epsilon_ = [1e-2, 1e-1, 1e-3]
# epsilon_ = [0.0]
# #s_ = [4, 8, 16]
# #s_ = [32, 64, 128]
# s_ = [4, 16]
# #s_ = [1, 2, 4, 8, 16]
# #s_ = [1, 4, 16]
# delta_ = [0.01, 0.02, 0.04, 0.08]
# #n_ = [100, 200, 400]
# n_ = [100]

# #ai_=[1,2,3,4,5,6]
# #ai_=[7,8]
# #ai_=[1,4]
# #ai_=[1,2,3,4,5,6,7,8]
# #ai_=[10]
# #ai_=[1,2]
# ai_=[4,10,11]
# #ai_=[3,10,11,12,13,14]

# #num_flows_ = [16, 32, 64]
# #pct_small_ = [0.3, 0.15, 0.6]

# num_flows_ = []
# pct_small_ = []

# #epsilon_ = [1e-2]
# #s_ = [4, 8]
# #delta_ = [0.02, 0.04]

# #num_flows_ = [8, 16, 24, 32, 40, 48, 56, 64]
# #pct_small_ = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]

# results = DataFrame(
#     s=Int[], delta=Float64[], epsilon=Float64[], noise=Bool[], 
#     num_flows=Int[], pct_small=Float64[],

#     birk_runtime=Float64[], birk_runtime_std=Float64[],
#     birk_less_runtime=Float64[], birk_less_runtime_std=Float64[],
#     bff_less_runtime=Float64[], bff_less_runtime_std=Float64[],
#     alg1_runtime=Float64[], alg1_runtime_std=Float64[],
#     alg2_runtime=Float64[], alg2_runtime_std=Float64[],
#     algMILP_runtime=Float64[], algMILP_runtime_std=Float64[],
#     naive_runtime=Float64[], naive_runtime_std=Float64[],
#     less_runtime=Float64[], less_runtime_std=Float64[],
#     alg1_BFF_runtime=Float64[], alg1_BFF_runtime_std=Float64[],
#     alg2_BFF_runtime=Float64[], alg2_BFF_runtime_std=Float64[],
#     alg1_BFF_delta0_runtime=Float64[], alg1_BFF_delta0_runtime_std=Float64[],
#     alg2_BFF_delta0_runtime=Float64[], alg2_BFF_delta0_runtime_std=Float64[],

#     alg1_makespan=Float64[], alg1_makespan_std=Float64[],
#     alg2_makespan=Float64[], alg2_makespan_std=Float64[],
#     algMILP_makespan=Float64[], algMILP_makespan_std=Float64[],
#     naive_makespan=Float64[], naive_makespan_std=Float64[],
#     less_makespan=Float64[], less_makespan_std=Float64[],
#     less_BFF_makespan=Float64[], less_BFF_makespan_std=Float64[],
#     alg1_BFF_makespan=Float64[], alg1_BFF_makespan_std=Float64[],
#     alg2_BFF_makespan=Float64[], alg2_BFF_makespan_std=Float64[],
#     alg1_BFF_delta0_makespan=Float64[], alg1_BFF_delta0_makespan_std=Float64[],
#     alg2_BFF_delta0_makespan=Float64[], alg2_BFF_delta0_makespan_std=Float64[],

#     n=Int[],
#     alg1_BFF_nz=Int[],
#     alg2_BFF_nz=Int[],
#     alg1_BFF_delta0_nz=Int[],
#     alg2_BFF_delta0_nz=Int[],
#     less_nz=Int[],

#     decomp_num=Int[],
#     sorted=Bool[],

#     lesspct=Float64[],
#     lessBFFpct=Float64[],

#     k=Float64[],
#     sum_w=Float64[],

#     ai=Int[]
# )

# if length(num_flows_) == 0 && length(pct_small_) == 0
#     # CACHE WARMING
#     testWithParams!(results, s_[1], delta_[1], epsilon_[1], n_[1]; ai=ai_[1]);
#     empty!(results)


#     for ai in ai_
#         for n in n_
#             for epsilon in epsilon_
#                 for s in s_
#                     for delta in delta_
#                         testWithParams!(results, s, delta, epsilon, n; ai=ai);
#                         CSV.write("birkhoff_qwen_MOE7.csv", results)
#                     end
#                 end
#             end
#         end
#     end

# else 
#     for n in n_
#         for epsilon in epsilon_
#             for s in s_
#                 for delta in delta_
#                     for num_flows in num_flows_
#                         testWithParams!(results, s, delta, epsilon, n; num_flows=num_flows);
#                         CSV.write("resultsSAFE.csv", results)
#                     end
#                     for pct_small in pct_small_
#                         testWithParams!(results, s, delta, epsilon, n; pct_small=pct_small);
#                         CSV.write("resultsSAFE.csv", results)
#                     end
#                 end
#             end
#         end
#     end
# end

# CSV.write("birkhoff_qwen_MOE7.csv", results)
