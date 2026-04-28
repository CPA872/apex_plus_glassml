# Summary of `spectra.jl`

This Julia script (`spectra.jl`) implements a simulation and evaluation framework for various algorithms designed to schedule network traffic. The core problem is to take a traffic demand matrix `D` (where `D[i,j]` is the traffic from source `i` to destination `j`) and schedule it across `s` parallel resources or sub-schedulers to minimize the total completion time (makespan).

The code explores several distinct strategies, primarily revolving around matrix decomposition, load balancing (bin packing), and direct event-driven scheduling.

## Core Concepts

1.  **Traffic Matrix (D)**: An `n x n` matrix representing traffic demands. The script includes functions to generate synthetic traffic (`traffic_matrix`) as well as load traffic patterns from real-world AI models like Mixture-of-Experts (MoE) and Qwen (`generate4b`, `generate11b`, `generateHotI3b`, etc.).

2.  **Matrix Decomposition**: A central strategy is to decompose the traffic matrix `D` into a weighted sum of permutation matrices: `D ≈ Σ w_i * P_i`. Each permutation matrix `P_i` represents a conflict-free set of transmissions, and its weight `w_i` represents the time duration for which this set is active. The set of weights `{w_i}` can be seen as a collection of "balls" to be scheduled.

3.  **Scheduling as Bin Packing**: Once decomposed, the problem becomes scheduling the "balls" (weights `w_i`) into `s` "bins" (the parallel resources). A reconfiguration cost `delta` is often added to a bin each time a new ball is placed in it. The goal is to minimize the maximum load of any bin.

4.  **Matrix Splitting (LESS)**: An alternative to full decomposition is to first split the matrix `D` into `s` smaller sub-matrices `D_1, ..., D_s` such that `D = Σ D_i`. Then, a scheduling algorithm can be run independently on each sub-matrix.

## Implemented Algorithms

The file implements a wide range of algorithms, which can be categorized as follows.

### 1. Matrix Decomposition Algorithms

These algorithms take a matrix `D` and produce a set of permutation matrices `P` and their corresponding weights `w`.

*   **`greedy_partial_decomposition`**: A classic and effective heuristic. It iteratively finds the maximum weight matching in the remaining demand matrix (using the Hungarian algorithm via `scipy.optimize.linear_sum_assignment`), peels it off as a new permutation `P_i`, assigns its weight `w_i`, and subtracts `w_i * P_i` from the matrix.
*   **`SPECTRA` / `SPECTRA_MILP`**: A custom decomposition heuristic. It appears to be a variant of the greedy matching approach where the cost matrix for the Hungarian algorithm is modified to prioritize completing rows/columns with high sparsity. `SPECTRA_MILP` refines the weights using a Mixed-Integer Linear Program.
*   **`birkdecomp` Family (`birkdecomp`, `birkdecomp1`...`birkdecomp7`)**: Implements several variations of the Birkhoff-von Neumann decomposition, which is theoretically guaranteed to decompose any doubly stochastic matrix. These appear to be Frank-Wolfe style algorithms that iteratively find an extreme point (a permutation matrix) of the Birkhoff polytope that moves closest to the target matrix. They include methods (`expandStochastic`, `vonN`) to handle sub-stochastic matrices by extending them to be doubly stochastic.
*   **`greedyMWM`**: A greedy Maximum Weight Matching algorithm that does not necessarily produce a full permutation.
*   **`wwfa`**: Appears to be a "Wavefront Arbiter" style decomposition heuristic.
*   **`eclipse`**: A heuristic that uses binary search on the values in the demand matrix to find an optimal `alpha` for a weighted matching problem.

### 2. Scheduling / Bin Packing Algorithms

These algorithms take a set of weights `w` (from a decomposition) and schedule them into `s` bins.

*   **`alg1`**: A greedy load-balancing algorithm, analogous to Longest Processing Time (LPT) in bin packing. It sorts the balls (weights) in descending order and places each ball into the bin with the currently lowest total load.
*   **`alg2`**: An iterative refinement algorithm. It starts with the result of `alg1` and then attempts to balance the load by moving a fraction of the largest "ball" from the most loaded bin to the least loaded bin. This process repeats until the load difference between the max and min bins is smaller than a given threshold (`delta`).
*   **`algMILP`**: A Mixed-Integer Linear Program that finds the optimal solution to the bin packing problem, allowing for balls to be split across bins. It serves as an optimal baseline.

### 3. Direct & Hybrid Scheduling Algorithms

These approaches schedule traffic without a clean separation between decomposition and bin packing.

*   **`bff` (Best-Fit First)**: An event-driven direct scheduler. It simulates the network over time, maintaining a priority queue of events (e.g., an input or output port becoming free). When a port becomes free, it greedily schedules the largest remaining flow available for that port.
*   **`alg1_BFF` / `alg2_BFF`**: These are hybrid algorithms. They first use a binning algorithm (`alg1` or `alg2`) to partition the *permutation matrices* (not just their weights) into `s` groups. Then, for each group, they reconstruct a sub-matrix and run the `bff` scheduler on it. The final makespan is the maximum makespan across all `s` `bff` instances.

### 4. Matrix Splitting Algorithms

*   **`compute_lessLP`**: This function implements the "LESS" scheduling approach. It recursively splits the initial demand matrix `D` into `s` sub-matrices.
*   **`compute_D1` / `improve_D1_via_alternating_cycles`**: These are the core splitting functions. `compute_D1` uses a Linear Program (LP) to find a valid split. `improve_D1_via_alternating_cycles` is a combinatorial heuristic that refines a split by finding and resolving imbalances along cycles in a corresponding bipartite graph.

## Experimentation Framework

The bottom part of the script contains a comprehensive testing framework.

*   **`testWithParams!`**: This is the main driver function. It iterates through different parameters (`s`, `delta`, `epsilon`, traffic type) and runs a suite of the implemented algorithms for each combination.
*   **Benchmarking**: It measures and records the runtime and resulting makespan for each algorithm.
*   **Results**: The results are collected into a `DataFrame` and saved to a CSV file (e.g., `birkhoff_qwen_MOE7.csv`). This allows for offline analysis and plotting of the performance trade-offs between the different algorithms.

## Dependencies

The script relies on several Julia and Python packages:

*   **Julia**:
    *   `JuMP`, `Gurobi`: For mathematical optimization (LP/MILP).
    *   `DataStructures`: For efficient priority queues (`BinaryMinHeap`, `BinaryMinMaxHeap`).
    *   `Graphs`: For graph-based algorithms like the alternating cycle search.
    *   `CSV`, `DataFrames`: For results management.
*   **Python (via `PyCall`)**:
    *   `torch`: To load traffic matrices saved as PyTorch tensors.
    *   `scipy`: Specifically for `scipy.optimize.linear_sum_assignment`, which is an efficient implementation of the Hungarian algorithm used for max-weight matching.