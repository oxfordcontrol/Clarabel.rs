#!/usr/bin/env julia

"""
Example demonstrating warm start functionality in Clarabel Julia bindings.

This example shows how to:
1. Solve an optimization problem
2. Modify the problem slightly 
3. Use the previous solution as a warm start for faster convergence
"""

using Clarabel
using SparseArrays
using LinearAlgebra

function create_qp_problem()
    """Create a simple quadratic programming problem."""
    # min (1/2)x'Px + q'x
    # s.t. Ax = b, x >= 0
    
    # Problem dimensions
    n = 4  # number of variables
    m = 2  # number of equality constraints
    
    # Objective: minimize ||x - [1,1,1,1]||^2
    P = sparse(Matrix(I, n, n))
    q = -ones(n)
    
    # Constraints: sum(x) = 2, x[0] + x[1] = 1
    A = sparse([
        1.0 1.0 1.0 1.0;
        1.0 1.0 0.0 0.0
    ])
    b = [2.0, 1.0]
    
    # Cone constraints: x >= 0
    cones = [Clarabel.NonnegativeConeT(n)]
    
    return P, q, A, b, cones
end

function solve_with_warm_start()
    """Demonstrate warm start functionality."""
    
    println("=== Clarabel Julia Warm Start Example ===\\n")
    
    # Create the problem
    P, q, A, b, cones = create_qp_problem()
    
    # Initial solve
    println("1. Solving initial problem...")
    settings = Clarabel.Settings()
    settings.verbose = true
    
    solver = Clarabel.Solver()
    Clarabel.setup!(solver, P, q, A, b, cones, settings)
    result1 = Clarabel.solve!(solver)
    
    println("Initial solution: x = ", result1.x)
    println("Initial solve time: ", round(result1.solve_time, digits=4), "s")
    println("Initial iterations: ", result1.iterations, "\\n")
    
    # Modify the problem slightly (change the objective)
    println("2. Modifying problem and solving with warm start...")
    q_new = q .+ 0.1  # slightly different objective
    
    # Create new solver with warm start settings
    settings_warm = Clarabel.Settings()
    settings_warm.verbose = true
    settings_warm.warm_start_enable = true
    settings_warm.warm_start_x = result1.x
    settings_warm.warm_start_s = result1.s
    settings_warm.warm_start_z = result1.z
    settings_warm.warm_start_tau = result1.tau
    settings_warm.warm_start_kappa = result1.kappa
    
    solver_warm = Clarabel.Solver()
    Clarabel.setup!(solver_warm, P, q_new, A, b, cones, settings_warm)
    result2 = Clarabel.solve!(solver_warm)
    
    println("Warm start solution: x = ", result2.x)
    println("Warm start solve time: ", round(result2.solve_time, digits=4), "s")
    println("Warm start iterations: ", result2.iterations, "\\n")
    
    # Compare with cold start
    println("3. Solving modified problem without warm start (cold start)...")
    settings_cold = Clarabel.Settings()
    settings_cold.verbose = true
    
    solver_cold = Clarabel.Solver()
    Clarabel.setup!(solver_cold, P, q_new, A, b, cones, settings_cold)
    result3 = Clarabel.solve!(solver_cold)
    
    println("Cold start solution: x = ", result3.x)
    println("Cold start solve time: ", round(result3.solve_time, digits=4), "s")
    println("Cold start iterations: ", result3.iterations, "\\n")
    
    # Summary
    println("=== Performance Comparison ===")
    println("Warm start iterations: ", result2.iterations)
    println("Cold start iterations: ", result3.iterations)
    
    if result3.iterations > 0
        speedup = result3.iterations / max(1, result2.iterations)
        println("Iteration speedup: ", round(speedup, digits=2), "x")
    end
    
    println("Warm start time: ", round(result2.solve_time, digits=4), "s")
    println("Cold start time: ", round(result3.solve_time, digits=4), "s")
end

# Run the example
solve_with_warm_start()
