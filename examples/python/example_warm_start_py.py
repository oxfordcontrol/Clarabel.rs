#!/usr/bin/env python3

"""
Example demonstrating warm start functionality in Clarabel Python bindings.

This example shows how to:
1. Solve an optimization problem
2. Modify the problem slightly 
3. Use the previous solution as a warm start for faster convergence
"""

import numpy as np
import clarabel

def create_qp_problem():
    """Create a simple quadratic programming problem."""
    # min (1/2)x'Px + q'x
    # s.t. Ax = b, x >= 0
    
    # Problem dimensions
    n = 4  # number of variables
    m = 2  # number of equality constraints
    
    # Objective: minimize ||x - [1,1,1,1]||^2
    P = np.eye(n)
    q = -np.ones(n)
    
    # Constraints: sum(x) = 2, x[0] + x[1] = 1
    A = np.array([
        [1., 1., 1., 1.],
        [1., 1., 0., 0.]
    ])
    b = np.array([2., 1.])
    
    # Cone constraints: x >= 0
    cones = [clarabel.NonnegativeConeT(n)]
    
    return P, q, A, b, cones

def solve_with_warm_start():
    """Demonstrate warm start functionality."""
    
    print("=== Clarabel Python Warm Start Example ===\n")
    
    # Create the problem
    P, q, A, b, cones = create_qp_problem()
    
    # Initial solve
    print("1. Solving initial problem...")
    settings = clarabel.DefaultSettings()
    settings.verbose = True
    
    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solver.solve()
    
    solution1 = solver.solution
    print(f"Initial solution: x = {solution1.x}")
    print(f"Initial solve time: {solver.info.solve_time:.4f}s")
    print(f"Initial iterations: {solver.info.iterations}\n")
    
    # Modify the problem slightly (change the objective)
    print("2. Modifying problem and solving with warm start...")
    q_new = q + 0.1 * np.ones(len(q))  # slightly different objective
    
    # Create new solver with warm start settings
    settings_warm = clarabel.DefaultSettings()
    settings_warm.verbose = True
    settings_warm.warm_start_enable = True
    settings_warm.warm_start_x = solution1.x.tolist()
    settings_warm.warm_start_s = solution1.s.tolist()
    settings_warm.warm_start_z = solution1.z.tolist()
    settings_warm.warm_start_tau = solution1.tau
    settings_warm.warm_start_kappa = solution1.kappa
    
    solver_warm = clarabel.DefaultSolver(P, q_new, A, b, cones, settings_warm)
    solver_warm.solve()
    
    solution2 = solver_warm.solution
    print(f"Warm start solution: x = {solution2.x}")
    print(f"Warm start solve time: {solver_warm.info.solve_time:.4f}s")
    print(f"Warm start iterations: {solver_warm.info.iterations}\n")
    
    # Compare with cold start
    print("3. Solving modified problem without warm start (cold start)...")
    settings_cold = clarabel.DefaultSettings()
    settings_cold.verbose = True
    
    solver_cold = clarabel.DefaultSolver(P, q_new, A, b, cones, settings_cold)
    solver_cold.solve()
    
    solution3 = solver_cold.solution
    print(f"Cold start solution: x = {solution3.x}")
    print(f"Cold start solve time: {solver_cold.info.solve_time:.4f}s")
    print(f"Cold start iterations: {solver_cold.info.iterations}\n")
    
    # Summary
    print("=== Performance Comparison ===")
    print(f"Warm start iterations: {solver_warm.info.iterations}")
    print(f"Cold start iterations: {solver_cold.info.iterations}")
    
    if solver_cold.info.iterations > 0:
        speedup = solver_cold.info.iterations / max(1, solver_warm.info.iterations)
        print(f"Iteration speedup: {speedup:.2f}x")
    
    print(f"Warm start time: {solver_warm.info.solve_time:.4f}s")
    print(f"Cold start time: {solver_cold.info.solve_time:.4f}s")

if __name__ == "__main__":
    solve_with_warm_start()
