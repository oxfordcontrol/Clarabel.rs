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
from scipy import sparse

def create_qp_problem():
    """Create a simple quadratic programming problem."""
    # min (1/2)x'Px + q'x s.t. Ax = b, x >= 0
    
    # Simple QP similar to the Rust example
    P = sparse.csc_matrix([[6., 0.], [0., 4.]])
    q = np.array([-1., -4.])
    
    A = sparse.csc_matrix([
        [1., -2.],        # equality constraint
        [1.,  0.],        # x[0] >= 0 bound
        [0.,  1.],        # x[1] >= 0 bound  
        [-1., 0.],        # x[0] <= some_bound
        [0., -1.]         # x[1] <= some_bound
    ])
    b = np.array([0., 1., 1., 1., 1.])
    
    cones = [clarabel.ZeroConeT(1), clarabel.NonnegativeConeT(4)]
    
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
    solution1 = solver.solve()
    
    print(f"Initial solution: x = {solution1.x}")
    print(f"Initial iterations: {solution1.iterations}")
    print(f"Warm start used: {solution1.warm_start_used}\n")
    
    # Modify the problem slightly (change the objective)
    print("2. Modifying problem and solving with warm start...")
    q_new = q + 0.1 * np.ones(len(q))  # slightly different objective
    
    # Create new solver with warm start settings
    settings_warm = clarabel.DefaultSettings()
    settings_warm.verbose = True
    settings_warm.warm_start_enable = True
    settings_warm.warm_start_x = solution1.x
    settings_warm.warm_start_s = solution1.s
    settings_warm.warm_start_z = solution1.z
    settings_warm.warm_start_tau = solution1.tau
    settings_warm.warm_start_kappa = solution1.kappa
    
    solver_warm = clarabel.DefaultSolver(P, q_new, A, b, cones, settings_warm)
    solution2 = solver_warm.solve()
    
    print(f"Warm start solution: x = {solution2.x}")
    print(f"Warm start iterations: {solution2.iterations}")
    print(f"Warm start used: {solution2.warm_start_used}\n")
    
    # Compare with cold start
    print("3. Solving modified problem without warm start (cold start)...")
    settings_cold = clarabel.DefaultSettings()
    settings_cold.verbose = True
    
    solver_cold = clarabel.DefaultSolver(P, q_new, A, b, cones, settings_cold)
    solution3 = solver_cold.solve()
    
    print(f"Cold start solution: x = {solution3.x}")
    print(f"Cold start iterations: {solution3.iterations}")
    print(f"Warm start used: {solution3.warm_start_used}\n")
    
    # Summary
    print("=== Performance Comparison ===")
    print(f"Warm start iterations: {solution2.iterations} (used: {solution2.warm_start_used})")
    print(f"Cold start iterations: {solution3.iterations} (used: {solution3.warm_start_used})")
    
    if solution3.iterations > 0:
        speedup = solution3.iterations / max(1, solution2.iterations)
        print(f"Iteration speedup: {speedup:.2f}x")

if __name__ == "__main__":
    solve_with_warm_start()
