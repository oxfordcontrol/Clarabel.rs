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
    # Simple QP: min (1/2)x'Px + q'x s.t. Ax <= b, x >= 0
    # This avoids equality constraints which can cause negative dual variables
    
    P = sparse.csc_matrix([[2., 0.], [0., 2.]])
    q = np.array([0., 0.])
    
    # Only inequality constraints: x >= 0, x <= [2, 2]
    A = sparse.csc_matrix([
        [1., 0.],    # x[0] >= 0 (will be x[0] <= inf)
        [0., 1.],    # x[1] >= 0 (will be x[1] <= inf)
        [-1., 0.],   # x[0] <= 2 (will be -x[0] <= -2, so x[0] >= 2, wait...)
        [0., -1.]    # x[1] <= 2
    ])
    b = np.array([2., 2., -0.5, -0.5])  # x <= [2,2] and x >= [0.5, 0.5]
    
    cones = [clarabel.NonnegativeConeT(4)]
    
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
    print(f"Warm start used: {solution1.warm_start_used}")
    print(f"Solution values for warm start:")
    print(f"  s = {solution1.s}")
    print(f"  z = {solution1.z}")
    print(f"  tau = {solution1.tau}")
    print(f"  kappa = {solution1.kappa}")
    print()
    
    # Modify the problem slightly (change the objective)
    print("2. Modifying problem and solving with warm start...")
    q_new = q + 0.01 * np.ones(len(q))  # very small change to objective
    
    # Use warm start values but adjust them slightly to be closer to the new optimum
    # The warm start values should be feasible and somewhat close to optimal for the new problem
    x_warm = solution1.x.copy()
    s_warm = solution1.s.copy()  
    z_warm = solution1.z.copy()
    tau_warm = solution1.tau
    kappa_warm = solution1.kappa
    
    # Create new solver with warm start settings
    settings_warm = clarabel.DefaultSettings()
    settings_warm.verbose = True
    settings_warm.warm_start_enable = True
    settings_warm.warm_start_x = x_warm
    settings_warm.warm_start_s = s_warm
    settings_warm.warm_start_z = z_warm
    settings_warm.warm_start_tau = tau_warm
    settings_warm.warm_start_kappa = kappa_warm
    
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
