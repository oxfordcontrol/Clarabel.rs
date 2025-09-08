#!/usr/bin/env python3

"""
Example showing access to mu parameter and warm_start_used 
in both solution and info objects.
"""

import clarabel
import numpy as np
from scipy import sparse

def test_mu_access():
    # Define a simple QP problem: min 0.5*x'*P*x + q'*x
    # subject to A*x = b and x >= 0
    
    # Problem: minimize x^2 + y^2 subject to x + y = 1, x >= 0, y >= 0
    P = sparse.csc_matrix([[2.0, 0.0], [0.0, 2.0]])
    q = np.array([0.0, 0.0])
    A = sparse.csc_matrix([[1.0, 1.0]])  # equality constraint: x + y = 1
    b = np.array([1.0])
    
    # Cone: 1 equality constraint
    cones = [clarabel.ZeroConeT(1)]
    
    # Settings with warm start enabled
    settings = clarabel.DefaultSettings()
    settings.warm_start_enable = True
    settings.verbose = True
    
    # Create solver
    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    
    # Initial solve
    print("=== Initial solve ===")
    solution = solver.solve()
    
    print(f"Solution status: {solution.status}")
    print(f"Warm start used: {solution.warm_start_used}")
    print(f"Primal solution: x = {solution.x}")
    
    # Access info object
    info = solver.get_info()
    print(f"Final mu value: {info.mu}")
    print(f"Info warm_start_used: {info.warm_start_used}")
    print(f"Info iterations: {info.iterations}")
    print(f"Info cost_primal: {info.cost_primal}")
    
    # Set warm start values close to the solution
    warm_x = np.array([0.5, 0.5])  # close to optimal [0.5, 0.5]
    warm_s = np.array([0.0])       # equality constraint slack
    warm_z = np.array([1.0])       # dual variable
    
    # Create new settings with warm start
    settings_warm = clarabel.DefaultSettings()
    settings_warm.warm_start_enable = True
    settings_warm.warm_start_x = warm_x
    settings_warm.warm_start_s = warm_s
    settings_warm.warm_start_z = warm_z
    settings_warm.verbose = True
    
    # Create new solver with warm start
    solver_warm = clarabel.DefaultSolver(P, q, A, b, cones, settings_warm)
    
    # Solve again with warm start
    print("\n=== Warm start solve ===")
    solution2 = solver_warm.solve()
    
    print(f"Solution status: {solution2.status}")
    print(f"Warm start used: {solution2.warm_start_used}")
    print(f"Primal solution: x = {solution2.x}")
    
    # Access info object after warm start
    info2 = solver_warm.get_info()
    print(f"Final mu value: {info2.mu}")
    print(f"Info warm_start_used: {info2.warm_start_used}")
    print(f"Info iterations: {info2.iterations}")
    
    # Should have fewer iterations with warm start
    print(f"\nIterations comparison:")
    print(f"Cold start: {info.iterations} iterations")
    print(f"Warm start: {info2.iterations} iterations")
    
    # Verify mu is accessible
    assert hasattr(info, 'mu'), "mu should be accessible in info object"
    assert hasattr(info, 'warm_start_used'), "warm_start_used should be accessible in info object"
    assert hasattr(solution, 'warm_start_used'), "warm_start_used should be accessible in solution object"
    
    print("\n✅ All mu and warm_start_used accesses work correctly!")

if __name__ == "__main__":
    test_mu_access()
