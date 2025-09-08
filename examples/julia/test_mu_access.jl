#!/usr/bin/env julia

"""
Example showing access to mu parameter in Julia interface.
"""

using Clarabel, SparseArrays
using ClarabelRs

function test_mu_access()
    println("=== Julia μ Access Test ===\n")
    
    # Define a simple QP problem: min 0.5*x'*P*x + q'*x
    # subject to A*x = b
    
    # Problem: minimize x^2 + y^2 subject to x + y = 1
    P = sparse([2.0 0.0; 0.0 2.0])
    q = [0.0, 0.0]
    A = sparse([1.0 1.0])  # equality constraint: x + y = 1 (1x2 matrix)
    b = [1.0]
    
    # Cone: 1 equality constraint (using Clarabel syntax)
    cones = [Clarabel.ZeroConeT(1)]
    
    # Settings
    settings = Clarabel.Settings()
    settings.verbose = true
    
    # Solve the problem
    println("=== Solving problem ===")
    solver = ClarabelRs.Solver(P, q, A, b, cones, settings)
    solution = ClarabelRs.solve!(solver)
    
    println("Solution status: ", solution.status)
    println("Primal solution: x = ", solution.x)
    
    # Access info object
    info = ClarabelRs.get_info(solver)
    println("Final μ value: ", info.mu)
    println("Info iterations: ", info.iterations)
    println("Info cost_primal: ", info.cost_primal)
    
    # Check if warm_start_used field exists
    if hasfield(typeof(info), :warm_start_used)
        println("Info warm_start_used: ", info.warm_start_used)
    else
        println("warm_start_used field not available in this version")
    end
    
    if hasfield(typeof(solution), :warm_start_used)
        println("Solution warm_start_used: ", solution.warm_start_used)
    else
        println("warm_start_used field not available in solution in this version")
    end
    
    # Verify μ is accessible
    @assert hasfield(typeof(info), :mu) "μ should be accessible in info object"
    
    println("\n✅ μ access works correctly!")
    
    # Show available fields in info and solution
    println("\nAvailable fields in info:")
    println(fieldnames(typeof(info)))
    
    println("\nAvailable fields in solution:")
    println(fieldnames(typeof(solution)))
end

# Run the test
test_mu_access()
