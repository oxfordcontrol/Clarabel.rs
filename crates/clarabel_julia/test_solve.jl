using SparseArrays

include("./types.jl")
include("./clarabel_rs.jl")

using Clarabel
using TimerOutputs

using Random
Random.seed!(1234)

P = sparse([3. 1.;1. 2.].*2)
q = [-1., -4.]
A = sparse([1.  0.;    #<-- LHS of inequality constraint (upper bound)
            0.  1.;    #<-- LHS of inequality constraint (upper bound)
           -1.  0.;    #<-- LHS of inequality constraint (lower bound)
            0. -1.;    #<-- LHS of inequality constraint (lower bound)
            ])

b = [ones(4);   #<-- RHS of inequality constraints
    ]

    cones =
    [Clarabel.NonnegativeConeT(4)]    #<--- for the inequality constraints


println("\n\n Calling Julia implementation\n-----------------")
TimerOutputs.enable_debug_timings(Clarabel)
settings = Clarabel.Settings(max_iter = 20, verbose = true)
solver   = Clarabel.Solver(settings)

Clarabel.setup!(solver, P, q, A, b, cones)
result = Clarabel.solve!(solver)
print(solver.info.timer) 

println("\n\n Calling Rust implementation Rust\n-----------------")

solve_time = solve_rs((P), q, A, b, cones)



