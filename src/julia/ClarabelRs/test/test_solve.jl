using SparseArrays, Clarabel
using ClarabelRs

P = sparse([3. 1.;1. 2.].*2)
q = [-1., -4.]
A = sparse([1.  0.;    #<-- LHS of inequality constraint (upper bound)
            0.  1.;    #<-- LHS of inequality constraint (upper bound)
           -1.  0.;    #<-- LHS of inequality constraint (lower bound)
            0. -1.;    #<-- LHS of inequality constraint (lower bound)
            ])

b = [ones(4);   #<-- RHS of inequality constraints
    ]

cones = Clarabel.SupportedCone[
    Clarabel.ZeroConeT(1) #<--- for the inequality constraints
    Clarabel.SecondOrderConeT(3) #<--- for the inequality constraints
]    


settings = Clarabel.Settings(max_iter = 25, verbose = true)

println("Rust version")
solver1 = ClarabelRs.Solver(P, q, A, b, cones, settings)
solution1 = ClarabelRs.solve!(solver1)
info1 = ClarabelRs.get_info(solver1)


println("Julia version")
solver2 = Clarabel.Solver(P, q, A, b, cones, settings)
solution2 = Clarabel.solve!(solver2)
info2 = solver2.info

