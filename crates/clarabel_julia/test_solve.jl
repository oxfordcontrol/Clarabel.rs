using SparseArrays
using Clarabel
using TimerOutputs

include("./types.jl")
include("./clarabel_rs.jl")

using Random
Random.seed!(1234)

n = 2000

P = sparse(I(n)).*1.
P = triu(P)
q = ones(n)
A = [I(n);-I(n)].*1.
b = ones(2*n)

m = 2*n;
P = sprandn(n,n,0.05)
P = P+P';
d = sum(abs.(P),dims=1);
P = P + Diagonal(d[:]);
P = triu(P);
A = sprandn(m,n,0.05);
b = ones(m)
q = randn(n)

println("\n\nLaunching Julia\n-----------------")
TimerOutputs.enable_debug_timings(Clarabel)
settings = Clarabel.Settings(max_iter = 0, verbose = true)
solver   = Clarabel.Solver(settings)
cone_types = [Clarabel.NonnegativeConeT]
cone_dims = [m]

Clarabel.setup!(solver, P, q, A, b, cone_types, cone_dims)
result = Clarabel.solve!(solver)
print(solver.info.timer) 

println("\n\nLaunching Rust\n-----------------")

solve_time = solve_rs(solver, P, q, A, b, cone_types, cone_dims)



print(solver.info.timer) 