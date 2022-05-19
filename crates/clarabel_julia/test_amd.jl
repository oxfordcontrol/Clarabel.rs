using SparseArrays
using Clarabel
using TimerOutputs
using BenchmarkTools

include("./types.jl")
push!(Base.DL_LOAD_PATH,"../../target/debug/")
lib = "libclarabel_julia.dylib"

using Random
Random.seed!(1234)

n = 4000
m = 2*n;
P = sprandn(n,n,0.05)
P = P+P';
d = sum(abs.(P),dims=1);
P = P + Diagonal(d[:]);
P = triu(P);
A = sprandn(m,n,0.05);

K = [P A';A -I(m)]

println("\n\nLaunching AMD\n-----------------")

M = CscMatrixRs(K)
dump(M)
println("colptr[1:3]", K.colptr[1:3]);
println("rowval[1:3]", K.rowval[1:3]);
println("nzval[1:3]", K.nzval[1:3]);

out = ccall((:amd,lib),Float64,
    (Ref{CscMatrixRs},),M)



