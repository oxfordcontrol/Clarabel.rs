using SparseArrays
using Clarabel
using TimerOutputs

include("./types.jl")
push!(Base.DL_LOAD_PATH,"../../target/debug/")
lib = "libclarabel_julia.dylib"

function solve_rs(P,q,A,b,cones)

    solve_time = ccall((:solve,lib),Float64,
        (Ref{CscMatrixRs},Ref{VectorRs},Ref{CscMatrixRs},Ref{VectorRs}),
        CscMatrixRs(P), VectorRs(q),CscMatrixRs(A), VectorRs(b))

    return solve_time
end 