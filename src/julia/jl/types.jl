using SparseArrays, BenchmarkTools
using Random

# The types defined here are for exchanging CSC matrix and 
# dense vector data between Rust and Julia.   

struct VectorJLRS{T <: Real}

    p::Ptr{T}
    len::UInt64 

    function VectorJLRS(v::Vector{T}) 
        p = Base.unsafe_convert(Ptr{T},v)
        len = length(v)
        new{T}(p,len)
    end
end 

#NB: mutability here would cause alignment errors
struct CscMatrixJLRS

    m::UInt64
    n::UInt64
    colptr::VectorJLRS{Int64}
    rowval::VectorJLRS{Int64}
    nzval::VectorJLRS{Float64}

    function CscMatrixJLRS(M::SparseMatrixCSC{Float64, Int64})
        m = M.m 
        n = M.n 
        colptr = VectorJLRS(M.colptr)
        rowval = VectorJLRS(M.rowval)
        nzval  = VectorJLRS(M.nzval)
        new(m,n,colptr,rowval,nzval)
    end
end 
