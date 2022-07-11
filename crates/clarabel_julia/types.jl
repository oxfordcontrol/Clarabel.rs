using SparseArrays, BenchmarkTools
using Random

struct VectorRs{T<:Real}

    p::Ptr{T}
    len::UInt64 

    function VectorRs(v::Vector{T}) where{T}
        p = Base.unsafe_convert(Ptr{T},v)
        len = length(v)
        new{T}(p,len)
    end
end 

#NB: mutability here would cause alignment errors
struct CscMatrixRs
    m::UInt64
    n::UInt64
    colptr::VectorRs{Int64}
    rowval::VectorRs{Int64}
    nzval::VectorRs{Float64}

    function CscMatrixRs(M::SparseMatrixCSC{Float64, Int64}) where{T}
        m = M.m 
        n = M.n 
        colptr = VectorRs(M.colptr)
        rowval = VectorRs(M.rowval)
        nzval  = VectorRs(M.nzval)
        new(m,n,colptr,rowval,nzval)
    end
end 
