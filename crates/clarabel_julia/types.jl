using SparseArrays, BenchmarkTools
using Random

push!(Base.DL_LOAD_PATH,"../../target/release/")
lib = "libclarabel_julia.dylib"

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

function printvec(v::Vector{Float64})

    x = VectorRs(v)
    dump(x)
    out = ccall((:printvec_f64,lib),Cvoid,
    (Ref{VectorRs{Float64}},),x)

end
    
function printvec(v::Vector{UInt64})
    
        x = VectorRs(v)
        dump(x)
        out = ccall((:printvec_u64,lib),Cvoid,
        (Ref{VectorRs{UInt64}},),x)

end

function printmat(K::SparseMatrixCSC)
    
    x = CscMatrixRs(K)
    dump(x)
    out = ccall((:printmat,lib),Cvoid,
    (Ref{CscMatrixRs},),x)

end

function amd(K::SparseMatrixCSC)
    
    x = CscMatrixRs(K)
    dump(x)
    out = ccall((:amd,lib),Cvoid,
    (Ref{CscMatrixRs},),x)

end


function testit()

    Random.seed!(1234)
    n = 3000
    m = 2*n;
    P = sprandn(n,n,0.01)
    P = P+P';
    d = sum(abs.(P),dims=1);
    P = P + Diagonal(d[:]);
    P = triu(P);
    A = sprandn(m,n,0.01);

    K = [P A';A -I(m)]

    # println("\n\nStarting\n-----------------\n")
    # println("rowval"); printvec(Vector{UInt64}(K.rowval))
    # println("colptr"); printvec(Vector{UInt64}(K.colptr))
    # println("nzval"); printvec(Vector{Float64}(K.nzval))
    out = ccall((:qdldl,lib),Cvoid,
    (Ref{CscMatrixRs},),CscMatrixRs(K))

    1
end

#Julia SparseArrays dot function is very slow for Symmtric
#matrices.  See https://github.com/JuliaSparse/SparseArrays.jl/issues/83
function symdot(
    x::AbstractArray{Tf},
    A::Symmetric{Tf,SparseMatrixCSC{Tf,Ti}},
    y::AbstractArray{Tf}
    ) where{Tf <: Real,Ti}

    if(A.uplo != 'U')
        error("Only implemented for upper triangular matrices")
    end
    M = A.data

    m, n = size(A)
    (length(x) == m && n == length(y)) || throw(DimensionMismatch())
    if iszero(m) || iszero(n)
        return dot(zero(eltype(x)), zero(eltype(A)), zero(eltype(y)))
    end

    Mc = M.colptr
    Mr = M.rowval
    Mv = M.nzval

    out = zero(Tf)

    @inbounds for j = 1:n    #col number
        tmp1 = zero(Tf)
        tmp2 = zero(Tf)
        for p = Mc[j]:(Mc[j+1]-1)
            i = Mr[p]  #row number
            if (i < j)  #triu terms only
                tmp1 += Mv[p]*x[i]
                tmp2 += Mv[p]*y[i]
            elseif i == j
                out += Mv[p]*x[i]*y[i]
            end
        end
        out += tmp1*y[j] + tmp2*x[j]
    end
    return out
end


function testsym(n = 4000)

    K = sprandn(n,n,0.1);
    K = triu(K)
    x = randn(n)
    y = randn(n)
    #println("Reference result", x'*Symmetric(K)*y)

    out = ccall((:symdot,lib),Float64,
    (Ref{CscMatrixRs},Ref{VectorRs},Ref{VectorRs},),
    CscMatrixRs(K),VectorRs(x),VectorRs(y))

    println("Rust value = ", out)

    Ksym = Symmetric(K)
    #x'*Ksym*y

    println("Julia value = ", symdot(x,Ksym,y))
    @btime symdot($x,$Ksym,$y)


end

function testdot(n = 4000)

    x = randn(n)
    y = randn(n)
    #println("Reference result", x'*Symmetric(K)*y)

    out = ccall((:mydot,lib),Float64,
    (Ref{VectorRs},Ref{VectorRs},),
    VectorRs(x),VectorRs(y))


end