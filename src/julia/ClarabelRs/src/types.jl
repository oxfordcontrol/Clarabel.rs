# The types defined here are for exchanging data
# between Rust and Julia.   

struct VectorJLRS{T<:Real} 

    p::Ptr{T}
    len::UInt64 

    function VectorJLRS(v::Vector{T}) where {T}
        len = length(v)
        p   = Base.unsafe_convert(Ptr{T},v)
        new{T}(p,len)
    end
end 

function Vector(v::VectorJLRS{T}) where {T}
    unsafe_wrap(Vector{T},v.p,v.len)
end


#NB: mutability here causes alignment errors?
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


struct SolutionJLRS

    x::VectorJLRS{Float64}
    z::VectorJLRS{Float64}
    s::VectorJLRS{Float64}
    status::UInt32   #0 indexed enum in RS/JL
    obj_val::Float64
    solve_time::Float64
    iterations::UInt32
    r_prim::Float64
    r_dual::Float64

end 


function DefaultSolution(sol::SolutionJLRS)

    Clarabel.DefaultSolution{Float64}(
        Vector(sol.x),
        Vector(sol.z),
        Vector(sol.s),
        Clarabel.SolverStatus(sol.status),
        sol.obj_val,
        sol.solve_time,
        sol.iterations,
        sol.r_prim,
        sol.r_dual
    )

end


@enum ConeEnumJLRS::UInt8 begin
    ZeroConeT = 0
    NonnegativeConeT = 1
    SecondOrderConeT = 2
    ExponentialConeT = 3
    PowerConeT       = 4
end


mutable struct Solver{T <: Float64} <: Clarabel.AbstractSolver{Float64}
    ptr:: Ptr{Cvoid}

    function Solver(ptr)
        obj = new(ptr)
        finalizer(solver_drop_jlrs,obj)
        return obj
    end

end