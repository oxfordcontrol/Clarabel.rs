# The types defined here are for exchanging data
# between Rust and Julia.   

using Clarabel

struct VectorJLRS{T} 

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

# it is not obvious at all how to pass data through 
# ccall for a data-carrying enum type in rust.   This 
# makes it very difficult to pass the `cones` object 
# directly.   Here we make an enum for the different 
# cone types, with a complementary enum type on 
# the rust side with equivalent base types and values 
# We then pass a data structure that looks something 
# like a tagged union 

#NB: mutability here causes alignment errors?
 struct ConeDataJLRS
    tag::UInt8
    int::UInt64
    float::Float64
    vec::VectorJLRS{Float64}
    function ConeDataJLRS(enum; int = 0,float = 0.0,vec = Float64[])
        return new(UInt8(enum),int,float,VectorJLRS(vec))
    end
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
    obj_val_dual::Float64
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
        sol.obj_val_dual,
        sol.solve_time,
        sol.iterations,
        sol.r_prim,
        sol.r_dual
    )
end

struct LinearSolverInfoJLRS
    name::VectorJLRS{UInt8}
    threads::UInt64
    direct::Bool
    nnzA::UInt64 
    nnzL::UInt64 
end

function LinearSolverInfo(info::LinearSolverInfoJLRS)

    Clarabel.LinearSolverInfo(
        Symbol(String(Vector(info.name))),
        info.threads,
        info.direct,
        info.nnzA,
        info.nnzL
    )
end


mutable struct DefaultInfoJLRS

    mu::Float64
    sigma::Float64
    step_length::Float64
    iterations::UInt32
    cost_primal::Float64
    cost_dual::Float64
    res_primal::Float64
    res_dual::Float64
    res_primal_inf::Float64
    res_dual_inf::Float64
    gap_abs::Float64
    gap_rel::Float64
    ktratio::Float64

    # previous iterate
    prev_cost_primal::Float64
    prev_cost_dual::Float64
    prev_res_primal::Float64
    prev_res_dual::Float64
    prev_gap_abs::Float64
    prev_gap_rel::Float64

    solve_time::Float64
    status::Clarabel.SolverStatus

    # linear solver information
    linsolver::LinearSolverInfoJLRS

end

function DefaultInfo(info::DefaultInfoJLRS)
    Clarabel.DefaultInfo{Float64}(
        info.mu,
        info.sigma,
        info.step_length,
        info.iterations,
        info.cost_primal,
        info.cost_dual,
        info.res_primal,
        info.res_dual,
        info.res_primal_inf,
        info.res_dual_inf,
        info.gap_abs,
        info.gap_rel,
        info.ktratio,
        info.prev_cost_primal,
        info.prev_cost_dual,
        info.prev_res_primal,
        info.prev_res_dual,
        info.prev_gap_abs,
        info.prev_gap_rel,
        info.solve_time,
        info.status,
        LinearSolverInfo(info.linsolver)
    )
end


@enum ConeEnumJLRS::UInt8 begin
    ZeroConeT = 0
    NonnegativeConeT = 1
    SecondOrderConeT = 2
    ExponentialConeT = 3
    PowerConeT       = 4
    GenPowerConeT    = 5
    PSDTriangleConeT = 6
end


mutable struct Solver{T <: Float64} <: Clarabel.AbstractSolver{Float64}
    ptr:: Ptr{Cvoid}

    function Solver{T}(ptr) where T

        if ptr == C_NULL
            throw(ErrorException("Solver constructor failed"))
        end

        obj = new(ptr)
        finalizer(solver_drop_jlrs,obj)
        return obj
    end

end