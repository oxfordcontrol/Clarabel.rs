

# here we just hijack the Clarabel.jl MOI wrapper by passing 
# a different module name into the constructor 

Optimizer(args...;kwargs...) = Clarabel.Optimizer{Float64}(args...;solver_module = ClarabelRs, kwargs...)