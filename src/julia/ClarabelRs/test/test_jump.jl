using Clarabel, JuMP, SparseArrays
using ClarabelRs

# this solves the problem using Clarabel.jl

model = JuMP.Model(Clarabel.Optimizer)
set_optimizer_attribute(model, "verbose", true)

@variable(model, x[1:2])
@constraint(model, x[1] == 2x[2])
@constraint(model,  -1 .<= x .<= 1)
@objective(model, Min, 3x[1]^2 + 2x[2]^2 - x[1] - 4x[2])

optimize!(model)


# this solves the problem using Clarabel.rs

model = JuMP.Model(ClarabelRs.Optimizer)
set_optimizer_attribute(model, "verbose", true)

@variable(model, x[1:2])
@constraint(model, x[1] == 2x[2])
@constraint(model,  -1 .<= x .<= 1)
@objective(model, Min, 3x[1]^2 + 2x[2]^2 - x[1] - 4x[2])

optimize!(model)