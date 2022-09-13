using Clarabel, JuMP, SparseArrays
using ClarabelRs

#some simple test problems in different cones 

function test_socp(model)
    @variable(model, x[1:2])
    @constraint(model, x[1] == 2x[2])
    @constraint(model,  -1 .<= x .<= 1)
    @objective(model, Min, 3x[1]^2 + 2x[2]^2 - x[1] - 4x[2])
    return model
end 

function test_expcone(model)
    @variable(model, x[1:3])
    @constraint(model, c1, x[2] == 1)
    @constraint(model, c2, x[3] == exp(5.))
    @constraint(model, c3, x in MOI.ExponentialCone())
    @objective(model, Max, x[1])
    return model
end 

# Solve the problems through both interfaces 

optimizers = [Clarabel.Optimizer, ClarabelRs.Optimizer]
test_problems = [test_expcone, test_socp]

for optimizer in optimizers 

    for test_problem in test_problems 

        model = JuMP.Model(optimizer)
        set_optimizer_attribute(model, "verbose", true)
        model = test_problem(model)
        optimize!(model)

    end 
end 
