# ============================ /test/MOI_wrapper.jl ============================
# Test structure taken from https://jump.dev/JuMP.jl/stable/moi/submodules/Test/overview/

module TestClarabelRs

import ClarabelRs, Clarabel, JuMP
using MathOptInterface
using Test

const MOI = MathOptInterface

T = Float64
optimizer =  JuMP.optimizer_with_attributes(
            ClarabelRs.Optimizer, 
            "chordal_decomposition_enable" => true, 
            "chordal_decomposition_merge_method" => :clique_graph,
            "chordal_decomposition_compact" => true,
            "chordal_decomposition_complete_dual" => true,
       );

const BRIDGED = MOI.instantiate(
    optimizer,
    with_bridge_type = Float64,
)

#


# See the docstring of MOI.Test.Config for other arguments.
MYCONFIG = MOI.Test.Config(
    # Modify tolerances as necessary.
    atol = 1e-4,
    rtol = 1e-4,
    # Use MOI.LOCALLY_SOLVED for local solvers.
    optimal_status = MOI.OPTIMAL,
    # Pass attributes or MOI functions to `exclude` to skip tests that
    # rely on this functionality.
    exclude = Any[MOI.VariableName,
                  MOI.ConstraintName,
                  MOI.VariableBasisStatus,
                  MOI.ConstraintBasisStatus,
                  MOI.delete,
                  MOI.ObjectiveBound,
                  ],
)

"""
    runtests()

This function runs all functions in the this Module starting with `test_`.
"""
function runtests()
    @testset "MOI" begin
        for name in names(@__MODULE__; all = true)
            if startswith("$(name)", "test_")
                @testset "$(name)" begin
                    getfield(@__MODULE__, name)()
                end
            end
        end
    end
end

"""
    test_runtests()

This function runs all the tests in MathOptInterface.Test.

Pass arguments to `exclude` to skip tests for functionality that is not
implemented or that your solver doesn't support.
"""
function test_MOI_standard()

    MOI.Test.runtests(
        BRIDGED,
        MYCONFIG,
        # use `include` to single out a problem class
        #include = String["test_basic_VectorQuadraticFunction_RelativeEntropyCone"],
        exclude = String[
            #these two tests fail intermittently depending on platform 
            #and MOI version.  They both converge to reasonable accuracy.
            #"test_conic_GeometricMeanCone_VectorAffineFunction",
            #"test_constraint_qcp_duplicate_diagonal",
        ],
        # This argument is useful to prevent tests from failing on future
        # releases of MOI that add new tests.
        exclude_tests_after = VersionNumber(Clarabel.moi_version()),
    )
    return
end

"""
    test_SolverName()

You can also write new tests for solver-specific functionality. Write each new
test as a function with a name beginning with `test_`.
"""
function test_SolverName()
    @test MOI.get(ClarabelRs.Optimizer(), MOI.SolverName()) == "ClarabelRs"
    return
end

# Settings don't work like this in the compiled versions
# function test_passing_settings()
#     optimizer = ClarabelRs.Optimizer{T}(; verbose=false)
#     @test optimizer.solver_settings.verbose === false
#     return
# end

end # module TestClarabelRs

# This line at the end of the file runs all the tests!
TestClarabelRs.runtests()
