using Convex, ClarabelRs, Test
@testset "ClarabelRs" begin
    Convex.ProblemDepot.run_tests([r""]; exclude=[r"mip", r"sdp"]) do p
        solve!(p, Convex.MOI.OptimizerWithAttributes(ClarabelRs.Optimizer, "verbose" => true))
    end
end
