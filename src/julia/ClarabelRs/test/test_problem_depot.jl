using Convex, ClarabelRs, Test



if false 
#tests everything 
    @testset "ClarabelRs" begin
    Convex.ProblemDepot.run_tests([r""]; exclude=[r"mip"]) do p
        solve!(p, Convex.MOI.OptimizerWithAttributes(
            ClarabelRs.Optimizer, "verbose" => true, 
            "chordal_decomposition_compact" => true,
            "chordal_decomposition_enable" => true,
            "chordal_decomposition_merge_method" => :clique_graph,
            "chordal_decomposition_complete_dual" => true,
            ))
    end
end

end 

@testset "ClarabelRs" begin
    Convex.ProblemDepot.run_tests([r"sdp_lieb_ando"]; exclude=[r"mip"]) do p
        solve!(p, Convex.MOI.OptimizerWithAttributes(
            Clarabel.Optimizer, "verbose" => true, 
            "chordal_decomposition_compact" => true,
            "chordal_decomposition_enable" => true,
            "chordal_decomposition_merge_method" => :clique_graph,
            "chordal_decomposition_complete_dual" => true,
            ))
    end
end
