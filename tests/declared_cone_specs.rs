#![allow(non_snake_case)]
//! End-to-end coverage for `DefaultSolution::declared_cone_specs` and the
//! `declared_*` accessors: the caller's cone declarations must survive the
//! internal collapse pass, even though the solver still solves the collapsed
//! problem.

use clarabel::{algebra::*, solver::*};

/// Trivial strictly-convex problem `min 1/2 x'x  s.t.  b - x in K`, with
/// `m` slack variables.  The cone list is whatever the caller supplies, so
/// long as its total `nvars()` is `m`.
fn solve_with(cones: &[SupportedConeT<f64>], m: usize) -> DefaultSolver<f64> {
    let P = CscMatrix::identity(m);
    let A = CscMatrix::identity(m);
    let c = vec![0.0; m];
    // A diagonally-dominant b keeps every PSD block comfortably interior.
    let b: Vec<f64> = (0..m).map(|i| 4.0 + (i as f64) * 0.5).collect();

    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, cones, settings).unwrap();
    solver.solve();
    solver
}

// -------------------------------------------------------------------
// The defect: a declared 1x1 PSD cone is collapsed to NonnegativeConeT(1)
// and is no longer recoverable as a PSD block from the collapsed list.
// -------------------------------------------------------------------

#[cfg(feature = "sdp")]
#[test]
fn test_declared_psd_singleton_is_recoverable() {
    // triu(2x2) = 3 slacks, plus 1 for the 1x1 block.
    let cones = vec![PSDTriangleConeT(2), PSDTriangleConeT(1)];
    let solver = solve_with(&cones, 4);
    let sol = &solver.solution;
    assert_eq!(sol.status, SolverStatus::Solved);

    // --- the collapsed view (unchanged behaviour) ---
    assert_eq!(sol.cone_specs.len(), 2);
    assert_eq!(sol.cone_specs[0].tag, SupportedConeTag::PSDTriangleCone);
    // the 1x1 PSD cone became a nonnegative cone
    assert_eq!(sol.cone_specs[1].tag, SupportedConeTag::NonnegativeCone);
    assert!(sol.dual_psd_block(1).is_none());

    // --- the declared view (what this PR adds) ---
    assert_eq!(sol.declared_cone_specs.len(), 2);
    assert_eq!(
        sol.declared_cone_specs[0].tag,
        SupportedConeTag::PSDTriangleCone
    );
    assert_eq!(
        sol.declared_cone_specs[1].tag,
        SupportedConeTag::PSDTriangleCone
    );
    assert_eq!(sol.declared_cone_specs[1].dim, 1);
    assert_eq!(sol.declared_cone_specs[1].range, 3..4);
    assert_eq!(sol.declared_cone_specs[1].input_index, 1);

    let blk = sol
        .declared_dual_psd_block(1)
        .expect("declared PSD(1) must be readable as a 1x1 PSD block");
    assert_eq!(blk.len(), 1);
    assert_eq!(blk[0].len(), 1);
    // A 1x1 svec has no off-diagonal scaling, so it is the raw z entry.
    assert_eq!(blk[0][0], sol.z[3]);

    let pblk = sol.declared_primal_psd_block(1).unwrap();
    assert_eq!(pblk[0][0], sol.s[3]);

    // The 2x2 block agrees between the two views, since nothing collapsed it.
    assert_eq!(
        sol.declared_dual_psd_block(0).unwrap(),
        sol.dual_psd_block(0).unwrap()
    );
}

// -------------------------------------------------------------------
// The index-shift half of the defect: merging a run of cones renumbers
// everything after it in the collapsed list.
// -------------------------------------------------------------------

#[cfg(feature = "sdp")]
#[test]
fn test_declared_indices_survive_a_merge() {
    // [PSD(1), Nonneg(2)] collapses to the single cone [Nonneg(3)].
    let cones = vec![PSDTriangleConeT(1), NonnegativeConeT(2)];
    let solver = solve_with(&cones, 3);
    let sol = &solver.solution;
    assert_eq!(sol.status, SolverStatus::Solved);

    // collapsed: two declared cones became one
    assert_eq!(sol.cone_specs.len(), 1);
    assert_eq!(sol.cone_specs[0].tag, SupportedConeTag::NonnegativeCone);
    assert_eq!(sol.cone_specs[0].range, 0..3);

    // declared: both cones still there, at the positions the caller used
    assert_eq!(sol.declared_cone_specs.len(), 2);
    assert_eq!(
        sol.declared_cone_specs[0].tag,
        SupportedConeTag::PSDTriangleCone
    );
    assert_eq!(sol.declared_cone_specs[0].range, 0..1);
    assert_eq!(
        sol.declared_cone_specs[1].tag,
        SupportedConeTag::NonnegativeCone
    );
    assert_eq!(sol.declared_cone_specs[1].range, 1..3);

    assert!(sol.declared_dual_psd_block(0).is_some());
    // cone 1 was declared nonnegative, so it is not a PSD block
    assert!(sol.declared_dual_psd_block(1).is_none());
    assert_eq!(sol.declared_dual_block(1).unwrap(), &sol.z[1..3]);
}

#[test]
fn test_declared_indices_survive_a_merge_without_sdp() {
    // Same shape, no sdp feature required: [SOC(1), Nonneg(2)] -> [Nonneg(3)].
    let cones = vec![SecondOrderConeT(1), NonnegativeConeT(2)];
    let solver = solve_with(&cones, 3);
    let sol = &solver.solution;
    assert_eq!(sol.status, SolverStatus::Solved);

    assert_eq!(sol.cone_specs.len(), 1);
    assert_eq!(sol.declared_cone_specs.len(), 2);
    assert_eq!(
        sol.declared_cone_specs[0].tag,
        SupportedConeTag::SecondOrderCone
    );
    assert_eq!(sol.declared_cone_specs[0].range, 0..1);
    assert_eq!(sol.declared_cone_specs[1].range, 1..3);
    assert_eq!(sol.declared_dual_block(1).unwrap(), &sol.z[1..3]);
}

// -------------------------------------------------------------------
// BlockDiagPSDConeT: per-block declared entries, grouped by input index.
// -------------------------------------------------------------------

#[cfg(feature = "sdp")]
#[test]
fn test_declared_block_diag_blocks_are_individually_addressable() {
    // triu(2) + triu(3) + triu(1) = 3 + 6 + 1 = 10
    let cones = vec![BlockDiagPSDConeT {
        block_dims: vec![2, 3, 1],
    }];
    let solver = solve_with(&cones, 10);
    let sol = &solver.solution;
    assert_eq!(sol.status, SolverStatus::Solved);

    assert_eq!(sol.declared_cone_specs.len(), 3);
    for spec in &sol.declared_cone_specs {
        assert_eq!(spec.tag, SupportedConeTag::PSDTriangleCone);
        // every block traces back to the single BlockDiag the caller wrote
        assert_eq!(spec.input_index, 0);
    }
    assert_eq!(sol.declared_cone_specs[0].range, 0..3);
    assert_eq!(sol.declared_cone_specs[1].range, 3..9);
    assert_eq!(sol.declared_cone_specs[2].range, 9..10);
    assert_eq!(sol.declared_positions_for_input(0), vec![0, 1, 2]);

    assert_eq!(sol.declared_dual_psd_block(0).unwrap().len(), 2);
    assert_eq!(sol.declared_dual_psd_block(1).unwrap().len(), 3);
    // the trailing 1x1 block -- the one the collapse pass rewrites away
    assert_eq!(sol.declared_dual_psd_block(2).unwrap().len(), 1);
    assert!(sol.dual_psd_block(2).is_none());
}

// -------------------------------------------------------------------
// Task B: an empty block_dims is a documented no-op.
// -------------------------------------------------------------------

#[cfg(feature = "sdp")]
#[test]
fn test_empty_block_dims_is_a_noop() {
    let with_empty = vec![
        PSDTriangleConeT(2),
        BlockDiagPSDConeT { block_dims: vec![] },
        PSDTriangleConeT(1),
    ];
    let without = vec![PSDTriangleConeT(2), PSDTriangleConeT(1)];

    let a = solve_with(&with_empty, 4);
    let b = solve_with(&without, 4);

    assert_eq!(a.solution.status, SolverStatus::Solved);
    assert_eq!(a.solution.x, b.solution.x);
    assert_eq!(a.solution.z, b.solution.z);
    assert_eq!(a.solution.s, b.solution.s);

    // The empty declaration contributes no declared cones, so the two
    // declared views are identical apart from the input indices.
    assert_eq!(a.solution.declared_cone_specs.len(), 2);
    assert!(a.solution.declared_positions_for_input(1).is_empty());
    assert_eq!(a.solution.declared_cone_specs[1].input_index, 2);
    assert_eq!(b.solution.declared_cone_specs[1].input_index, 1);
}

// -------------------------------------------------------------------
// The structural invariant the declared ranges rely on.
// -------------------------------------------------------------------

#[test]
fn test_declared_ranges_tile_the_slack_vector_and_refine_collapsed() {
    let cases: Vec<(Vec<SupportedConeT<f64>>, usize)> = vec![
        (vec![NonnegativeConeT(3), NonnegativeConeT(2)], 5),
        (vec![SecondOrderConeT(1), NonnegativeConeT(3)], 4),
        (
            vec![
                NonnegativeConeT(2),
                SecondOrderConeT(3),
                NonnegativeConeT(0),
                NonnegativeConeT(1),
            ],
            6,
        ),
    ];

    for (cones, m) in cases {
        let solver = solve_with(&cones, m);
        let sol = &solver.solution;
        assert_eq!(sol.status, SolverStatus::Solved, "{:?}", cones);

        // declared ranges tile [0, m) contiguously and in order
        let mut cursor = 0;
        for spec in &sol.declared_cone_specs {
            assert_eq!(spec.range.start, cursor, "{:?}", cones);
            cursor = spec.range.end;
        }
        assert_eq!(cursor, m, "{:?}", cones);
        assert_eq!(sol.declared_cone_specs.len(), cones.len(), "{:?}", cones);

        // and each non-empty declared range lies inside exactly one collapsed range
        for spec in &sol.declared_cone_specs {
            if spec.range.is_empty() {
                continue;
            }
            let n = sol
                .cone_specs
                .iter()
                .filter(|cs| cs.range.start <= spec.range.start && spec.range.end <= cs.range.end)
                .count();
            assert_eq!(n, 1, "declared {:?} for {:?}", spec.range, cones);
        }
    }
}

// -------------------------------------------------------------------
// Presolve interaction.  The solution's s/z are returned at the original
// (pre-presolve) length, so the declared ranges -- which are computed from
// the caller's own cone list and never see the presolve -- stay valid.
// -------------------------------------------------------------------

#[test]
fn test_declared_ranges_valid_under_presolve() {
    let n = 3;
    let P = CscMatrix::identity(n);
    let mut A2 = CscMatrix::identity(n);
    A2.negate();
    let mut A = CscMatrix::vcat(&P, &A2).unwrap();
    A.scale(2.);

    let c = vec![3., -2., 1.];
    let mut b = vec![1.; 2 * n];
    b[3] = 1e30_f64; // triggers the presolve reduction

    let cones = vec![NonnegativeConeT(3), NonnegativeConeT(3)];

    let mut settings = DefaultSettings::default();
    settings.verbose = false;
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();
    solver.solve();
    let sol = &solver.solution;
    assert_eq!(sol.status, SolverStatus::Solved);

    // presolve dropped one row internally, but the solution is un-reduced
    assert_eq!(solver.variables.z.len(), 5);
    assert_eq!(sol.z.len(), 6);

    // the declared view describes the caller's cones against the full vectors
    assert_eq!(sol.declared_cone_specs.len(), 2);
    assert_eq!(sol.declared_cone_specs[0].range, 0..3);
    assert_eq!(sol.declared_cone_specs[1].range, 3..6);
    assert_eq!(sol.declared_dual_block(1).unwrap().len(), 3);
    assert_eq!(sol.declared_primal_block(1).unwrap(), &sol.s[3..6]);
}

// -------------------------------------------------------------------
// API compatibility: adding `declared_cone_specs` must not change the
// serialized shape of `ConeSpec`, and must not break deserialization of
// solutions written before this field existed.
// -------------------------------------------------------------------

#[cfg(feature = "serde")]
#[test]
fn test_pre_existing_solution_json_still_deserializes() {
    // A DefaultSolution as it would have been serialized before this PR:
    // `cone_specs` present, `declared_cone_specs` absent.
    let old = r#"{
        "x":[1.0],"z":[2.0],"s":[3.0],
        "status":"Solved","obj_val":1.0,"obj_val_dual":1.0,
        "solve_time":0.0,"iterations":3,"r_prim":0.0,"r_dual":0.0,
        "cone_specs":[{"tag":"NonnegativeCone","range":{"start":0,"end":1},"dim":1}]
    }"#;
    let sol: DefaultSolution<f64> =
        serde_json::from_str(old).expect("pre-PR solution JSON must still deserialize");
    assert_eq!(sol.cone_specs.len(), 1);
    // the new field defaults to empty rather than failing the parse
    assert_eq!(sol.declared_cone_specs.len(), 0);
}

#[cfg(feature = "serde")]
#[test]
fn test_cone_spec_wire_shape_is_unchanged() {
    // ConeSpec deliberately did NOT gain a field: its serialized form has to
    // round-trip byte-identically for anyone with stored witnesses.
    let old = r#"{"tag":"NonnegativeCone","range":{"start":0,"end":4},"dim":4}"#;
    let cs: ConeSpec = serde_json::from_str(old).expect("ConeSpec shape must be unchanged");
    assert_eq!(serde_json::to_string(&cs).unwrap(), old);
}
