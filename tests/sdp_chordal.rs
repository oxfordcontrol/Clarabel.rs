#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]
#![cfg(feature = "sdp")]

use clarabel::algebra::*;
use clarabel::solver::*;

fn sdp_chordal_data() -> (
    CscMatrix<f64>,
    Vec<f64>,
    CscMatrix<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
) {
    let P = CscMatrix {
        m: 8,
        n: 8,
        colptr: vec![0, 0, 0, 0, 0, 0, 0, 0, 0],
        rowval: vec![],
        nzval: vec![],
    };
    let q = vec![-1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0];
    let A = CscMatrix {
        m: 28,
        n: 8,
        colptr: vec![0, 1, 4, 5, 8, 9, 10, 13, 16],
        rowval: vec![24, 7, 10, 22, 8, 12, 15, 25, 9, 13, 18, 21, 26, 0, 23, 27],
        nzval: vec![
            -1.0,
            -std::f64::consts::SQRT_2,
            -1.0,
            -1.0,
            -std::f64::consts::SQRT_2,
            -std::f64::consts::SQRT_2,
            -1.0,
            -1.0,
            -std::f64::consts::SQRT_2,
            -std::f64::consts::SQRT_2,
            -std::f64::consts::SQRT_2,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
        ],
    };
    let b = vec![
        0.0,
        3.0,
        2. * std::f64::consts::SQRT_2,
        2.0,
        std::f64::consts::SQRT_2,
        std::f64::consts::SQRT_2,
        3.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    let cones = vec![
        NonnegativeConeT(1),
        PSDTriangleConeT(6),
        PowerConeT(0.3333333333333333),
        PowerConeT(0.5),
    ];

    (P, q, A, b, cones)
}

#[test]
fn test_sdp_chordal() {
    let (P, c, A, b, cones) = sdp_chordal_data();

    let mut settings = DefaultSettingsBuilder::default()
        .verbose(true)
        .chordal_decomposition_enable(true)
        .chordal_decomposition_compact(true)
        .chordal_decomposition_complete_dual(true)
        .chordal_decomposition_merge_method("clique_graph".to_string())
        .max_iter(50)
        .build()
        .unwrap();

    for compact in [false, true] {
        for complete_dual in [false, true] {
            for merge_method in ["clique_graph", "parent_child", "none"] {
                settings.chordal_decomposition_compact = compact;
                settings.chordal_decomposition_complete_dual = complete_dual;
                settings.chordal_decomposition_merge_method = merge_method.to_string();
                let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings.clone());
                solver.solve();
                assert_eq!(solver.solution.status, SolverStatus::Solved);
            }
        }
    }
}
