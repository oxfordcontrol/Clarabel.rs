#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]
#![cfg(feature = "sdp")]
use clarabel::{algebra::*, solver::*};

fn basic_sdp_data() -> (
    CscMatrix<f64>,
    Vec<f64>,
    CscMatrix<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
) {
    //  problem will be 3x3, so upper triangle
    //  of problem data has 6 entries

    let P = CscMatrix::identity(6);

    // A = [1. 1;1 0; 0 1]; A = [-A;A]
    let A = CscMatrix::identity(6);

    let c = vec![0.0; 6];
    let b = vec![-3., 1., 4., 1., 2., 5.];

    let cones = vec![PSDTriangleConeT(3)];

    (P, c, A, b, cones)
}

fn basic_sdp_solution() -> (Vec<f64>, f64) {
    let refsol = vec![
        -3.0729833267361095,
        0.3696004167288786,
        -0.022226685581313674,
        0.31441213129613066,
        -0.026739700851545107,
        -0.016084530571308823,
    ];
    let refobj = 4.840076866013861;

    (refsol, refobj)
}

#[test]
fn test_sdp_feasible() {
    let (P, c, A, b, cones) = basic_sdp_data();
    let (refsol, refobj) = basic_sdp_solution();

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);
    assert!(solver.solution.x.dist(&refsol) <= 1e-6);
    assert!(f64::abs(solver.info.cost_primal - refobj) <= 1e-6);
}

#[test]
fn test_sdp_empty_cone() {
    let (P, c, A, b, mut cones) = basic_sdp_data();
    let (refsol, refobj) = basic_sdp_solution();

    cones.append(&mut vec![PSDTriangleConeT(0)]);

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);
    assert!(solver.solution.x.dist(&refsol) <= 1e-6);
    assert!(f64::abs(solver.info.cost_primal - refobj) <= 1e-6);
}

#[test]
fn test_sdp_primal_infeasible() {
    let (P, c, A, mut b, mut cones) = basic_sdp_data();

    // this adds a negative definiteness constraint to x
    let mut A2 = A.clone();
    A2.negate();
    let A = CscMatrix::vcat(&A, &A2).unwrap();
    b.extend(vec![0.0; b.len()]);
    cones.extend([cones[0].clone()]);

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::PrimalInfeasible);
}
