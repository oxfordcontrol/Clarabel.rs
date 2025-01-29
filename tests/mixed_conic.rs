#![allow(non_snake_case)]

use clarabel::{algebra::*, solver::*};

#[test]
fn test_mixed_conic_feasible() {
    // solves a problem with a mix of symmetric and asymmetric
    // cones.   This exercises the barrier methods and unit
    // initializations of the symmetric cones

    let n = 3;
    let P = CscMatrix::<f64>::identity(3);
    let c = vec![1., 1., 1.];

    let I = CscMatrix::<f64>::identity(3);

    // put a 3 dimensional vector into the composition of multiple
    // cones, all with b = 0 on the RHS
    let cones = vec![
        ZeroConeT(3),
        NonnegativeConeT(3),
        SecondOrderConeT(3),
        PowerConeT(0.5),
        ExponentialConeT(),
    ];

    let A = CscMatrix::vcat(&I, &I).unwrap();
    let A = CscMatrix::vcat(&A, &A).unwrap();
    let A = CscMatrix::vcat(&A, &I).unwrap(); // produces 5 stacked copies of I

    let b = vec![0.; 5 * n];

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);
    assert!(f64::abs(solver.info.cost_primal - 0.) <= 1e-8);
}
