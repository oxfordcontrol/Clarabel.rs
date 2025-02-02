#![allow(non_snake_case)]

use clarabel::{algebra::*, solver::*};

#[test]
fn test_powcone() {
    // solve the following power cone problem
    // max  x1^0.6 y^0.4 + x2^0.1
    // s.t. x1, y, x2 >= 0
    //      x1 + 2y  + 3x2 == 3
    // which is equivalent to
    // max z1 + z2
    // s.t. (x1, y, z1) in K_pow(0.6)
    //      (x2, 1, z2) in K_pow(0.1)
    //      x1 + 2y + 3x2 == 3

    // x = (x1, y, z1, x2, y2, z2)

    let n = 6;
    let P = CscMatrix::<f64>::zeros((n, n));
    let c = vec![0., 0., -1., 0., 0., -1.];

    // (x1, y, z1) in K_pow(0.6)
    // (x2, y2, z2) in K_pow(0.1)
    let mut A1 = CscMatrix::<f64>::identity(n);
    A1.negate();
    let b1 = vec![0.; n];
    let cones1 = vec![PowerConeT(0.6), PowerConeT(0.1)];

    // x1 + 2y + 3x2 == 3
    // y2 == 1
    let A2 = CscMatrix::from(&[
        [1., 2., 0., 3., 0., 0.], //
        [0., 0., 0., 0., 1., 0.], //
    ]);
    let b2 = vec![3., 1.];
    let cones2 = vec![ZeroConeT(2)];

    let A = CscMatrix::vcat(&A1, &A2).unwrap();
    let b = [b1, b2].concat();
    let cones = [cones1, cones2].concat();

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);

    let refobj = -1.8458;
    assert!(f64::abs(solver.info.cost_primal - refobj) <= 1e-3);
}
