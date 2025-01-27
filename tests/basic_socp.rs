#![allow(non_snake_case)]

use clarabel::{algebra::*, solver::*};

#[allow(clippy::type_complexity)]
fn basic_socp_data() -> (
    CscMatrix<f64>,
    Vec<f64>,
    CscMatrix<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
) {
    // P matrix data taken from corresponding Julia unit test.
    // These nzvals form a 3x3 positive definite matrix
    let nzval = vec![
        1.4652521089139698,
        0.6137176286085666,
        -1.1527861771130112,
        0.6137176286085666,
        2.219109946678485,
        -1.4400420548730628,
        -1.1527861771130112,
        -1.4400420548730628,
        1.6014483534926371,
    ];

    let P = CscMatrix::new(
        3,                               // m
        3,                               // n
        vec![0, 3, 6, 9],                // colptr
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2], // rowval
        nzval,                           // nzval
    );

    // A = [2I;-2I;I]
    let I1 = CscMatrix::<f64>::identity(3);
    let mut I2 = CscMatrix::<f64>::identity(3);
    I2.negate();
    let mut A = CscMatrix::vcat(&I1, &I2).unwrap();
    A.scale(2.);
    let A = CscMatrix::vcat(&A, &I1).unwrap();

    let c = vec![0.1, -2.0, 1.0];
    let b = vec![1., 1., 1., 1., 1., 1., 0., 0., 0.];

    let cones = vec![
        NonnegativeConeT(3),
        NonnegativeConeT(3),
        SecondOrderConeT(3),
    ];

    (P, c, A, b, cones)
}

#[test]
fn test_socp_feasible() {
    let (P, c, A, b, cones) = basic_socp_data();

    let settings = DefaultSettings::<f64>::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);

    let refsol = vec![-0.5, 0.435603, -0.245459];
    assert!(solver.solution.x.dist(&refsol) <= 1e-4);

    let refobj = -8.4590e-01;
    assert!(f64::abs(solver.solution.obj_val - refobj) <= 1e-4);
    assert!(f64::abs(solver.solution.obj_val_dual - refobj) <= 1e-4);
}

#[test]
fn test_socp_feasible_sparse() {
    // same data, but with one SOC cone so that we get the
    // sparse representation for code coverage
    let (P, c, A, b, _) = basic_socp_data();
    let cones = vec![NonnegativeConeT(3), SecondOrderConeT(6)];

    let settings = DefaultSettings::<f64>::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);
}

#[test]
fn test_socp_infeasible() {
    let (P, c, A, mut b, cones) = basic_socp_data();

    //make the cone constraint unsatisfiable
    b[6] = -10.;

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::PrimalInfeasible);
    assert!(solver.solution.obj_val.is_nan());
    assert!(solver.solution.obj_val_dual.is_nan());
}
