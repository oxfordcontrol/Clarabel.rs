#![allow(non_snake_case)]

use clarabel::{algebra::*, solver::*};

#[allow(clippy::type_complexity)]
fn basic_expcone_data() -> (
    CscMatrix<f64>, //P
    Vec<f64>,       //c
    CscMatrix<f64>, //A
    Vec<f64>,       //b
    Vec<SupportedConeT<f64>>,
) {
    // produces data for the following exponential cone problem
    // max  x
    // s.t. y * exp(x / y) <= z
    //      y == 1, z == exp(5)

    let P = CscMatrix::<f64>::zeros((3, 3));
    let c = vec![-1., 0., 0.];

    let mut A1 = CscMatrix::<f64>::identity(3);
    A1.negate();
    let b1 = vec![0.; 3];

    let A2 = CscMatrix::new(
        2,                // m
        3,                // n
        vec![0, 0, 1, 2], //colptr
        vec![0, 1],       //rowval
        vec![1., 1.],     //nzval
    );
    let b2 = vec![1., f64::exp(5.)];

    let A = CscMatrix::vcat(&A1, &A2).unwrap();
    let b = [b1, b2].concat();

    let cones = vec![ExponentialConeT(), ZeroConeT(2)];

    (P, c, A, b, cones)
}

#[test]
fn test_expcone_feasible() {
    // solve the following exponential cone problem
    // max  x
    // s.t. y * exp(x / y) <= z
    //      y == 1, z == exp(5)
    //
    // This is just the default problem data above

    let (P, c, A, b, cones) = basic_expcone_data();

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);

    let refsol = vec![5.0, 1.0, f64::exp(5.0)];
    assert!(solver.solution.x.dist(&refsol) <= 1e-6);

    let refobj = -5.0;
    assert!(f64::abs(solver.info.cost_primal - refobj) <= 1e-6);
}

#[test]
fn test_expcone_primal_infeasible() {
    // solve the following exponential cone problem
    // max  x
    // s.t. y * exp(x / y) <= z
    //      y == 1, z == -1
    //
    // Same as default, but last element of b is different

    let (P, c, A, mut b, cones) = basic_expcone_data();

    b[4] = -1.; //

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::PrimalInfeasible);
}

#[test]
fn test_expcone_dual_infeasible() {
    // solve the following exponential cone problem
    // max  x
    // s.t. y * exp(x / y) <= z
    //
    // Same as default, but no equality constraint

    let P = CscMatrix::<f64>::zeros((3, 3));
    let c = vec![-1., 0., 0.];

    let mut A = CscMatrix::<f64>::identity(3);
    A.negate();
    let b = vec![0.; 3];
    let cones = vec![ExponentialConeT()];

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::DualInfeasible);
}
