#![allow(non_snake_case)]

use clarabel::{algebra::*, solver::*};

#[allow(clippy::type_complexity)]
fn basic_lp_data() -> (
    CscMatrix<f64>,
    Vec<f64>,
    CscMatrix<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
) {
    let P = CscMatrix::<f64>::zeros((3, 3));

    let I1 = CscMatrix::<f64>::identity(3);
    let mut I2 = CscMatrix::<f64>::identity(3);
    I2.negate();
    let mut A = CscMatrix::vcat(&I1, &I2).unwrap();
    A.scale(2.);

    let c = vec![3., -2., 1.];
    let b = vec![1.; 6];

    let cones = vec![NonnegativeConeT(3), NonnegativeConeT(3)];

    (P, c, A, b, cones)
}

#[test]
fn test_lp_feasible() {
    let (P, c, A, b, cones) = basic_lp_data();

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);

    let refsol = vec![-0.5, 0.5, -0.5];
    assert!(solver.solution.x.dist(&refsol) <= 1e-8);

    let refobj = -3.;
    assert!(f64::abs(solver.solution.obj_val - refobj) <= 1e-8);
    assert!(f64::abs(solver.solution.obj_val_dual - refobj) <= 1e-8);
}

#[test]
fn test_lp_primal_infeasible() {
    let (P, c, A, mut b, cones) = basic_lp_data();

    b[0] = -1.;
    b[3] = -1.;

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::PrimalInfeasible);
    assert!(solver.solution.obj_val.is_nan());
    assert!(solver.solution.obj_val_dual.is_nan());
}

#[test]
fn test_lp_dual_infeasible() {
    let (P, _c, mut A, b, cones) = basic_lp_data();

    A.nzval[1] = 1.; //swap lower bound on first variable to redundant upper bound
    let c = vec![1., 0., 0.];

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::DualInfeasible);
    assert!(solver.solution.obj_val.is_nan());
    assert!(solver.solution.obj_val_dual.is_nan());
}

#[test]
fn test_lp_dual_infeasible_ill_cond() {
    let (P, _c, mut A, b, cones) = basic_lp_data();

    A.nzval[0] = f64::EPSILON;
    A.nzval[1] = 0.0;
    let c = vec![1., 0., 0.];

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::DualInfeasible);
    assert!(solver.solution.obj_val.is_nan());
    assert!(solver.solution.obj_val_dual.is_nan());
}
