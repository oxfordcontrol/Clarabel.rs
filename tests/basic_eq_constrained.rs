#![allow(non_snake_case)]

use clarabel::{algebra::*, solver::*};

#[allow(clippy::type_complexity)]
fn eq_constrained_A1() -> CscMatrix<f64> {
    // A =
    //[ 0. 1.  1.;
    //  0. 1. -1.]
    CscMatrix::new(
        2,                     // m
        3,                     // n
        vec![0, 0, 2, 4],      //colptr
        vec![0, 1, 0, 1],      //rowval
        vec![1., 1., 1., -1.], //nzva;
    )
}
fn eq_constrained_A2() -> CscMatrix<f64> {
    // A = [
    // 0    1.0   1.0;
    // 0    1.0  -1.0;
    //1.0   2.0  -1.0l
    //2.0  -1.0   3.0l
    //]
    CscMatrix::new(
        4,                                               // m
        3,                                               // n
        vec![0, 2, 6, 10],                               //colptr
        vec![2, 3, 0, 1, 2, 3, 0, 1, 2, 3],              //rowval
        vec![1., 2., 1., 1., 2., -1., 1., -1., -1., 3.], //nzva;
    )
}

#[test]
fn test_eq_constrained_feasible() {
    let P = CscMatrix::identity(3);
    let c = [0., 0., 0.];
    let A = eq_constrained_A1(); // <- two constraints
    let b = [2., 0.];
    let cones = [ZeroConeT(2)];

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();
    let refsol = [0., 1., 1.];
    assert_eq!(solver.solution.status, SolverStatus::Solved);
    assert!(solver.solution.x.dist(&refsol) <= 1e-6);
}

#[test]
fn test_eq_constrained_primal_infeasible() {
    let P = CscMatrix::identity(3);
    let c = [0.; 3];
    let A = eq_constrained_A2(); // <- 4 constraints, 3 vars
    let b = [1.; 4];
    let cones = [ZeroConeT(4)];

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::PrimalInfeasible);
}

#[test]
fn test_eq_constrained_dual_infeasible() {
    let mut P = CscMatrix::identity(3);
    P.nzval[0] = 0.;
    let c = [1.; 3];
    let A = eq_constrained_A1(); // <- two constraints
    let b = [2., 0.];
    let cones = [ZeroConeT(2)];

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::DualInfeasible);
}
