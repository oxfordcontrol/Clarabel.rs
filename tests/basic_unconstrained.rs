#![allow(non_snake_case)]

use clarabel::{algebra::*, solver::*};

#[test]
fn test_unconstrained_feasible() {
    let P = CscMatrix::identity(3);
    let mut c = [1., 2., -3.];
    let A = CscMatrix::zeros((0, 3)); // <- no constraints
    let b = [];
    let cones = [];

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();

    let refsol = c.negate();
    assert!(solver.solution.x.dist(refsol) <= 1e-6);
    assert_eq!(solver.solution.status, SolverStatus::Solved);
}

#[test]
fn test_unconstrained_dual_infeasible() {
    let P = CscMatrix::zeros((3, 3));
    let c = [1., 0., 0.];
    let A = CscMatrix::zeros((0, 3)); // <- no constraints
    let b = [];
    let cones = [];

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::DualInfeasible);
}
