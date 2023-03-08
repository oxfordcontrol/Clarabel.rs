#![allow(non_snake_case)]

use clarabel::{algebra::*, solver::*};

#[allow(clippy::type_complexity)]
fn presolve_test_data() -> (
    CscMatrix<f64>,
    Vec<f64>,
    CscMatrix<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
) {
    let n = 3;
    let P = CscMatrix::identity(n);
    let mut A2 = CscMatrix::identity(n);
    A2.negate();
    let mut A = CscMatrix::vcat(&P, &A2);
    A.scale(2.);

    let c = vec![3., -2., 1.];
    let b = vec![1.; 2 * n];

    let cones = vec![NonnegativeConeT(3), NonnegativeConeT(3)];

    (P, c, A, b, cones)
}

#[test]
fn test_presolve_single_unbounded() {
    let (P, c, A, mut b, cones) = presolve_test_data();

    b[3] = 1e30_f64;

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);
    assert_eq!(solver.variables.z.len(), 5);
    assert_eq!(solver.solution.z[3], 0.);
    assert_eq!(solver.solution.s[3], get_infinity());
}

#[test]
fn test_presolve_completely_redundant_cone() {
    let (P, c, A, mut b, cones) = presolve_test_data();

    b[0] = 1e30_f64;
    b[1] = 1e30_f64;
    b[2] = 1e30_f64;

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);
    assert_eq!(solver.variables.z.len(), 3);
    assert_eq!(solver.solution.z[0..3], vec![0., 0., 0.]);
    let inf = get_infinity();
    assert_eq!(solver.solution.s[0..3], vec![inf, inf, inf]);
    let refsol = vec![-0.5, 2., -0.5];
    assert!(solver.solution.x.dist(&refsol) <= 1e-6);
}

#[test]
fn test_presolve_every_constraint_redundant() {
    let (P, mut c, A, mut b, cones) = presolve_test_data();

    b.fill(1e30_f64);

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);
    assert_eq!(solver.variables.z.len(), 0);
    assert!(solver.solution.x.dist(c.negate()) <= 1e-6);
}

#[test]
fn test_presolve_settable_bound() {
    default_infinity();
    let default_bound = get_infinity();
    set_infinity(1e21_f64);
    assert_eq!(get_infinity(), 1e21_f64);
    default_infinity();
    assert_eq!(get_infinity(), default_bound);
}
