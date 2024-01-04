#![allow(non_snake_case)]

use clarabel::{algebra::*, solver::*};

#[allow(clippy::type_complexity)]
fn updating_test_data() -> (
    CscMatrix<f64>,
    Vec<f64>,
    CscMatrix<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
    DefaultSettings<f64>,
) {
    // P = [4. 1;1 2]
    let P = CscMatrix::from(&[
        [4., 1.], //
        [1., 2.], //
    ]);

    // A = [1 0; 0 1]; A = [-A;A]
    let A = CscMatrix::identity(2);

    let (mut A1, A2) = (A.clone(), A);
    A1.negate();
    let A = CscMatrix::vcat(&A1, &A2);

    let q = vec![1.; 2];
    let b = vec![1.; 4];

    let cones = vec![NonnegativeConeT(2), NonnegativeConeT(2)];

    let settings = DefaultSettingsBuilder::default()
        .presolve_enable(false)
        .equilibrate_enable(false)
        .build()
        .unwrap();

    (P, q, A, b, cones, settings)
}

#[test]
fn test_update_P_matrix_form() {
    // original problem
    let (P, q, A, b, cones, settings) = updating_test_data();
    let mut solver1 = DefaultSolver::new(&P, &q, &A, &b, &cones, settings.clone());
    //solver1.solve();

    // change P and re-solve
    let mut P2 = P.clone();
    P2.nzval[0] = 100.;

    // revised original solver
    assert!(solver1.update_P(&P2.to_triu()).is_ok());
    solver1.solve();

    //new solver
    let mut solver2 = DefaultSolver::new(&P2, &q, &A, &b, &cones, settings);
    solver2.solve();

    assert!(solver1.solution.x.dist(&solver2.solution.x) <= 1e-6);
}

#[test]
fn test_update_P_vector_form() {
    // original problem
    let (P, q, A, b, cones, settings) = updating_test_data();
    let mut solver1 = DefaultSolver::new(&P, &q, &A, &b, &cones, settings.clone());
    solver1.solve();

    // change P and re-solve
    let mut P2 = P.clone();
    P2.nzval[0] = 100.;

    // revised original solver
    assert!(solver1.update_P(&P2.to_triu().nzval).is_ok());
    solver1.solve();

    //new solver
    let mut solver2 = DefaultSolver::new(&P2, &q, &A, &b, &cones, settings);
    solver2.solve();

    assert!(solver1.solution.x.dist(&solver2.solution.x) <= 1e-6);
}

#[test]
fn test_update_A_matrix_form() {
    // original problem
    let (P, q, A, b, cones, settings) = updating_test_data();
    let mut solver1 = DefaultSolver::new(&P, &q, &A, &b, &cones, settings.clone());
    //solver1.solve();

    // change A and re-solve
    let mut A2 = A.clone();
    A2.nzval[2] = -1000.;

    // intention was to update the entry at position (1,1).  Did it work?
    assert!(A2.get_entry((1, 1)).unwrap() == -1000.);

    // revised original solver
    assert!(solver1.update_A(&A2).is_ok());
    solver1.solve();

    //new solver
    let mut solver2 = DefaultSolver::new(&P, &q, &A2, &b, &cones, settings);
    solver2.solve();

    assert!(solver1.solution.x.dist(&solver2.solution.x) <= 1e-6);
}

#[test]
fn test_update_A_vector_form() {
    // original problem
    let (P, q, A, b, cones, settings) = updating_test_data();
    let mut solver1 = DefaultSolver::new(&P, &q, &A, &b, &cones, settings.clone());
    solver1.solve();

    // change A and re-solve
    let mut A2 = A.clone();
    A2.nzval[2] = -1000.;

    // intention was to update the entry at position (1,1).  Did it work?
    assert!(A2.get_entry((1, 1)).unwrap() == -1000.);

    // revised original solver
    assert!(solver1.update_A(&A2.nzval).is_ok());
    solver1.solve();

    //new solver
    let mut solver2 = DefaultSolver::new(&P, &q, &A2, &b, &cones, settings);
    solver2.solve();

    assert!(solver1.solution.x.dist(&solver2.solution.x) <= 1e-6);
}

#[test]
fn test_update_q() {
    // original problem
    let (P, q, A, b, cones, settings) = updating_test_data();
    let mut solver1 = DefaultSolver::new(&P, &q, &A, &b, &cones, settings.clone());
    solver1.solve();

    // change 1 and re-solve
    let mut q2 = q.clone();
    q2[0] = 1000.;

    // revised original solver
    assert!(solver1.update_q(&q2).is_ok());
    solver1.solve();

    //new solver
    let mut solver2 = DefaultSolver::new(&P, &q2, &A, &b, &cones, settings);
    solver2.solve();

    assert!(solver1.solution.x.dist(&solver2.solution.x) <= 1e-6);
}

#[test]
fn test_update_b() {
    // original problem
    let (P, q, A, b, cones, settings) = updating_test_data();
    let mut solver1 = DefaultSolver::new(&P, &q, &A, &b, &cones, settings.clone());
    solver1.solve();

    // change 1 and re-solve
    let mut b2 = b.clone();
    b2[0] = 0.;

    // revised original solver
    assert!(solver1.update_b(&b2).is_ok());
    solver1.solve();

    //new solver
    let mut solver2 = DefaultSolver::new(&P, &q, &A, &b2, &cones, settings);
    solver2.solve();

    assert!(solver1.solution.x.dist(&solver2.solution.x) <= 1e-6);
}

#[test]
fn test_update_noops() {
    // original problem
    let (P, q, A, b, cones, settings) = updating_test_data();
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings.clone());
    solver.solve();

    // apply no-op updates to check for crashes
    solver.update_P(&[]).unwrap();
    solver.update_A(&[]).unwrap();
    solver.update_q(&[]).unwrap();
    solver.update_b(&[]).unwrap();

    // try noops in various combinations
    let P2 = P.clone().to_triu();
    let A2 = A.clone();
    let b2 = b.clone();
    let q2 = q.clone();

    solver.update_data(&[], &[], &[], &[]).unwrap();
    solver.update_data(&P2, &[], &A2, &[]).unwrap();
    solver.update_data(&P2.nzval, &[], &A2.nzval, &[]).unwrap();
    solver.update_data(&[], &q2, &[], &b2).unwrap();
    solver.update_data(&P2, &q2, &[], &[]).unwrap();
    solver.update_data(&[], &[], &A2, &b2).unwrap();
}
