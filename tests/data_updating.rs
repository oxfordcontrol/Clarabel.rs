#![allow(non_snake_case)]

use std::iter::zip;

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
    // huge values to ensure equilibration
    // scaling term is small and carries
    // through update
    let P = CscMatrix::from(&[
        [40000., 1.], //
        [1., 20000.], //
    ]);
    let q = vec![10000.; 2];

    // A = [1 0; 0 1]; A = [-A;A]
    let A = CscMatrix::identity(2);

    let (mut A1, A2) = (A.clone(), A);
    A1.negate();
    let A = CscMatrix::vcat(&A1, &A2).unwrap();

    let b = vec![1.; 4];

    let cones = vec![NonnegativeConeT(2), NonnegativeConeT(2)];

    let settings = DefaultSettingsBuilder::default()
        .presolve_enable(false)
        .equilibrate_enable(true)
        .build()
        .unwrap();

    (P, q, A, b, cones, settings)
}

#[test]
fn test_update_P_matrix_form() {
    // original problem
    let (P, q, A, b, cones, settings) = updating_test_data();
    let mut solver1 = DefaultSolver::new(&P, &q, &A, &b, &cones, settings.clone());
    solver1.solve();

    // change P and re-solve
    let mut P2 = P.to_triu();
    P2.nzval[0] = 100.;

    // revised original solver
    assert!(solver1.update_P(&P2).is_ok());
    solver1.solve();

    //new solver
    let mut solver2 = DefaultSolver::new(&P2, &q, &A, &b, &cones, settings);
    solver2.solve();

    assert!(solver1.solution.x.dist(&solver2.solution.x) <= 1e-7);
}

#[test]
fn test_update_P_vector_form() {
    // original problem
    let (P, q, A, b, cones, settings) = updating_test_data();
    let mut solver1 = DefaultSolver::new(&P, &q, &A, &b, &cones, settings.clone());
    solver1.solve();

    // change P and re-solve
    let mut P2 = P.to_triu();
    P2.nzval[0] = 100.;

    // revised original solver
    assert!(solver1.update_P(&P2.nzval).is_ok());
    solver1.solve();

    //new solver
    let mut solver2 = DefaultSolver::new(&P2, &q, &A, &b, &cones, settings);
    solver2.solve();

    assert!(solver1.solution.x.dist(&solver2.solution.x) <= 1e-7);
}

#[test]
fn test_update_P_tuple() {
    // original problem
    let (P, q, A, b, cones, settings) = updating_test_data();
    let mut solver1 = DefaultSolver::new(&P, &q, &A, &b, &cones, settings.clone());
    solver1.solve();

    // revised original solver
    let values = [3., 5.];
    let index = [1, 2];
    let Pdata = zip(&index, &values);
    assert!(solver1.update_P(&Pdata).is_ok());
    solver1.solve();

    //new solver
    let P00 = P.nzval[0];
    let P2 = CscMatrix::from(&[
        [P00, 3.], //
        [0., 5.],  //
    ]);
    let mut solver2 = DefaultSolver::new(&P2, &q, &A, &b, &cones, settings);
    solver2.solve();

    assert!(solver1.solution.x.dist(&solver2.solution.x) <= 1e-7);
}

#[test]
fn test_update_A_matrix_form() {
    // original problem
    let (P, q, A, b, cones, settings) = updating_test_data();
    let mut solver1 = DefaultSolver::new(&P, &q, &A, &b, &cones, settings.clone());
    //solver1.solve();

    // change A and re-solve
    let mut A2 = A;
    A2.nzval[2] = -1000.;

    // intention was to update the entry at position (1,1).  Did it work?
    assert!(A2.get_entry((1, 1)).unwrap() == -1000.);

    // revised original solver
    assert!(solver1.update_A(&A2).is_ok());
    solver1.solve();

    //new solver
    let mut solver2 = DefaultSolver::new(&P, &q, &A2, &b, &cones, settings);
    solver2.solve();

    assert!(solver1.solution.x.dist(&solver2.solution.x) <= 1e-7);
}

#[test]
fn test_update_A_vector_form() {
    // original problem
    let (P, q, A, b, cones, settings) = updating_test_data();
    let mut solver1 = DefaultSolver::new(&P, &q, &A, &b, &cones, settings.clone());
    solver1.solve();

    // change A and re-solve
    let mut A2 = A;
    A2.nzval[2] = -1000.;

    // intention was to update the entry at position (1,1).  Did it work?
    assert!(A2.get_entry((1, 1)).unwrap() == -1000.);

    // revised original solver
    assert!(solver1.update_A(&A2.nzval).is_ok());
    solver1.solve();

    //new solver
    let mut solver2 = DefaultSolver::new(&P, &q, &A2, &b, &cones, settings);
    solver2.solve();

    assert!(solver1.solution.x.dist(&solver2.solution.x) <= 1e-7);
}

#[test]
fn test_update_A_tuple_form() {
    // original problem
    let (P, q, A, b, cones, settings) = updating_test_data();
    let mut solver1 = DefaultSolver::new(&P, &q, &A, &b, &cones, settings.clone());
    solver1.solve();

    // revised original solver
    let values = [0.5, -0.5];
    let index = [1, 2];
    let Adata = zip(&index, &values);
    assert!(solver1.update_A(&Adata).is_ok());
    solver1.solve();

    //new solver
    let mut A2 = A;
    A2.nzval[1] = 0.5;
    A2.nzval[2] = -0.5;
    let mut solver2 = DefaultSolver::new(&P, &q, &A2, &b, &cones, settings);
    solver2.solve();

    assert!(solver1.solution.x.dist(&solver2.solution.x) <= 1e-7);
}

#[test]
fn test_update_q() {
    // original problem
    let (P, q, A, b, cones, settings) = updating_test_data();
    let mut solver1 = DefaultSolver::new(&P, &q, &A, &b, &cones, settings.clone());
    solver1.solve();

    // change 1 and re-solve
    let mut q2 = q;
    q2[1] = 10.;

    // revised original solver
    assert!(solver1.update_q(&q2).is_ok());
    solver1.solve();

    //new solver
    let mut solver2 = DefaultSolver::new(&P, &q2, &A, &b, &cones, settings);
    solver2.solve();

    assert!(solver1.solution.x.dist(&solver2.solution.x) <= 1e-7);
}

#[test]
fn test_update_q_tuple() {
    // original problem
    let (P, q, A, b, cones, settings) = updating_test_data();
    let mut solver1 = DefaultSolver::new(&P, &q, &A, &b, &cones, settings.clone());
    solver1.solve();

    // revised original solver
    let values = [10.];
    let index = [1];
    let qdata = zip(&index, &values);
    assert!(solver1.update_q(&qdata).is_ok());
    solver1.solve();

    //new solver
    let mut q2 = q;
    q2[1] = 10.;
    let mut solver2 = DefaultSolver::new(&P, &q2, &A, &b, &cones, settings);
    solver2.solve();

    assert!(solver1.solution.x.dist(&solver2.solution.x) <= 1e-7);
}

#[test]
fn test_update_b() {
    // original problem
    let (P, q, A, b, cones, settings) = updating_test_data();
    let mut solver1 = DefaultSolver::new(&P, &q, &A, &b, &cones, settings.clone());
    solver1.solve();

    // change 1 and re-solve
    let mut b2 = b;
    b2[0] = 0.;

    // revised original solver
    assert!(solver1.update_b(&b2).is_ok());
    solver1.solve();

    //new solver
    let mut solver2 = DefaultSolver::new(&P, &q, &A, &b2, &cones, settings);
    solver2.solve();

    assert!(solver1.solution.x.dist(&solver2.solution.x) <= 1e-7);
}

#[test]
fn test_update_b_tuple() {
    // original problem
    let (P, q, A, b, cones, settings) = updating_test_data();
    let mut solver1 = DefaultSolver::new(&P, &q, &A, &b, &cones, settings.clone());
    solver1.solve();

    // revised original solver
    let values = [0., 0.];
    let index = [1, 3];
    let bdata = zip(&index, &values);
    assert!(solver1.update_b(&bdata).is_ok());
    solver1.solve();

    //new solver
    let b2 = vec![1., 0., 1., 0.];
    let mut solver2 = DefaultSolver::new(&P, &q, &A, &b2, &cones, settings);
    solver2.solve();

    assert!(solver1.solution.x.dist(&solver2.solution.x) <= 1e-7);
}

#[test]
fn test_update_noops() {
    // original problem
    let (P, q, A, b, cones, settings) = updating_test_data();
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);
    solver.solve();

    // apply no-op updates to check for crashes
    solver.update_P(&[]).unwrap();
    solver.update_A(&[]).unwrap();
    solver.update_q(&[]).unwrap();
    solver.update_b(&[]).unwrap();

    // try noops in various combinations
    let P2 = P.to_triu();
    let A2 = A;
    let b2 = b;
    let b2zip = zip(&[1, 3], &[0., 0.]);
    let q2 = q;

    solver.update_data(&[], &[], &[], &[]).unwrap();
    solver.update_data(&P2, &[], &A2, &[]).unwrap();
    solver.update_data(&P2.nzval, &[], &A2.nzval, &[]).unwrap();
    solver.update_data(&P2, &[], &A2.nzval, &[]).unwrap(); //mixed formats
    solver.update_data(&[], &q2, &[], &b2zip).unwrap(); //mixed formats
    solver.update_data(&P2.nzval, &[], &A2, &b2zip).unwrap(); //mixed formats
    solver.update_data(&[], &q2, &[], &b2).unwrap();
    solver.update_data(&P2, &q2, &[], &[]).unwrap();
    solver.update_data(&[], &[], &A2, &b2).unwrap();
}

#[test]
fn test_fail_on_presolve_enable() {
    // original problem
    let (P, q, A, mut b, cones, mut settings) = updating_test_data();
    settings.presolve_enable = true;
    let solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings.clone());

    // presolve enabled but nothing eliminated
    assert!(solver.is_data_update_allowed());

    // presolved disabled in settings
    b[0] = 1e40;
    settings.presolve_enable = false;
    let solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings.clone());
    assert!(solver.is_data_update_allowed());

    // should be eliminated
    b[0] = 1e40;
    settings.presolve_enable = true;
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings.clone());
    assert!(!solver.is_data_update_allowed());

    // apply no-op updates to check that updates are rejected
    // when presolve is active
    assert!(matches!(
        solver.update_P(&[]).err(),
        Some(DataUpdateError::PresolveIsActive)
    ));
    assert!(matches!(
        solver.update_A(&[]).err(),
        Some(DataUpdateError::PresolveIsActive)
    ));
    assert!(matches!(
        solver.update_b(&[]).err(),
        Some(DataUpdateError::PresolveIsActive)
    ));
    assert!(matches!(
        solver.update_q(&[]).err(),
        Some(DataUpdateError::PresolveIsActive)
    ));
}
