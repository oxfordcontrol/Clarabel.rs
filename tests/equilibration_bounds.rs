#![allow(non_snake_case)]

use clarabel::{algebra::*, solver::*};

#[allow(clippy::type_complexity)]
fn equilibration_test_data() -> (
    CscMatrix<f64>,
    Vec<f64>,
    CscMatrix<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
) {
    // P = [4. 1;1 2]
    let P = CscMatrix::new(
        2,                    // m
        2,                    // n
        vec![0, 2, 4],        // colptr
        vec![0, 1, 0, 1],     // rowval
        vec![4., 1., 1., 2.], // nzval
    );

    // A = [1. 1;1 0; 0 1]; A = [-A;A]
    let A = CscMatrix::new(
        3,                    // m
        2,                    // n
        vec![0, 2, 4],        //colptr
        vec![0, 1, 0, 2],     //rowval
        vec![1., 1., 1., 1.], //nzva;
    );

    let (mut A1, A2) = (A.clone(), A);
    A1.negate();
    let A = CscMatrix::vcat(&A1, &A2).unwrap();

    let c = vec![1., 1.];
    let b = vec![-1., 0., 0., 1., 0.7, 0.7];

    let cones = vec![NonnegativeConeT(3), NonnegativeConeT(3)];

    (P, c, A, b, cones)
}

#[test]
fn test_equilibrate_lower_bound() {
    let (mut P, c, A, b, cones) = equilibration_test_data();
    let settings = DefaultSettings::default();

    P.nzval[0] = 1e-15;

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings.clone());

    solver.solve();

    let d = &solver.data.equilibration.d;
    let e = &solver.data.equilibration.e;

    assert!(d.minimum() >= settings.equilibrate_min_scaling);
    assert!(e.minimum() >= settings.equilibrate_min_scaling);
    assert!(d.maximum() <= settings.equilibrate_max_scaling);
    assert!(e.maximum() <= settings.equilibrate_max_scaling);
}

#[test]
fn test_equilibrate_upper_bound() {
    let (P, c, mut A, b, cones) = equilibration_test_data();

    A.nzval[0] = 1e+15;

    let settings = DefaultSettingsBuilder::default()
        .max_iter(10)
        .build()
        .unwrap();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings.clone());

    let d = &solver.data.equilibration.d;
    let e = &solver.data.equilibration.e;

    assert!(d.minimum() >= settings.equilibrate_min_scaling);
    assert!(e.minimum() >= settings.equilibrate_min_scaling);
    assert!(d.maximum() <= settings.equilibrate_max_scaling);
    assert!(e.maximum() <= settings.equilibrate_max_scaling);

    // forces poorly converging test for codecov
    solver.solve();
    assert!(solver.solution.status == SolverStatus::MaxIterations);
}

#[test]
fn test_equilibrate_zero_rows() {
    let (P, c, mut A, b, cones) = equilibration_test_data();
    let settings = DefaultSettings::default();

    A.nzval.set(0.0);

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();

    let e = &solver.data.equilibration.e;

    assert!(e.iter().all(|&v| v == 1.));
}
