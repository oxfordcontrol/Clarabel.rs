#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]
use clarabel::{algebra::*, solver::*};

// a collection of tests to ensure that data of
// incompatible dimension won't be accepted

fn api_dim_check_data() -> (
    CscMatrix<f64>,
    Vec<f64>,
    CscMatrix<f64>,
    Vec<f64>,
    Vec<SupportedCone<f64>>,
) {
    let P = CscMatrix::<f64>::spalloc(4, 4, 0);
    let q = vec![0.; 4];
    let A = CscMatrix::<f64>::spalloc(6, 4, 0);
    let b = vec![0.; 6];
    let cones = vec![ZeroConeT(1), NonnegativeConeT(2), NonnegativeConeT(3)];
    (P, q, A, b, cones)
}

#[test]
fn api_dim_check_working() {
    // This example should work because dimensions are
    // all compatible.  All following checks vary one
    // of these sizes to test dimension checks

    let (P, q, A, b, cones) = api_dim_check_data();

    let settings = DefaultSettings::default();
    let _solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);
}

#[test]
#[should_panic]
fn api_dim_check_bad_P() {
    let (_P, q, A, b, cones) = api_dim_check_data();
    let P = CscMatrix::<f64>::spalloc(3, 3, 0);

    let settings = DefaultSettings::default();
    let _solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);
}

#[test]
#[should_panic]
fn api_dim_check_bad_A_rows() {
    let (P, q, _A, b, cones) = api_dim_check_data();
    let A = CscMatrix::<f64>::spalloc(5, 4, 0);

    let settings = DefaultSettings::default();
    let _solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);
}

#[test]
#[should_panic]
fn api_dim_check_bad_A_cols() {
    let (P, q, _A, b, cones) = api_dim_check_data();
    let A = CscMatrix::<f64>::spalloc(6, 3, 0);

    let settings = DefaultSettings::default();
    let _solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);
}

#[test]
#[should_panic]
fn api_dim_check_P_not_square() {
    let (_P, q, A, b, cones) = api_dim_check_data();
    let P = CscMatrix::<f64>::spalloc(4, 3, 0);

    let settings = DefaultSettings::default();
    let _solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);
}

#[test]
#[should_panic]
fn api_dim_check_bad_cones() {
    let (P, q, A, b, _cones) = api_dim_check_data();
    let cones = vec![ZeroConeT(1), NonnegativeConeT(2), NonnegativeConeT(4)];

    let settings = DefaultSettings::default();
    let _solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);
}
