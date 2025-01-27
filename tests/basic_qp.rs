#![allow(non_snake_case)]
#[cfg(target_family = "wasm")]
use wasm_bindgen_test::*;

use clarabel::{algebra::*, solver::*};

#[allow(clippy::type_complexity)]
fn basic_qp_data() -> (
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

#[allow(clippy::type_complexity)]
fn basic_qp_data_dual_inf() -> (
    CscMatrix<f64>,
    Vec<f64>,
    CscMatrix<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
) {
    // P = [1. 1;1 1]
    let P = CscMatrix::new(
        2,                    // m
        2,                    // n
        vec![0, 2, 4],        // colptr
        vec![0, 1, 0, 1],     // rowval
        vec![1., 1., 1., 1.], // nzval
    );

    // A = [1. 1;1 0]
    let A = CscMatrix::new(
        2,                // m
        2,                // n
        vec![0, 2, 3],    // colptr
        vec![0, 1, 0],    // rowval
        vec![1., 1., 1.], // nzval
    );

    let c = vec![1., -1.];
    let b = vec![1., 1.];

    let cones = vec![NonnegativeConeT(2)];

    (P, c, A, b, cones)
}

#[test]
fn test_qp_univariate() {
    let P = CscMatrix::identity(1);
    let c = [0.];
    let A = CscMatrix::identity(1);
    let b = [1.];
    let cones = [NonnegativeConeT(1)];

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);

    assert!(f64::abs(solver.solution.x[0]) <= 1e-6);
    assert!(f64::abs(solver.solution.obj_val) <= 1e-6);
    assert!(f64::abs(solver.solution.obj_val_dual) <= 1e-6);
}

#[test]
fn test_qp_feasible() {
    let (P, c, A, b, cones) = basic_qp_data();

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);

    let refsol = vec![0.3, 0.7];
    assert!(solver.solution.x.dist(&refsol) <= 1e-6);

    let refobj = 1.8800000298331538;
    assert!(f64::abs(solver.solution.obj_val - refobj) <= 1e-6);
    assert!(f64::abs(solver.solution.obj_val_dual - refobj) <= 1e-6);
}

#[test]
fn test_qp_primal_infeasible() {
    let (P, c, A, mut b, cones) = basic_qp_data();

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
fn test_qp_dual_infeasible() {
    let (P, c, A, b, cones) = basic_qp_data_dual_inf();

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::DualInfeasible);
    assert!(solver.solution.obj_val.is_nan());
    assert!(solver.solution.obj_val_dual.is_nan());
}

#[test]
fn test_qp_dual_infeasible_ill_cond() {
    let (P, c, _A, _b, _cones) = basic_qp_data_dual_inf();

    // Set A [1, 1];, i.e. a single row sparse matrix
    let A = CscMatrix {
        m: 1,
        n: 2,
        nzval: vec![1., 1.],
        rowval: vec![0, 0],
        colptr: vec![0, 1, 2],
    };

    let cones = vec![NonnegativeConeT(1)];
    let b = vec![1.];

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::DualInfeasible);
    assert!(solver.solution.obj_val.is_nan());
    assert!(solver.solution.obj_val_dual.is_nan());
}

// a minimal test to check that the wasm build is working

#[cfg(target_family = "wasm")]
#[wasm_bindgen_test]
fn test_qp_feasible_wasm() {
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
    test_qp_feasible();
}
