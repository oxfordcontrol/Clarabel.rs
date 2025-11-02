#![allow(non_snake_case)]

#[cfg(feature = "serde")]
use clarabel::{algebra::*, solver::*};
#[cfg(feature = "serde")]
use tempfile::tempdir;

#[cfg(feature = "serde")]
fn test_fileio_roundtrip(ext: &str) {
    let P = CscMatrix {
        m: 1,
        n: 1,
        colptr: vec![0, 1],
        rowval: vec![0],
        nzval: vec![2.0],
    };
    let q = [1.0];
    let A = CscMatrix {
        m: 1,
        n: 1,
        colptr: vec![0, 1],
        rowval: vec![0],
        nzval: vec![-1.0],
    };
    let b = [-2.0];
    let cones = vec![SupportedConeT::NonnegativeConeT(1)];

    let settings = DefaultSettingsBuilder::default().build().unwrap();

    let mut solver = DefaultSolver::<f64>::new(&P, &q, &A, &b, &cones, settings).unwrap();
    solver.solve();

    let dir = tempdir().unwrap();
    let file_path = dir.path().join(format!("problem.{ext}"));
    solver.save_to_file(&file_path).unwrap();

    let mut solver2 = DefaultSolver::<f64>::load_from_file(&file_path, None).unwrap();
    solver2.solve();
    assert_eq!(solver.solution.x, solver2.solution.x);

    let settings = DefaultSettingsBuilder::default()
        .max_iter(1)
        .build()
        .unwrap();
    let mut solver3 = DefaultSolver::<f64>::load_from_file(&file_path, Some(settings)).unwrap();
    solver3.solve();
    assert_eq!(solver3.solution.status, SolverStatus::MaxIterations);
}

#[cfg(feature = "serde")]
#[test]
fn test_json_io() {
    test_fileio_roundtrip("json");
}

#[cfg(feature = "serde")]
#[test]
fn test_cbin_io() {
    test_fileio_roundtrip("cbin");
}
