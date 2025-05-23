#![allow(non_snake_case)]

#[cfg(feature = "serde")]
#[test]
fn test_json_io() {
    use clarabel::{algebra::*, solver::*};
    use std::io::{Seek, SeekFrom};

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

    // write the problem to a file
    let mut file = tempfile::tempfile().unwrap();
    solver.save_to_file(&mut file).unwrap();

    // read the problem from the file
    file.seek(SeekFrom::Start(0)).unwrap();
    let mut solver2 = DefaultSolver::<f64>::load_from_file(&mut file, None).unwrap();
    solver2.solve();
    assert_eq!(solver.solution.x, solver2.solution.x);

    // read the problem from the file with custom settings
    file.seek(SeekFrom::Start(0)).unwrap();
    let settings = DefaultSettingsBuilder::default()
        .max_iter(1)
        .build()
        .unwrap();
    let mut solver3 = DefaultSolver::<f64>::load_from_file(&mut file, Some(settings)).unwrap();
    solver3.solve();
    assert_eq!(solver3.solution.status, SolverStatus::MaxIterations);
}
