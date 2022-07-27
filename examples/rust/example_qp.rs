#![allow(non_snake_case)]

use clarabel::algebra::*;
use clarabel::solver::*;

fn problem_data() -> (CscMatrix<f64>, Vec<f64>, CscMatrix<f64>, Vec<f64>) {
    let P: CscMatrix<f64> = CscMatrix::new(
        2,             // m
        2,             // n
        vec![0, 1, 2], // colptr
        vec![0, 1],    // rowval
        vec![6., 4.],  // nzval
    );

    let q = vec![-1., -4.];

    let A: CscMatrix<f64> = CscMatrix::new(
        5,                               // m
        2,                               // n
        vec![0, 3, 6],                   // colptr
        vec![0, 1, 3, 0, 2, 4],          // rowval
        vec![1., 1., -1., -2., 1., -1.], // nzval
    );

    let b = vec![0., 1., 1., 1., 1.];

    (P, q, A, b)
}

fn main() {
    let (P, q, A, b) = problem_data();

    let cones = [ZeroConeT(1), NonnegativeConeT(4)];

    let settings = DefaultSettingsBuilder::default()
        .equilibrate_enable(true)
        .max_iter(50)
        .build()
        .unwrap();

    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);

    solver.solve();
}
