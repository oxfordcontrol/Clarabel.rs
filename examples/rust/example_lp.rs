#![allow(non_snake_case)]

use clarabel::algebra::*;
use clarabel::solver::*;

fn problem_data() -> (CscMatrix<f64>, Vec<f64>, CscMatrix<f64>, Vec<f64>) {
    // an empty P matrix / P = 0
    let P: CscMatrix<f64> = CscMatrix::<f64>::spalloc(2, 2, 0);

    let q = vec![1., -1.];

    //a 2-d box constraint, separated into 4 inequalities.
    //A = [I; -I]
    let A = CscMatrix::new(
        4,                      // m
        2,                      // n
        vec![0, 2, 4],          // colptr
        vec![0, 2, 1, 3],       // rowval
        vec![1., -1., 1., -1.], // nzval
    );

    let b = vec![1.; 4];

    (P, q, A, b)
}

fn main() {
    let (P, q, A, b) = problem_data();

    let cones = [NonnegativeConeT(4)];

    let settings = DefaultSettingsBuilder::default()
        .equilibrate_enable(true)
        .max_iter(50)
        .build()
        .unwrap();

    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);

    solver.solve();
}
