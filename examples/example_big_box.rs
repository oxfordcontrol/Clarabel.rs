#![allow(non_snake_case)]

use clarabel::algebra::*;
use clarabel::solver::*;

fn problem_data() -> (CscMatrix<f64>, Vec<f64>, CscMatrix<f64>, Vec<f64>) {
    let n = 20000;

    let mut P = CscMatrix::<f64>::spalloc(n, n, n);

    for i in 0..n {
        P.colptr[i] = i;
        P.rowval[i] = i;
        P.nzval[i] = 1.;
    }
    P.colptr[n] = n;

    let mut A = CscMatrix::<f64>::spalloc(2 * n, n, 2 * n);

    for i in 0..n {
        A.colptr[i] = 2 * i;
        A.rowval[2 * i] = i;
        A.rowval[2 * i + 1] = i + n;
        A.nzval[2 * i] = 1.;
        A.nzval[2 * i + 1] = -1.
    }
    A.colptr[n] = 2 * n;

    let q = vec![1.; n];
    let b = vec![1.; 2 * n];

    (P, q, A, b)
}

fn main() {
    let (P, q, A, b) = problem_data();

    let cones = [NonnegativeConeT(b.len())];

    let settings = DefaultSettingsBuilder::default()
        .equilibrate_enable(true)
        .max_iter(50)
        .build()
        .unwrap();

    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);

    solver.solve();
}
