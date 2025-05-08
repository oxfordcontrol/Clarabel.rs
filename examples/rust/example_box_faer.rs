#![allow(non_snake_case)]
use clarabel::algebra::*;
use clarabel::solver::*;

fn problem_data() -> (CscMatrix<f64>, Vec<f64>, CscMatrix<f64>, Vec<f64>) {
    let n = 200;

    let P = CscMatrix::identity(n);

    // construct A = [I; -I]
    let I1 = CscMatrix::<f64>::identity(n);
    let mut I2 = CscMatrix::<f64>::identity(n);
    I2.negate();

    let A = CscMatrix::vcat(&I1, &I2).unwrap();

    let q = vec![1.; n];
    let b = vec![1.; 2 * n];

    (P, q, A, b)
}

fn main() {
    let (P, q, A, b) = problem_data();

    let cones = [NonnegativeConeT(b.len())];

    let settings = DefaultSettingsBuilder::default()
        .direct_solve_method("faer".to_owned())
        .equilibrate_enable(true)
        .max_iter(50)
        .verbose(true)
        .build()
        .unwrap();

    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();

    solver.solve();
}
