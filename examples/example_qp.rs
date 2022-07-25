#![allow(non_snake_case)]

use clarabel::core::*;
use clarabel::implementations::default::*;

fn _problem_data() -> (CscMatrix<f64>, Vec<f64>, CscMatrix<f64>, Vec<f64>) 
{
    let P: CscMatrix<f64> = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 1],
        nzval: vec![6., 4.],
    };

    let q = vec![-1., -4.];

    let A: CscMatrix<f64> = CscMatrix {
        m: 5,
        n: 2,
        colptr: vec![0, 3, 6],
        rowval: vec![0, 1, 3, 0, 2, 4],
        nzval: vec![1., 1., -1., -2., 1., -1.],
    };

    let b = vec![0., 1., 1., 1., 1.];

    (P, q, A, b)
}

fn main() {
    let (P, q, A, b) = _problem_data();

    let cones = [ZeroConeT(1), NonnegativeConeT(4)];

    let settings = DefaultSettingsBuilder::default()
        .equilibrate_enable(true)
        .max_iter(50)
        .build()
        .unwrap();

    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);

    solver.solve();
}
