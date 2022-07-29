#![allow(non_snake_case)]
use clarabel::algebra::*;
use clarabel::solver::*;

fn main() {
    // QP Example

    // let P = CscMatrix::identity(2);    // For P = I
    // let P = CscMatrix::spalloc(2,2,0); // For P = 0

    let P = CscMatrix::new(
        2,             // m
        2,             // n
        vec![0, 1, 2], // colptr
        vec![0, 1],    // rowval
        vec![6., 4.],  // nzval
    );

    let q = vec![-1., -4.];

    let A = CscMatrix::new(
        5,                               // m
        2,                               // n
        vec![0, 3, 6],                   // colptr
        vec![0, 1, 3, 0, 2, 4],          // rowval
        vec![1., 1., -1., -2., 1., -1.], // nzval
    );

    let b = vec![0., 1., 1., 1., 1.];

    let cones = [ZeroConeT(1), NonnegativeConeT(4)];

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);

    solver.solve();

    println!("Solution = {:?}", solver.solution.x);
}
