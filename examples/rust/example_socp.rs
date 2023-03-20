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
        vec![0, 0, 1], // colptr
        vec![1],       // rowval
        vec![2.],      // nzval
    );

    let q = vec![0., 0.];

    let A = CscMatrix::new(
        3,              // m
        2,              // n
        vec![0, 1, 2],  // colptr
        vec![1, 2],     // rowval
        vec![-2., -1.], // nzval
    );

    let b = vec![1., -2., -2.];

    let cones = [SecondOrderConeT(3)];

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);

    solver.solve();

    println!("Solution = {:?}", solver.solution.x);
}
