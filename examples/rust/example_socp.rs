#![allow(non_snake_case)]
use clarabel::algebra::*;
use clarabel::solver::*;

fn main() {
    // SOCP Example

    let P = CscMatrix::from(&[
        [0., 0.], //
        [0., 2.], //
    ]);

    let q = vec![0., 0.];

    let A = CscMatrix::from(&[
        [0., 0.],  //
        [-2., 0.], //
        [0., -1.], //
    ]);

    let b = vec![1., -2., -2.];

    let cones = [SecondOrderConeT(3)];

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);

    solver.solve();

    println!("Solution = {:?}", solver.solution.x);
}
