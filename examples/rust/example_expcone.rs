#![allow(non_snake_case)]
use clarabel::algebra::*;
use clarabel::solver::*;

// Exponential Cone Example

// max  x
// s.t. y * exp(x / y) <= z
//      y == 1, z == exp(5)

fn main() {
    let P = CscMatrix::spalloc(3, 3, 0); // For P = 0
    let q = vec![-1., 0., 0.];

    let A = CscMatrix::new(
        5,                           // m
        3,                           // n
        vec![0, 1, 3, 5],            // colptr
        vec![0, 1, 3, 2, 4],         // rowval
        vec![-1., -1., 1., -1., 1.], // nzval
    );

    let b = vec![0., 0., 0., 1., (5f64).exp()];

    let cones = [ExponentialConeT(), ZeroConeT(2)];

    let settings = DefaultSettings {
        verbose: true,
        ..DefaultSettings::default()
    };
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);
    solver.solve();
}
