#![allow(non_snake_case)]
use clarabel::algebra::*;
use clarabel::solver::*;

// Exponential Cone Example

// max  x
// s.t. y * exp(x / y) <= z
//      y == 1, z == exp(5)

fn main() {
    let P = CscMatrix::zeros((3, 3));
    let q = vec![-1., 0., 0.];

    let A = CscMatrix::from(&[
        [-1., 0., 0.],
        [0., -1., 0.],
        [0., 0., -1.],
        [0., 1., 0.],
        [0., 0., 1.],
    ]);
    let b = vec![0., 0., 0., 1., (5f64).exp()];

    let cones = [ExponentialConeT(), ZeroConeT(2)];

    let settings = DefaultSettings {
        verbose: true,
        ..DefaultSettings::default()
    };
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);
    solver.solve();
}
