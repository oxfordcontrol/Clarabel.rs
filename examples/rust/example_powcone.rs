#![allow(non_snake_case)]
use clarabel::algebra::*;
use clarabel::solver::*;

// Power Cone Example
//
//  solve the following power cone problem
//  max  x1^0.6 y^0.4 + x2^0.1
//  s.t. x1, y, x2 >= 0
//       x1 + 2y  + 3x2 == 3
//  which is equivalent to
//  max z1 + z2
//  s.t. (x1, y, z1) in K_pow(0.6)
//       (x2, 1, z2) in K_pow(0.1)
//       x1 + 2y + 3x2 == 3

fn main() {
    let P = CscMatrix::spalloc(6, 6, 0); // For P = 0
    let q = vec![0.0, 0.0, -1.0, 0.0, 0.0, -1.0];

    let A = CscMatrix::new(
        8,                                                      // m
        6,                                                      // n
        vec![0, 2, 4, 5, 7, 9, 10],                             // colptr
        vec![0, 6, 1, 6, 2, 3, 6, 4, 7, 5],                     // rowval
        vec![-1., -1., -1., -2., -1., -1., -3., -1., -1., -1.], // nzval
    );

    let b = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, -1.0];

    let cones = [PowerConeT(0.6), PowerConeT(0.1), ZeroConeT(1), ZeroConeT(1)];

    let settings = DefaultSettings {
        verbose: true,
        max_iter: 100,
        ..DefaultSettings::default()
    };
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);
    solver.solve();
}
