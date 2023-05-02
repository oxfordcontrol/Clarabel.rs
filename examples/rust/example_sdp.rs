#![allow(non_snake_case)]

fn main() {
    use clarabel::algebra::*;
    use clarabel::solver::*;

    let n = 3;
    let nvec = (n * (n + 1)) >> 1;

    // SDP Example
    let P = CscMatrix::zeros((nvec, nvec));
    let c = vec![1., 0., 1., 0., 0., 1.];

    let sqrt2 = 2f64.sqrt();
    let A = CscMatrix::from(&[
        [-1., 0., 0., 0., 0., 0.],
        [0., -sqrt2, 0., 0., 0., 0.],
        [0., 0., -1., 0., 0., 0.],
        [0., 0., 0., -sqrt2, 0., 0.],
        [0., 0., 0., 0., -sqrt2, 0.],
        [0., 0., 0., 0., 0., -1.],
        [1., 4., 3., 8., 10., 6.],
    ]);

    let mut b = vec![0.0; 6];
    b.push(1.);

    let cones = vec![PSDTriangleConeT(n), ZeroConeT(1)];

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();
}
