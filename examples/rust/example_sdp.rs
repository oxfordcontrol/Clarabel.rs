#![allow(non_snake_case)]

fn main() {
    use clarabel::algebra::*;
    use clarabel::solver::*;

    // SDP Example

    let P = CscMatrix::identity(6);

    // A = [1. 1;1 0; 0 1]; A = [-A;A]
    let A = CscMatrix::identity(6);

    let c = vec![0.0; 6];
    let b = vec![-3., 1., 4., 1., 2., 5.];

    let cones = vec![PSDTriangleConeT(3)];

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings);

    solver.solve();

    println!("Solution = {:?}", solver.solution.x);
}
